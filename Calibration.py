import datetime
from Shared_Functions import A_function, B_function, C_function, construct_correlation_matrix
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import quad
import bisect
import time
import matplotlib.pyplot as plt

sofr_curve = pd.read_csv('Data/Input/SOFR curve.csv')
agency_curve = pd.read_csv('Data/Input/Agency curve.csv')
test_curve = pd.read_csv('Data/Input/test_curve.csv')
global_mse_data = []
np.set_printoptions(threshold=1000)

"""Select Cap/Swaption Maturities to Calibrate To"""
max_maturity = 20
time_step = 1/2
max_maturity_ceil = max_maturity + (2 * time_step)
selected_maturities = list(np.arange(0, max_maturity_ceil, time_step))
global_swaption_model_vols = []
global_cap_model_vols = []
verbose = True

calibrated_cap_maturities = [0.5, 1.5, 2.5, 3.5, 4.5]

calibrated_swaption_exp_mats = [(1.0, 2), (2.0, 2), (3.0, 2), (5.0, 2), (7.0, 2),
                                (1.0, 3), (2.0, 3), (3.0, 3), (5.0, 3), (7.0, 3),
                                (1.0, 5), (2.0, 5), (3.0, 5), (5.0, 5), (7.0, 5),
                                (1.0, 10), (2.0, 10), (3.0, 10), (5.0, 10), (7.0, 10),
                                (10.0, 2), (10.0, 3), (10.0, 5), (10.0, 10)]


"""Select Curve to Calibrate On"""
global_curve = sofr_curve # agency_curve or sofr_curve
global_curve_string = 'SOFR' # AGENCY or SOFR


"""Select Parameter Initial Guesses, Bounds, and Breakpoints"""
theta_parameter_reduc_denom = 1
decay_rate = 0.10 # for initial guess exponential decay correlation matrix best to pick values betweeen 0.05 and 0.25
initial_guess = np.array([0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0, # A(x) parameters correspond to x = 0 - 15yrs for the 8 A params then exponential constant for x > 15yrs
                          0.90, 0.3, # B(T) parameters, base level, lambda for: B(T) = 1 + (1 - base level) * np.exp(-T * lambda)
                          0.90,0.3] + # C(t) parameters, base level, lambda for: C(t) = base level + (1-base level) * e^(-t * lambda)
                         [0.5, 0.5] * (len(selected_maturities) // theta_parameter_reduc_denom))  # Theta parameters for (np.cos(theta[i, 0]) * np.cos(theta[j, 0]) + np.sin(theta[i, 0]) * np.sin(theta[j, 0]) * np.cos(theta[i, 1] - theta[j, 1]))

bounds = ([(1 / 10000, 1000 / 10000)] * 8 + [(0, 1)] + # A(x) parameter bounds
          [(0.8, 1)] + [(0.05, 0.3)] + # B(T) parameter bounds
          [(0.8, 1)] + [(0.05, 0.3)] + # C(t) parameter bounds
          [(-np.pi / 2, np.pi / 2)] * (2 * (len(selected_maturities) // theta_parameter_reduc_denom))) # Theta bounds

global_a_breakpoints = [1, 2, 3, 5, 7, 9, 10, 15] # What years you want to calibrate the volatility to directly, anything between will be interpolated linearly

"""A Penalty Function Parameters"""
a_param_smoothness_penalty = False # three parameters below only matter if True
a_lambda_reg = 0.00005  # regularization coef
a_threshold = 0.0025  # difference threshold is 0.0025 to keep the optimizer from exploiting volatility averages
a_large_penalty = 100 # large penalty if difference in A params is greater than threshold

"""Optimizer Tolerances/Options"""
global_optimizer_method = 'L-BFGS-B'
global_optimizer_options_dict = {
    'maxiter': 3000, # Maximum number of iterations the optimizer will run. Increasing this value allows the optimizer more attempts to reach convergence. Recommended size for more complex problems: between 1000-5000.
    'maxfun': 6000, # Maximum number of function evaluations allowed. A higher value gives the optimizer more flexibility to explore the function space, useful for larger, complex problems. Recommended size: typically 2x - 3x the max iterations (1000-10000).
    'disp': True, # Display optimization progress. Setting this to True shows information about each iteration.
    # 'xtol': 1e-5, # Tolerance for changes in the parameter values (specific to some algorithms like 'L-BFGS-B' or 'Nelder-Mead'). Smaller values (e.g., 1e-9 or 1e-10) require closer convergence in parameter space, while larger values allow for looser convergence. Recommended range: 1e-6 to 1e-9.
    'ftol': 1e-7, # Tolerance for the change in the objective function to determine convergence. Smaller values (e.g., 1e-7 or 1e-8) lead to more precise convergence but may slow down the process; higher values (1e-4 or 1e-5) may allow faster, less precise results.
    'gtol': 1e-7, # Gradient norm tolerance for convergence, Smaller values (e.g., 1e-8) can increase accuracy by requiring a tighter convergence on the gradient but may also make convergence slower. Typical range: 1e-6 to 1e-8.
    'eps': 1e-7, # Step size used for finite difference calculations in gradient approximation. Smaller values provide a finer gradient approximation but can make the optimization slower. Common values range from 1e-6 to 1e-8 depending on precision needs.
}


# Grabbed from PolyPaths screenshot
atm_strikes_raw = {
    '1Y': 4.85,
    '2Y': 4.34,
    '3Y': 4.08,
    '4Y': 3.93,
    '5Y': 3.85,
    '6Y': 3.79,
    '7Y': 3.77,
    '8Y': 3.75,
    '9Y': 3.74,
    '10Y': 3.74,
    '12Y': 3.74,
    '15Y': 3.75,
    '20Y': 3.72,
    '30Y': 3.54,
}

def interpolate_atm_strikes(atm_strikes_dict):
    years_raw = np.array([float(key.strip('Y')) for key in atm_strikes_dict.keys()])
    strikes_raw = np.array([value for value in atm_strikes_dict.values()])
    spline = CubicSpline(years_raw, strikes_raw)

    atm_strikes_interpolated_keys = list(np.arange(0, max_maturity_ceil, time_step))
    atm_strikes_interpolated_keys = [str(i) + 'Y' for i in atm_strikes_interpolated_keys]
    atm_strikes_interpolated_values = spline([float(key.strip('Y')) for key in atm_strikes_interpolated_keys])
    atm_strikes_global = {}
    for i, key in enumerate(atm_strikes_interpolated_keys):
        atm_strikes_global[key] = atm_strikes_interpolated_values[i]
    return atm_strikes_global

atm_strikes_global = interpolate_atm_strikes(atm_strikes_raw)


def riemann_sum(func, start, end, num_intervals=75):
    step_size = (end - start) / num_intervals
    result = 0.0
    for n in range(num_intervals):
        t = start + (n + 0.5) * step_size
        result += func(t) * step_size
    return result

def smoothness_penalty(a_params):
    """
    Calculate a smoothness penalty for the A parameters.
    The penalty is the sum of squared differences between consecutive A parameters,
    with an additional large penalty if the difference exceeds 0.025.
    """
    differences = np.diff(a_params)
    penalties = np.where(differences > a_threshold, a_large_penalty, differences ** 2)
    return a_lambda_reg * np.sum(penalties)

def getIVCalc(info_tuple, zcpPrc, maturities, a_params, b_params, c_params, fr_from_zero, correlation_matrix, type):
    """
    :param info_tuple:(current time, maturity, market_vol)
    :param zcpPrc: discount zero coupon 1 dollar
    :param maturities: list of maturities
    :param a_params: 8 breakpoints for x between 0 and 15 inclusive and then 2 parameters for exponential for x > 15 where x = T - t
    :param b_params: a base and a lambda for the exponential
    :param c_params: a base and a lambda for the exponential
    :return: analytic approximation of implied volatility
    """
    T0, Tn, bsIV = info_tuple
    variance_sum = 0.0
    curve_impact_list = []
    curve_impact_list2 = []
    if type == 'swaption':
        start = bisect.bisect_left(maturities, T0)  # Get index of first critical time point >= T0
        end = bisect.bisect_left(maturities, Tn)
        wht = [0.0 for _ in zcpPrc]
        temp = sum([time_step * zcpPrc[i] for i in range(start, end)])
        for i in range(start, end):
            wht[i] = time_step * zcpPrc[i] / temp
        swap_rate = sum([wht[i] * fr_from_zero[i][0] for i in range(start, end)])
        for i in range(start, end):
            B_i = B_function(maturities[i], b_params)
            for j in range(start, end):
                B_j = B_function(maturities[j], b_params)
                c = lambda t: (A_function(maturities[i] - t, a_params,breakpoints=global_a_breakpoints) * C_function(t, c_params) * B_i) * \
                              (A_function(maturities[j] - t, a_params, breakpoints=global_a_breakpoints) * C_function(t, c_params) * B_j)

                # integrate from 0 up until expiration
                integrated_covariance = riemann_sum(c, 0, T0) * (1/T0) * correlation_matrix[i, j]
                # integrated_covariance = quad(c, 0, T0)[0] * (1/T0) * correlation_matrix[i, j]
                # sum from expiration to maturity
                # this is essentially one massive weight multiplied by the integrated covariance
                variance_sum += (integrated_covariance * wht[i] * wht[j] * fr_from_zero[i][0] * fr_from_zero[j][0]) / (swap_rate ** 2)
                curve_impact_list.append((wht[i] * wht[j] * fr_from_zero[i][0] * fr_from_zero[j][0]) / (swap_rate ** 2))
                curve_impact_list2.append((integrated_covariance))
        if verbose:
            print(f"{global_curve_string}-Investigation: {curve_impact_list}")
            print(f"{global_curve_string}-Covariance: {curve_impact_list2}")
        return np.sqrt(variance_sum)

    if type == 'caplet':
        f = lambda t: (A_function(Tn - t, a_params, breakpoints=global_a_breakpoints) * C_function(t, c_params) * B_function(Tn, b_params))**2
        # variance_sum_v2 = quad(f, maturities[start], maturities[end])[0] # produces the same result to several decimal places compared to riemann sum but is more computationally expensive
        variance_sum = riemann_sum(f, T0, Tn)
        return np.sqrt((1/Tn) * variance_sum)

    return None

def refine_theta_for_exponential_decay(num_maturities, base_level, decay_rate):
    """
    :param num_maturities: number of maturities in dataset
    :param base_level: set to 1 for simplicity
    :param decay_rate: dictates how quicly the correlation decays between instruments with different maturities
    :return: thetas representative of exponential decay relationship
    """
    theta = np.zeros((num_maturities, 2))
    theta[:, 0] = np.linspace(0, np.pi, num_maturities)  # Spread out the angles
    theta[:, 1] = np.linspace(0, 2 * np.pi, num_maturities)  # Spread out the angles

    for i in range(num_maturities):
        theta[i, 0] = base_level * np.exp(-decay_rate * i)
        theta[i, 1] = base_level * np.exp(-decay_rate * i)

    return theta.flatten()

def calibrate_vol_surface(cap_vols, swaption_vols, zero_curve, maturities, discount_factors, time_step, initial_guess, bounds):
    num_maturities = len(maturities) # attempting without subtracting 1
    base_level = 1.0
    # Start using thetas that produce a correlation matrix with exponential decay assumption
    adjusted_theta = refine_theta_for_exponential_decay(num_maturities//theta_parameter_reduc_denom, base_level, decay_rate)
    initial_guess[-2 * (num_maturities//theta_parameter_reduc_denom):] = adjusted_theta
    steps = int(maturities[-1] / time_step) + 1
    maturity_times = np.linspace(0, maturities[-1], steps)
    zero_curve_filtered = np.interp(maturity_times, maturities, zero_curve)
    B_0 = np.exp(-zero_curve_filtered * maturity_times)
    forward_rate_from_zero = np.zeros((steps - 1, steps - 1))
    for i in range(steps - 1):
        forward_rate_from_zero[i][0] = (1 / time_step) * (B_0[i] / B_0[i + 1] - 1)
    print(f"Curve Used {global_curve_string}: {forward_rate_from_zero[:,0]}")
    def optimization_function(params):
        a_params = params[:9]
        if a_param_smoothness_penalty:
            penalty = smoothness_penalty(a_params)
        else:
            penalty = 0
        calibration_error_value = calibration_error(params, cap_vols, swaption_vols, maturities, discount_factors, time_step, forward_rate_from_zero)

        total_objective = calibration_error_value + penalty
        if verbose:
            print(f"params: {params}")
            print(f"Total Objective (Error + Penalty): {total_objective}, Penalty: {penalty}, Calibration Error: {calibration_error_value}")
            print(f"Ratio: Calibration Error:Penalty {calibration_error_value/(total_objective):.2f}:{penalty/total_objective:.2f}")
        return total_objective

    result = minimize(optimization_function, initial_guess, method=global_optimizer_method, options=global_optimizer_options_dict, bounds=bounds)

    print(f"Optimization Result: {result}")

    return result.x


def calibration_error(params, cap_vols, swaption_vols, maturities, discount_factors, time_step, forward_rate_from_zero):
    a_params = params[:9]  # A(x) parameters
    b_params = params[9:11]  # B(T) parameters
    c_params = params[11:13]  # C(t) parameters
    theta_params = params[13:]  # Theta parameters
    correlation_matrix = construct_correlation_matrix(theta_params)

    # Caps
    cap_error = 0
    global_cap_model_vols.clear()
    for i in range(len(calibrated_cap_maturities)):
        T0_cap = maturities[0]
        Tn_cap = maturities[int(calibrated_cap_maturities[i] * (1/time_step))]  # cap maturity
        market_volatility_cap = cap_vols[cap_vols['TTM'] == Tn_cap]['ATM_Volatility'].values[0]
        cap_model_vol = getIVCalc((T0_cap, Tn_cap, market_volatility_cap), discount_factors, maturities,
                                a_params, b_params, c_params, forward_rate_from_zero, correlation_matrix, 'caplet')
        # More info on page 150/198 of Brigo/Mercurio textbook.
        global_cap_model_vols.append((str(Tn_cap), cap_model_vol, market_volatility_cap))
        cap_error += np.sum(((cap_model_vol - market_volatility_cap) ** 2)) # SSE
        if verbose:
            print(f"Cap Market Volatility for {Tn_cap}: {market_volatility_cap}")
            print(f"Cap Model Volatility for {Tn_cap}: {cap_model_vol}")


    # Swaptions
    # (1, 2, 3, 5, 7, 10) by (2, 3, 5, 10) Swaptions
    swaption_error = 0
    global_swaption_model_vols.clear()
    for i, (expiration, maturity) in enumerate(calibrated_swaption_exp_mats):
        T0_swaption = expiration
        # calculating implied volatility for the period starting T0 and ending Tn
        Tn_swaption = maturity + expiration
        market_volatility_swaption = swaption_vols[swaption_vols['Maturity'] == maturity][str(expiration)].values[0]/10000
        swaption_model_vol = getIVCalc((T0_swaption, Tn_swaption, market_volatility_swaption), discount_factors, maturities,
                            a_params, b_params, c_params, forward_rate_from_zero, correlation_matrix, 'swaption')
        global_swaption_model_vols.append((str(expiration) + " by " + str(maturity),swaption_model_vol, market_volatility_swaption))
        swaption_error += np.sum(((swaption_model_vol - market_volatility_swaption) ** 2)) # SSE
        if verbose:
            print(f"Swaption Market Volatility for {expiration} by {maturity}: {market_volatility_swaption}")
            print(f"Swaption Model Volatility for {expiration} by {maturity}: {swaption_model_vol}")

    return cap_error + swaption_error

def interpolate_cap_volatilities():
    cap_vols = pd.read_csv('Data/Input/CapVolsSOFR_BPTS.csv').rename(columns={'Strike': 'TTM'})
    cap_vols.columns = [cap_vols.columns[0]] + [float(col) for col in cap_vols.columns[1:]]
    print(f"Converted columns: {cap_vols.columns}")
    # Define selected maturities for interpolation
    selected_maturities = np.arange(0, max_maturity_ceil, time_step)

    interpolated_cap_vols = pd.DataFrame({'TTM': selected_maturities})
    # Interpolate cap volatilities across maturities
    for strike in cap_vols.columns[1:]:
        cs_maturities = PchipInterpolator(cap_vols['TTM'], cap_vols[strike])
        interpolated_cap_vols[strike] = cs_maturities(selected_maturities)
    interpolated_cap_vols = np.clip(interpolated_cap_vols, 0, None)
    print(f"Interpolated cap volatilities across maturities:\n{interpolated_cap_vols}")
    # Interpolate ATM volatilities and apply clipping to avoid negative values
    interpolated_atm_vols = pd.DataFrame({'TTM': selected_maturities})
    for maturity in selected_maturities:
        maturity_key = f"{maturity}Y"
        if maturity_key in atm_strikes_global:
            atm_strike = atm_strikes_global[maturity_key]
            print(f"Interpolating for maturity: {maturity}Y, ATM Strike: {atm_strike}")

            strikes = interpolated_cap_vols.columns[1:]
            if atm_strike < min(strikes) or atm_strike > max(strikes):
                print(f"ATM Strike {atm_strike} out of bounds for available strikes.")
                continue  # Skip if the strike is out of bounds

            cs_strikes = PchipInterpolator(strikes,
                                           interpolated_cap_vols.loc[interpolated_cap_vols['TTM'] == maturity].values[
                                               0][1:])
            interpolated_atm_vol = cs_strikes(atm_strike)

            # Apply a lower limit (0) to avoid negative interpolated ATM volatilities
            interpolated_atm_vol = max(interpolated_atm_vol, 0)
            interpolated_atm_vols.loc[interpolated_atm_vols['TTM'] == maturity, 'ATM'] = interpolated_atm_vol
            print(f"Interpolated ATM vol for maturity {maturity}: {interpolated_atm_vol}")
    interpolated_atm_vols = interpolated_atm_vols.rename(columns={'ATM': 'ATM_Volatility'})
    interpolated_atm_vols['ATM_Volatility'] = interpolated_atm_vols['ATM_Volatility'] / 10000
    print(f"Interpolated ATM volatilities:\n{interpolated_atm_vols}")
    with open('Data/Output/InterpolatedATMCapVols.txt', 'w') as f:
        f.write(interpolated_atm_vols.to_string())
    with open('Data/Output/InterpolatedCapVols.txt', 'w') as f:
        f.write(interpolated_cap_vols.to_string())
    return interpolated_cap_vols
def run_calibration(curve, curve_used):
    print("Program started.")
    start_time = time.time()
    interpolate_cap_volatilities()
    interpolated_atm_cap_vols = pd.read_csv('Data/Output/InterpolatedATMCapVols.txt', sep=r'\s+')
    interpolated_swaption_vols = pd.read_csv('Data/Output/InterpolatedSwaptionVols.txt', sep=r'\s+')
    interpolated_atm_cap_vols.index, interpolated_swaption_vols.index = interpolated_atm_cap_vols['TTM'], interpolated_swaption_vols['Maturity']
    if curve_used == "SOFR":
        col_name = "SOFR"
    elif curve_used == 'AGENCY':
        col_name = 'Agency Spot'
    elif curve_used == "TEST":
        col_name = 'Agency Spot'
    sofr_terms = curve['Term'].values
    sofr_rates = curve[col_name].values / 100
    zero_curve_filtered = np.interp(selected_maturities, sofr_terms, sofr_rates)
    discount_factors = np.exp(-zero_curve_filtered * selected_maturities)

    calibrated_params = calibrate_vol_surface(interpolated_atm_cap_vols, interpolated_swaption_vols, zero_curve_filtered,
                                                selected_maturities, discount_factors, time_step, initial_guess, bounds)
    # Summary Report
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken} seconds")
    with open('Data/Output/' + curve_used + '_calibration.txt', 'w') as file:
        file.write(f"Trial Run at: {datetime.datetime.now()}\n")
        file.write(f"Time Taken: {str(time_taken) + str(time.time())}\n")
        file.write(f"Curve Used: {curve_used}\n")
        file.write(
            f"A Params: {np.array2string(calibrated_params[:9], formatter={'float_kind': lambda x: f'{x:.6f}'})}\n")
        file.write(
            f"B Params: {np.array2string(calibrated_params[9:11], formatter={'float_kind': lambda x: f'{x:.6f}'})}\n")
        file.write(
            f"C Params: {np.array2string(calibrated_params[11:13], formatter={'float_kind': lambda x: f'{x:.6f}'})}\n")
        file.write(f"Theta Params: {calibrated_params[13:]}\n")
        file.write(f"Swaption Model,Market Vols:{global_swaption_model_vols}\n")
        file.write(f"Cap Model,Market Vols:{global_cap_model_vols}\n")
    np.savetxt(r'Data\Output\\' + curve_used + 'CorrelationMatrix.txt', construct_correlation_matrix(calibrated_params[13:]), delimiter=', ', fmt='%s')
    return calibrated_params, time_taken


calibrated_params, time_taken= run_calibration(global_curve, global_curve_string)