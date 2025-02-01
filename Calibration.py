import datetime

from win32comext.adsi.demos.scp import verbose

from Shared_Functions import A_function, B_function, C_function, construct_correlation_matrix
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import quad
import bisect
import time
import matplotlib.pyplot as plt
np.set_printoptions(threshold=1000)




class BGMCalibration:
    """
        A class to handle the calibration of caplet and swaption volatilities, using A, B, and C parameter models
        to fit interest rate curves and volatilities.

        Attributes
        ----------
        sofr_curve : pd.DataFrame
            DataFrame containing the SOFR curve data.
        agency_curve : pd.DataFrame
            DataFrame containing the agency curve data.
        test_curve : pd.DataFrame
            DataFrame containing the test curve data.
        atm_strikes : dict
            A dictionary of ATM strike values by tenor (e.g., {'1Y': 4.85, '2Y': 4.34, ...}).
        max_maturity : int
            The highest maturity (in years) to calibrate to (e.g., 20 years).
        time_step : float
            The time step (in years) used for calibration (e.g., 0.5 for semi-annual steps).
        verbose : bool
            Whether to enable detailed print statements for debugging (default is True).
        calibrated_cap_maturities : list of float
            Maturities (in years) of the cap volatilities to calibrate (e.g., [0.5, 1.5, 2.5, ...]).
        calibrated_swaption_exp_mats : list of tuple
            List of (expiration, maturity) pairs to calibrate for swaptions.
        a_breakpoints : list of int
            Breakpoints (in years) for A(x) parameters, used for volatility interpolation (e.g., [1, 2, 3, 5, 7, 9, 12, 15]).
        selected_maturities : list of float
            A list of maturities (in years) including extended time intervals for calibration.
        curve_df : pd.DataFrame
            The DataFrame containing the selected curve to calibrate on (e.g., agency_curve, sofr_curve, etc.).
        curve_string : str
            A string representing the curve type (e.g., 'TEST', 'AGENCY', 'SOFR').
        curve_col_name : str
            The column name in the curve DataFrame to use for the calibration (e.g., 'SOFR', 'Agency Spot').
        corr_initial_decay_rate : float
            Initial decay rate for the exponential decay of the correlation matrix (recommended range: 0.05 to 0.25).
        initial_guess : np.ndarray
            Initial guesses for the optimizer, including A, B, C, and Theta parameters.
            - A(x) parameters: for x = 0 to 15 years, with an exponential constant for x > 15 years.
            - B(T) parameters: base level and lambda for B(T) = 1 + (1 - base level) * exp(-T * lambda).
            - C(t) parameters: base level and lambda for C(t) = base level + (1 - base level) * exp(-t * lambda).
            - Theta parameters: cosine and sine components for constructing the correlation matrix.
        bounds : list of tuple
            Bounds for the optimizer, specifying ranges for A, B, C, and Theta parameters.
        a_param_smoothness_penalty : bool
            Whether to apply a smoothness penalty to A parameters during optimization (default is True).
        a_penalty_threshold : float
            Threshold for differences between consecutive A parameters to avoid exploiting volatility averages.
        a_penalty_size : float
            The penalty applied if differences between A parameters exceed the threshold.
        optimizer_method : str
            Optimization method to use (e.g., 'L-BFGS-B' or 'TNC'). Note: 'TNC' may take significantly longer.
        optimizer_options_dict : dict
            Options for the optimizer, including:
            - maxiter: Maximum number of iterations (default: 2000).
            - maxfun: Maximum number of function evaluations (default: 4000).
            - disp: Whether to display progress during optimization (default: True).
            - ftol: Tolerance for the objective function (default: 1e-7).
            - gtol: Gradient norm tolerance for convergence (default: 1e-7).
            - eps: Step size for finite difference gradient approximation (default: 1e-7).
"""

    def __init__(self, max_maturity, time_step, verbose, calibrated_cap_maturities, calibrated_swaption_exp_mats,
                 curve_df, curve_string, curve_col_name, corr_initial_decay_rate, initial_guess, bounds, a_breakpoints,
                 a_param_smoothness_penalty, a_penalty_threshold, a_penalty_size, optimizer_method, optimizer_options_dict, atm_strikes ):
        self.max_maturity = max_maturity
        self.time_step = time_step
        self.max_maturity_ceil = self.max_maturity + (2 * self.time_step)
        self.selected_maturities = list(np.arange(0, self.max_maturity_ceil, self.time_step))
        self.verbose = verbose
        self.calibrated_cap_maturities = calibrated_cap_maturities
        self.calibrated_swaption_exp_mats = calibrated_swaption_exp_mats
        self.curve_df = curve_df
        self.curve_string = curve_string
        self.curve_col_name = curve_col_name
        self.corr_initial_decay_rate = corr_initial_decay_rate
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.a_breakpoints = a_breakpoints
        self.a_param_smoothness_penalty = a_param_smoothness_penalty
        self.a_penalty_threshold = a_penalty_threshold
        self.a_penalty_size = a_penalty_size
        self.optimizer_method = optimizer_method
        self.optimizer_options_dict = optimizer_options_dict
        self.atm_strikes = atm_strikes
        self.global_mse_data = []
        self.global_cap_model_vols = []
        self.global_swaption_model_vols = []
        pass


    def interpolate_atm_strikes(self, atm_strikes_dict):
        years_raw = np.array([float(key.strip('Y')) for key in atm_strikes_dict.keys()])
        strikes_raw = np.array([value for value in atm_strikes_dict.values()])
        spline = CubicSpline(years_raw, strikes_raw)

        atm_strikes_interpolated_keys = list(np.arange(0, self.max_maturity_ceil, self.time_step))
        atm_strikes_interpolated_keys = [str(i) + 'Y' for i in atm_strikes_interpolated_keys]
        atm_strikes_interpolated_values = spline([float(key.strip('Y')) for key in atm_strikes_interpolated_keys])
        atm_strikes_global = {}
        for i, key in enumerate(atm_strikes_interpolated_keys):
            atm_strikes_global[key] = atm_strikes_interpolated_values[i]
        return atm_strikes_global



    def riemann_sum(self, func, start, end, num_intervals=75):
        step_size = (end - start) / num_intervals
        result = 0.0
        for n in range(num_intervals):
            t = start + (n + 0.5) * step_size
            result += func(t) * step_size
        return result

    def smoothness_penalty(self, a_params):
        """
        Calculate a smoothness penalty for the A parameters.
        The penalty is the sum of squared differences between consecutive A parameters,
        with an additional large penalty if the difference exceeds 0.025.
        """
        differences = np.diff(a_params)
        penalties = np.where(differences > self.a_penalty_threshold, self.a_penalty_size, differences ** 2)
        return np.sum(penalties)

    def getIVCalc(self, info_tuple, zcpPrc, maturities, a_params, b_params, c_params, fr_from_zero, correlation_matrix, type):
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
        investigation_list = []
        covariance_list = []
        if type == 'swaption':
            start = bisect.bisect_left(maturities, T0)  # Get index of first critical time point >= T0
            end = bisect.bisect_left(maturities, Tn)
            wht = [0.0 for _ in zcpPrc]
            temp = sum([self.time_step * zcpPrc[i] for i in range(start, end)])
            for i in range(start, end):
                wht[i] = self.time_step * zcpPrc[i] / temp
            swap_rate = sum([wht[i] * fr_from_zero[i][0] for i in range(start, end)])
            for i in range(start, end):
                B_i = B_function(maturities[i], b_params)
                for j in range(start, end):
                    B_j = B_function(maturities[j], b_params)
                    c = lambda t: (A_function(maturities[i] - t, a_params,breakpoints=self.a_breakpoints) * C_function(t, c_params) * B_i) * \
                                  (A_function(maturities[j] - t, a_params, breakpoints=self.a_breakpoints) * C_function(t, c_params) * B_j)

                    # integrate from 0 up until expiration. Riemann sum provided similar result to quadrature integration out to several decimal places
                    integrated_covariance = self.riemann_sum(c, 0, T0) * (1/T0) * correlation_matrix[i, j]
                    # integrated_covariance = quad(c, 0, T0)[0] * (1/T0) * correlation_matrix[i, j]
                    # sum from expiration to maturity
                    # this is essentially a matrix of weights multplied by their counterpart in matrix of covariances
                    variance_sum += (integrated_covariance * wht[i] * wht[j] * fr_from_zero[i][0] * fr_from_zero[j][0]) / (swap_rate ** 2)
                    investigation_list.append((wht[i] * wht[j] * fr_from_zero[i][0] * fr_from_zero[j][0]) / (swap_rate ** 2)) # these are for debugging output
                    covariance_list.append((integrated_covariance)) # these are for debugging output
            if self.verbose:
                print(f"{self.curve_string}-Investigation: {investigation_list}")
                print(f"{self.curve_string}-Covariance: {covariance_list}")
            return np.sqrt(variance_sum)

        if type == 'caplet':
            f = lambda t: (A_function(Tn - t, a_params, breakpoints=self.a_breakpoints) * C_function(t, c_params) * B_function(Tn, b_params))**2
            # variance_sum_v2 = quad(f, maturities[start], maturities[end])[0] # produces the same result to several decimal places compared to riemann sum but is more computationally expensive
            variance_sum = self.riemann_sum(f, T0, Tn)
            return np.sqrt((1/Tn) * variance_sum)

        return None

    def refine_theta_for_exponential_decay(self, num_maturities, base_level, decay_rate):
        """
        :param num_maturities: number of maturities in dataset
        :param base_level: set to 1 for simplicity
        :param decay_rate: dictates how quicly the correlation decays between instruments with different maturities
        :return: thetas representative of exponential decay relationship
        """
        theta = np.zeros((num_maturities, 2))
        theta[:, 0] = np.linspace(0, np.pi, num_maturities)
        theta[:, 1] = np.linspace(0, 2 * np.pi, num_maturities)

        for i in range(num_maturities):
            theta[i, 0] = base_level * np.exp(-decay_rate * i)
            theta[i, 1] = base_level * np.exp(-decay_rate * i)

        return theta.flatten()

    def calibrate_vol_surface(self, cap_vols, swaption_vols, spot_curve, maturities, discount_factors, time_step, initial_guess, bounds):
        num_maturities = len(maturities)
        base_level = 1.0
        # Start using thetas that produce a correlation matrix with exponential decay assumption
        adjusted_theta = self.refine_theta_for_exponential_decay(num_maturities, base_level, self.corr_initial_decay_rate)
        initial_guess[-2 * (num_maturities):] = adjusted_theta
        steps = int(maturities[-1] / time_step) + 1
        maturity_times = np.linspace(0, maturities[-1], steps)
        cubic_spline = CubicSpline(maturities, spot_curve)
        spot_curve_filtered = cubic_spline(maturity_times)
        B_0 = np.exp(-spot_curve_filtered * maturity_times)
        forward_rate_from_zero = np.zeros((steps - 1, steps - 1))
        for i in range(steps - 1):
            forward_rate_from_zero[i][0] = (1 / time_step) * (B_0[i] / B_0[i + 1] - 1)
        if verbose:
            print(f"Interpolated Spot Curve: {spot_curve_filtered}")
            print(f"Zero Curve: {B_0}")
            print(f"Initial Forwards {self.curve_string}: {forward_rate_from_zero[:,0]}")
        def optimization_function(params):
            a_params = params[:9]

            if self.a_param_smoothness_penalty:
                penalty = self.smoothness_penalty(a_params)
            else:
                penalty = 0
            calibration_error_value = self.calibration_error(params, cap_vols, swaption_vols, maturities, discount_factors, time_step, forward_rate_from_zero)

            total_objective = calibration_error_value + penalty
            if self.verbose:
                print(f"params: {params}")
                print(f"Total Objective (Error + Penalty): {total_objective}, Penalty: {penalty}, Calibration Error: {calibration_error_value}")
                print(f"Ratio: Calibration Error:Penalty {calibration_error_value/(total_objective):.2f}:{penalty/total_objective:.2f}")
            return total_objective
        result = minimize(optimization_function, self.initial_guess, method=self.optimizer_method, options=self.optimizer_options_dict, bounds=bounds)

        print(f"Optimization Result: {result}")

        return result.x


    def calibration_error(self, params, cap_vols, swaption_vols, maturities, discount_factors, time_step, forward_rate_from_zero):
        a_params = params[:9]  # A(x) parameters
        b_params = params[9:11]  # B(T) parameters
        c_params = params[11:13]  # C(t) parameters
        theta_params = params[13:]  # Theta parameters
        correlation_matrix = construct_correlation_matrix(theta_params)

        # Caps
        cap_error = 0
        self.global_cap_model_vols.clear()
        for i in range(len(self.calibrated_cap_maturities)):
            T0_cap = maturities[0]
            Tn_cap = maturities[int(self.calibrated_cap_maturities[i] * (1/time_step))]  # cap maturity
            market_volatility_cap = cap_vols[cap_vols['TTM'] == Tn_cap]['ATM_Volatility'].values[0]
            cap_model_vol = self.getIVCalc((T0_cap, Tn_cap, market_volatility_cap), discount_factors, maturities,
                                    a_params, b_params, c_params, forward_rate_from_zero, correlation_matrix, 'caplet')
            # More info on page 150/198 of Brigo/Mercurio textbook.
            self.global_cap_model_vols.append((str(Tn_cap), cap_model_vol, market_volatility_cap))
            cap_error += np.sum(((cap_model_vol - market_volatility_cap) ** 2)) # SSE
            if self.verbose:
                print(f"Cap Market Volatility for {Tn_cap}: {market_volatility_cap}")
                print(f"Cap Model Volatility for {Tn_cap}: {cap_model_vol}")


        # Swaptions
        # (1, 2, 3, 5, 7, 10) by (2, 3, 5, 10) Swaptions
        swaption_error = 0
        self.global_swaption_model_vols.clear()
        for i, (expiration, maturity) in enumerate(self.calibrated_swaption_exp_mats):
            T0_swaption = expiration
            # calculating implied volatility for the period starting T0 and ending Tn
            Tn_swaption = maturity + expiration
            market_volatility_swaption = swaption_vols[swaption_vols['Maturity'] == maturity][str(expiration)].values[0]/10000
            swaption_model_vol = self.getIVCalc((T0_swaption, Tn_swaption, market_volatility_swaption), discount_factors, maturities,
                                a_params, b_params, c_params, forward_rate_from_zero, correlation_matrix, 'swaption')
            self.global_swaption_model_vols.append((str(expiration) + " by " + str(maturity),swaption_model_vol, market_volatility_swaption))
            swaption_error += np.sum(((swaption_model_vol - market_volatility_swaption) ** 2)) # SSE
            if self.verbose:
                print(f"Swaption Market Volatility for {expiration} by {maturity}: {market_volatility_swaption}")
                print(f"Swaption Model Volatility for {expiration} by {maturity}: {swaption_model_vol}")

        return cap_error + swaption_error

    def interpolate_cap_volatilities(self, atm_strikes_interpolated):
        cap_vols = pd.read_csv('Data/Input/CapVolsSOFR_BPTS.csv').rename(columns={'Strike': 'TTM'})
        cap_vols.columns = [cap_vols.columns[0]] + [float(col) for col in cap_vols.columns[1:]]
        selected_maturities = np.arange(0, self.max_maturity_ceil, self.time_step)
        interpolated_cap_vols = pd.DataFrame({'TTM': selected_maturities})
        for strike in cap_vols.columns[1:]:
            cs_maturities = PchipInterpolator(cap_vols['TTM'], cap_vols[strike])
            interpolated_cap_vols[strike] = cs_maturities(selected_maturities)
        interpolated_cap_vols = np.clip(interpolated_cap_vols, 0, None)
        interpolated_atm_vols = pd.DataFrame({'TTM': selected_maturities})
        for maturity in selected_maturities:
            maturity_key = f"{maturity}Y"
            if maturity_key in atm_strikes_interpolated:
                atm_strike = atm_strikes_interpolated[maturity_key]

                strikes = interpolated_cap_vols.columns[1:]
                if atm_strike < min(strikes) or atm_strike > max(strikes):
                    print(f"ATM Strike {atm_strike} out of bounds for available strikes.")
                    continue

                cs_strikes = PchipInterpolator(strikes,
                                               interpolated_cap_vols.loc[interpolated_cap_vols['TTM'] == maturity].values[
                                                   0][1:])
                interpolated_atm_vol = cs_strikes(atm_strike)

                interpolated_atm_vol = max(interpolated_atm_vol, 0)
                interpolated_atm_vols.loc[interpolated_atm_vols['TTM'] == maturity, 'ATM'] = interpolated_atm_vol
        interpolated_atm_vols = interpolated_atm_vols.rename(columns={'ATM': 'ATM_Volatility'})
        interpolated_atm_vols['ATM_Volatility'] = interpolated_atm_vols['ATM_Volatility'] / 10000
        with open('Data/Output/InterpolatedATMCapVols.txt', 'w') as f:
            f.write(interpolated_atm_vols.to_string())
        with open('Data/Output/InterpolatedCapVols.txt', 'w') as f:
            f.write(interpolated_cap_vols.to_string())
        return interpolated_cap_vols



    def run_calibration(self):
        print("Program started.")
        start_time = time.time()
        atm_strikes_interpolated = self.interpolate_atm_strikes(self.atm_strikes)
        self.interpolate_cap_volatilities(atm_strikes_interpolated)
        interpolated_atm_cap_vols = pd.read_csv('Data/Output/InterpolatedATMCapVols.txt', sep=r'\s+')
        interpolated_swaption_vols = pd.read_csv('Data/Output/InterpolatedSwaptionVols.txt', sep=r'\s+')
        interpolated_atm_cap_vols.index, interpolated_swaption_vols.index = interpolated_atm_cap_vols['TTM'], interpolated_swaption_vols['Maturity']
        sofr_terms = self.curve_df['Term'].values
        sofr_rates = self.curve_df[self.curve_col_name].values / 100
        zero_curve_filtered = np.interp(self.selected_maturities, sofr_terms, sofr_rates)
        discount_factors = np.exp(-zero_curve_filtered * self.selected_maturities)

        calibrated_params = self.calibrate_vol_surface(interpolated_atm_cap_vols, interpolated_swaption_vols, zero_curve_filtered,
                                                    self.selected_maturities, discount_factors, self.time_step, self.initial_guess, self.bounds)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken: {time_taken} seconds")
        with open('Data/Output/' + self.curve_string + '_calibration.txt', 'w') as file:
            file.write(f"Trial Run at: {datetime.datetime.now()}\n")
            file.write(f"Time Taken: {str(time_taken) + str(time.time())}\n")
            file.write(f"Curve Used: {self.curve_string}\n")
            file.write(
                f"A Params: {np.array2string(calibrated_params[:9], formatter={'float_kind': lambda x: f'{x:.6f}'})}\n")
            file.write(
                f"B Params: {np.array2string(calibrated_params[9:11], formatter={'float_kind': lambda x: f'{x:.6f}'})}\n")
            file.write(
                f"C Params: {np.array2string(calibrated_params[11:13], formatter={'float_kind': lambda x: f'{x:.6f}'})}\n")
            file.write(f"Theta Params: {calibrated_params[13:]}\n")
            file.write(f"Swaption Model,Market Vols:{self.global_swaption_model_vols}\n")
            file.write(f"Cap Model,Market Vols:{self.global_cap_model_vols}\n")
        np.savetxt(r'Data\Output\\' + self.curve_string + 'CorrelationMatrix.txt', construct_correlation_matrix(calibrated_params[13:]), delimiter=', ', fmt='%s')
        return calibrated_params, time_taken


