import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Shared_Functions import A_function, B_function, C_function, construct_correlation_matrix, create_zero_curve
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import bisect
import time
import sys

stochastic_terms = []
sofr_calib_fwd_rates = []
agency_calib_fwd_rates = []

def calculate_mbs_price_with_swap_rates(principal, mortgage_interest_rate, mortgage_term_months,
                                        forward_rate_rows_dict, year_frac, agency_curve, spread=0.0, forward_rates_from_zero=None, swap_rate_factor=1):
    zero_curve = agency_curve
    weighted_swap_curve = None
    for i, (term,list) in enumerate(forward_rate_rows_dict.items()):
        if i == 0:
            weighted_swap_curve = interpolate_monthly_rates(list[0], mortgage_term_months, year_frac, mortgage_term_months/12, list[1])
        else:
            weighted_swap_curve = weighted_swap_curve + interpolate_monthly_rates(list[0], mortgage_term_months, year_frac,
                                                            mortgage_term_months / 12, list[1])
    print(f"Forward Rate Rows Dict: {forward_rate_rows_dict}")
    print(forward_rates_from_zero)
    agency_rate_curve = interpolate_monthly_rates(zero_curve, mortgage_term_months, year_frac, mortgage_term_months/12, 1, spread=spread)
    print(f"Zero Curve for Discounting: {agency_rate_curve}")
    mbs_price = 0
    monthly_interest_rate = mortgage_interest_rate/12
    remaining_balance = principal
    prepayment_base_rate = 0.002  # Starting base prepayment rate
    total_monthly_payment = (monthly_interest_rate * principal) / (1 - (1 + monthly_interest_rate)**(-mortgage_term_months))

    for month in range(0, mortgage_term_months):
        if remaining_balance == 0:
            break

        current_swap_rate = swap_rate_factor * weighted_swap_curve[month - 1]

        print(f"Month: {month}")
        interest_payment = remaining_balance * monthly_interest_rate
        total_monthly_payment = min(total_monthly_payment, remaining_balance + interest_payment)
        print(f"Total Monthly Payment: {total_monthly_payment}")
        print(f"Swap Rate: {current_swap_rate}")
        prepayment_rate_adjustment = mortgage_interest_rate - current_swap_rate
        print(f"Prepayment Rate Adjustment: {prepayment_rate_adjustment}")
        prepayment_rate = max(0, prepayment_base_rate + prepayment_rate_adjustment)
        print(f"Prepayment Rate: {prepayment_rate}")
        principal_payment = min(remaining_balance, total_monthly_payment - interest_payment)
        print(f"Interest Payment: {interest_payment}")
        print(f"Principal Payment: {principal_payment}")
        prepayment = min(remaining_balance - principal_payment, remaining_balance * prepayment_rate)
        print(f"Prepayment: {prepayment}")
        total_payment = principal_payment + prepayment
        total_payment = min(total_payment, remaining_balance)  # Prevent overpayment
        print(principal + interest_payment)
        remaining_balance -= total_payment
        print(f"Remaining Balance: {remaining_balance}")
        discount_factor = 1 / np.prod([(1 + (agency_rate_curve[t]) / 12) for t in range(month - 1)])
        mbs_price += ((principal_payment + prepayment + interest_payment) * discount_factor)
        print(f"MBS Price: {mbs_price}")

    return mbs_price
def interpolate_monthly_rates(forward_rate_row, num_months, year_frac, time_ceil, weight, spread=0.0):
    num_quarters = len(forward_rate_row)
    quarterly_terms = np.linspace(year_frac, num_quarters * year_frac, num_quarters)
    monthly_terms = np.linspace(year_frac, time_ceil, num_months)  # 30 years in monthly steps
    interpolator = interp1d(quarterly_terms, forward_rate_row, kind='linear', fill_value="extrapolate")
    monthly_rates = interpolator(monthly_terms)
    adjusted_monthly_rates = (monthly_rates + spread) * weight
    return adjusted_monthly_rates


def LIBOR_Market_Model(
        time_step, maturity, zero_curve_for_modelling, zero_curve_for_mbs_discounting, a_params, b_params, c_params,
        correlation_matrix, N, mortgage_interest_rate=0.07,
        mortgage_principal=1000000, mortgage_term=360, extend_int_coef=1,
        random_seed=42, calibration_type='SOFR', mortgage_swap_term_years_dict=None,
        spread=0.0, vol_factor=1, swap_rate_factor=1, use_CEV=False, alpha_cev=0.2, global_preview_index_list=[0],calculate_mbs_price=True, curve_modelled=''):
    """
    LIBOR Market Model implementation with Monte Carlo simulation and antithetic/quadratic resampling.
    Allows toggling between the standard log-normal SDE and the CEV model.
    """

    plt.figure(figsize=(12, 8))  # Set up a graph for the Monte Carlo paths
    colors = cm.viridis(np.linspace(0, 1, N))  # Set a range of colors for paths
    np.random.seed(random_seed)  # Fix the random seed for reproducibility

    steps = int(maturity / time_step)  # Number of steps per simulation
    term_steps = steps + 1
    time_steps = extend_int_coef * steps + 1
    terms = np.linspace(time_step, maturity, term_steps - 1)  # Array of terms
    terms_B_0 = np.linspace(0, maturity, term_steps)  # Terms for zero-coupon bonds
    times = np.linspace(0, extend_int_coef * maturity, time_steps)  # Array of times
    year_frac = time_step  # Fraction of the year for time_step - can delete/change this just used both terminology earlier on
    B_0 = np.exp(-zero_curve_for_modelling * terms_B_0)  # Zero-coupon bond prices from SOFR curve

    forward_rate_from_zero = np.zeros((steps, time_steps - 1))
    for k in range(steps):
        forward_rate_from_zero[k][0] = (1 / year_frac) * (B_0[k] / B_0[k + 1] - 1)
        if k in global_preview_index_list:
            print(f"Spot for {curve_modelled} {(k + 1)*time_step}Y: {zero_curve_for_modelling[k]}\n...with Initial Forward Rate: {forward_rate_from_zero[k][0]}")
    mbs_prices = []
    all_paths = []
    extend_increment = 0
    for _ in range(N // 2):  # Only N//2 simulations as each has an antithetic pair
        forward_rate_matrix = forward_rate_from_zero.copy()
        antithetic_forward_rate_matrix = forward_rate_from_zero.copy()

        for J in range(0, time_steps - (extend_int_coef - 1)):  # Iterate over all time steps
            # if J % (term_steps - 1) == 0:
                # extend_increment += 1
            # j = J if J < (term_steps - 1) else J - (extend_increment - 1) * (term_steps - 1)

            for k in range(term_steps - 1):  # Iterate over each term
                sum1 = 0 # drift summation
                # j_r = j if J == 0 or k == 0 else j % (k + 1) # not using this with the current assumption of resetting term rate every 3 months seen in loop range below (0, k), otherwise replace 0 with j_R
                T_k = terms[k]
                A_x_k = A_function(times[J], a_params)
                # A_x_k = A_function((maturity * extend_int_coef) - times[J], a_params)
                B_T_k = B_function(T_k, b_params)
                C_t = C_function(times[J], c_params)
                fwd_rate_vol = vol_factor * A_x_k * B_T_k * C_t

                # 0 means SOFR term rates reset every 3 months. If we didn't reset the rate we would want the j_r variable as lower bound so the summation lower bound only includes "rates" that haven't "matured" yet
                for i in range(0, k):
                    T_i = terms[i]
                    A_x_i = A_function(times[J], a_params)
                    # A_x_i = A_function((maturity * extend_int_coef) - times[J], a_params)
                    # We are operating under the assumption that A parameters show the snapshot view of volatility over the entire timeline rather than within a specific term interval
                    B_T_i = B_function(T_i, b_params)
                    sum1 += ((vol_factor * A_x_i * B_T_i * C_t * correlation_matrix[i, k] *
                             (year_frac * forward_rate_matrix[i][J]**(1-alpha_cev))) / (1 + year_frac * forward_rate_matrix[i][J]))

                sum1 *= vol_factor * fwd_rate_vol
                w = np.random.standard_normal()  # Generate Wiener process variable
                ln_fwd_rate = np.log(forward_rate_matrix[k][J])

                if use_CEV:
                    # CEV form in log-space
                    drift_term = (forward_rate_matrix[k][J]**(-alpha_cev)) * sum1 - ((forward_rate_matrix[k][J]**(-2 * alpha_cev)) * (fwd_rate_vol ** 2) * 0.5)
                    stochastic_term = (forward_rate_matrix[k][J]**(-alpha_cev)) * fwd_rate_vol * w * np.sqrt(year_frac)
                else:
                    # Standard log-normal form
                    drift_term = sum1 - ((fwd_rate_vol ** 2) / 2)
                    stochastic_term = fwd_rate_vol * w * np.sqrt(year_frac)

                ln_fwd_rate += (drift_term * year_frac) + stochastic_term
                forward_rate_matrix[k][J + 1] = np.exp(ln_fwd_rate)

                ln_antithetic_fwd_rate = np.log(antithetic_forward_rate_matrix[k][J])
                antithetic_stochastic_term = -stochastic_term

                ln_antithetic_fwd_rate += (drift_term * year_frac) + antithetic_stochastic_term
                antithetic_forward_rate_matrix[k][J + 1] = np.exp(ln_antithetic_fwd_rate)

        all_paths.append(forward_rate_matrix[:, :time_steps - 1])
        all_paths.append(antithetic_forward_rate_matrix[:, :time_steps - 1])


    initial_forward_rate = all_paths[0][:, 0]
    constant_path = np.tile(initial_forward_rate,
                            (all_paths[0].shape[1], 1)).T  # Extend constant forward rate across all time steps
    constant_path_payoff = np.sum(constant_path[-1, :]) # can change resampling to be based on different factors
    stacked_paths = np.stack(all_paths, axis=0)  # Shape: (num_paths, num_terms, num_time_steps)
    mean_path_payoff_per_term = np.mean(stacked_paths, axis=(0, 2))
    path_payoffs_per_term = np.mean(stacked_paths,axis=2)  # Average over time steps for each path and term (shape: num_paths x num_terms)
    abs_diffs = np.abs(path_payoffs_per_term - mean_path_payoff_per_term)
    weights_per_term = 1 / (abs_diffs + 1e-6)
    variance_based_weights = np.mean(weights_per_term,axis=1)
    base_weight = 1 / len(all_paths)
    base_weights = np.full_like(variance_based_weights, base_weight)
    weights =  (0.0 * base_weights + 1.0 * variance_based_weights)/sum(variance_based_weights)


    print(f"Weights: {weights}")

    print(f"Mean Path Payoff: {mean_path_payoff_per_term}")
    resampled_paths = np.zeros(all_paths[0].shape)
    for i, path in enumerate(all_paths):
        resampled_paths += weights[i] * path

    if calculate_mbs_price:
        forward_rate_rows_dict = {}
        for term, weight in mortgage_swap_term_years_dict.items():
                forward_rate_rows_dict[term] = [resampled_paths[int((term / time_step) - 1)], weight]

        mbs_price = calculate_mbs_price_with_swap_rates(
                mortgage_principal, mortgage_interest_rate, mortgage_term,
                forward_rate_rows_dict, year_frac, zero_curve_for_mbs_discounting, spread=spread,
                forward_rates_from_zero=forward_rate_from_zero[:, 0], swap_rate_factor=swap_rate_factor)
        mbs_prices.append(mbs_price)

    t_graph = np.arange(0, time_steps - 1)
    resampled_paths_dict[calibration_type][curve_modelled] = {}
    for global_preview_index in global_preview_index_list:
        term_years = (global_preview_index + 1) * time_step  # Term in years
        resampled_paths_dict[calibration_type][curve_modelled][term_years] = resampled_paths[global_preview_index,
                                                                             :].copy()
        plt.figure(figsize=(14, 8))
        for i, path in enumerate(all_paths):

            plt.plot(t_graph, path[global_preview_index, :], color=colors[i], alpha=0.3, linewidth=2)
        with open(f"Data/Output/LogResampledInfo_{calibration_type}Calibration.txt", 'a') as file:
            file.write(f"Calibration Type: {calibration_type}\n")
            file.write(f"{curve_modelled}{float((global_preview_index + 1) * time_step)}Y")
            file.write(f"Mean Path Payoff: {mean_path_payoff_per_term[global_preview_index]}\n")
            file.write(f"Weights: {weights}\n")
            file.write(f"Resampled Paths: {resampled_paths[global_preview_index, :]}\n")
        average_path = np.mean(all_paths, axis=0)
        # plt.plot(t_graph, average_path[global_preview_index, :], color="purple", linewidth=2, label="Average Path")
        plt.plot(t_graph, constant_path[global_preview_index, :], color="blue", linewidth=2, label="Constant Path", alpha=0.5)
        plt.plot(t_graph, resampled_paths[global_preview_index, :], color="red", linewidth=2, label="Resampled Path")

        plt.title(f"{curve_modelled}{float((global_preview_index + 1)*time_step)}Y Monte Carlo Simulation with Resampling - {calibration_type} Calibration")
        plt.xlabel(f"Time ({12 * time_step} Month Steps)")
        plt.ylabel("Forward Rate")
        plt.legend()
        plt.grid(True)
        plt.show()
    all_paths.clear()

    return resampled_paths, mbs_prices


def plot_resampled_paths_by_term(resampled_paths_dict):
    colors = {'SOFR': 'blue', 'AGENCY': 'green', 'TEST': 'orange'}

    term_curve_dict = {}

    for calibration_type, modelled_dict in resampled_paths_dict.items():
        for curve_modelled, term_dict in modelled_dict.items():
            for term, resampled_path in term_dict.items():
                key = (curve_modelled, term)
                if key not in term_curve_dict:
                    term_curve_dict[key] = {}
                term_curve_dict[key][calibration_type] = resampled_path

    for (curve_modelled, term), calibration_paths in term_curve_dict.items():
        plt.figure(figsize=(14, 8))
        y_min, y_max = float('inf'), float('-inf')  # Initialize for dynamic range calculation

        for calibration_type, resampled_path in calibration_paths.items():
            label = f"{calibration_type} Calibration"
            plt.plot(resampled_path, label=label, color=colors[calibration_type], linewidth=2)

            y_min = min(y_min, resampled_path.min())
            y_max = max(y_max, resampled_path.max())

        plt.ylim(y_min - 0.01, y_max + 0.01)

        plt.title(f"Resampled Paths for {curve_modelled} {term}Y")
        plt.xlabel("Time Steps")
        plt.ylabel("Forward Rate")
        plt.legend()
        plt.grid(True)
        plt.show()


def extract_parameters(lines):
    def find_block(lines, keyword):
        for i, line in enumerate(lines):
            if keyword in line:
                start_index = i
                break
        else:
            raise ValueError(f"{keyword} not found in file")

        block_lines = []
        bracket_count = 0
        for line in lines[start_index:]:
            block_lines.append(line)
            bracket_count += line.count('[')
            bracket_count -= line.count(']')
            if bracket_count == 0 and ']' in line:
                break

        block_text = ' '.join(block_lines)
        import re
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', block_text)
        return np.array([float(num) for num in numbers])

    try:
        A_params = find_block(lines, 'A Params')
        B_params = find_block(lines, 'B Params')
        C_params = find_block(lines, 'C Params')
        theta_params = find_block(lines, 'Theta Params')
    except ValueError as e:
        raise ValueError(f"Error extracting parameters: {e}")

    return A_params, B_params, C_params, theta_params

def read_parameters(output_type):
    file_mapping = {
        'AGENCY': 'Data/Output/AGENCY_calibration.txt',
        'SOFR': 'Data/Output/SOFR_calibration.txt',
        'GOVT': 'Data/Output/GOVT_calibration.txt',
        'TEST': 'Data/Output/TEST_calibration.txt'
    }

    if output_type not in file_mapping:
        raise ValueError(f"Invalid output_type. Expected one of {list(file_mapping.keys())}")

    file_path = file_mapping[output_type]

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    print(f"Loaded {output_type} calibration parameters from {file_path}")
    return extract_parameters(lines)

def main(calibration_type, random_seed):
    a, b, c, theta = read_parameters(calibration_type)
    corr_matrix = construct_correlation_matrix(theta)
    zero_curve_for_modelling = create_zero_curve('SOFR', sofr_curve, maturity, time_step, col_name='SOFR')
    zero_curve_for_agency = create_zero_curve('AGENCY', agency_curve, maturity, time_step, col_name='Agency Spot')

    agency_forwards, null_mbs_prices = LIBOR_Market_Model(time_step, maturity, zero_curve_for_agency, None, a, b, c, corr_matrix, N,
    extend_int_coef=extend_int_coef, random_seed=random_seed, calibration_type=calibration_type, mortgage_swap_term_years_dict=mortgage_swap_term_years_dict,
    spread=spread, use_CEV=True, alpha_cev=alpha_cev, global_preview_index_list=preview_index_list_agency,calculate_mbs_price=False, curve_modelled='AGENCY')

    zero_curve_for_mbs_discounting = agency_forwards[0, :] # take 3 month forward agency curve for discounting

    forward_rate_matrix, mbs_prices = LIBOR_Market_Model(time_step, maturity, zero_curve_for_modelling, zero_curve_for_mbs_discounting, a, b, c, corr_matrix, N, mortgage_interest_rate,
    mortgage_principal=mortgage_principal, mortgage_term=mortgage_term, extend_int_coef=extend_int_coef,
    random_seed=random_seed, calibration_type=calibration_type, mortgage_swap_term_years_dict=mortgage_swap_term_years_dict,
    spread=spread, swap_rate_factor=swap_rate_factor, use_CEV=True, alpha_cev=alpha_cev, global_preview_index_list=preview_index_list,calculate_mbs_price=True, curve_modelled='SOFR')
    np.savetxt('Data/Output/' + calibration_type + '_Forward_Rates.txt', forward_rate_matrix, delimiter=',', fmt='%f')
    np.savetxt('Data/Output/' + calibration_type + '_MBS_Prices.txt', mbs_prices, delimiter=',', fmt='%f')
    return forward_rate_matrix, mbs_prices





"""
Set parameters below:
"""
calibration_types = ['AGENCY','SOFR', 'TEST'] # File header information (pulls calibration file)
resampled_paths_dict = {'SOFR': {},'AGENCY': {},'TEST': {}}
sofr_curve = pd.read_csv('Data/Input/SOFR curve.csv')
agency_curve = pd.read_csv('Data/Input/Agency curve.csv')
test_curve = pd.read_csv('Data/Input/test_curve.csv')
current_time = datetime.now()
seed = 42 # random seed is fixed across simulations for testing purposes
maturity = 10.0  # Highest term necessary to model. i.e. I only care about SOFR 0-10Y
time_step = 0.25  # time step in years used in calibration
extend_int_coef = 3  # means we will forecast to 30 years but our data is 10 years in year_frac yr increments
N = 256  # Number of monte carlo paths, main loop will only run have of these because the other half are antithetic
mortgage_principal = 100
mortgage_interest_rate = 0.07  # 7% mortgage rate
mortgage_term = 360  # 30-year mortgage in months
alpha_cev = 0.85 # assuming (1 - alpha) as the exponent in CEV
# {term: weight}
# mortgage_swap_term_years_dict = {2: 0.561898, 10: 0.438102}
mortgage_swap_term_years_dict = {2: 0.5, 10: 0.5} # Pick the forward curves to create weighted swap curve. Make sure the weights sum to 1.
volatility_scaler = 1 # If you want to scale volatility for test purposes
spread = 0.001 # Add to agency to discount cashflows
swap_rate_factor = 1.5 # what you multiply the weighted swap curve by to account for lack of primary secondary spread in simplified model
preview_index_list = [7, 39] # for 2 and 10 year SOFR
preview_index_list_agency = [0] # for 0.25 Agency
for i in range(len(calibration_types)):
    with open(f"Data/Output/LogResampledInfo_{calibration_types[i]}Calibration.txt", 'w') as file:
        pass  # This clears the file content
    print(f"{calibration_types[i]} Calibration")
    fw_rates = main(calibration_types[i], seed)
plot_resampled_paths_by_term(resampled_paths_dict)


