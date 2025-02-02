import pandas as pd
import numpy as np
import Calibration
import BGMModel
import uuid


def main(calibrate=True, BGM=True):
    sofr_curve = pd.read_csv('Data/Input/SOFR curve.csv')
    agency_curve = pd.read_csv('Data/Input/Agency curve.csv')
    test_curve = pd.read_csv('Data/Input/test_curve.csv')

    max_maturity = 20 # Highest maturity to calibrate (e.g., 20 years)
    time_step = 1/2  # Semi-annual time step, faster than 1/4 considering none of our instruments fall in a quarter year increment that is not also a half year.

    max_maturity_ceil = max_maturity + (2 * time_step)
    selected_maturities = list(np.arange(0, max_maturity_ceil, time_step))
    """
    Set parameters for Calibration below:
    """
    BGMCal = Calibration.BGMCalibration(
        max_maturity=max_maturity, # ~ set above for now
        time_step=time_step,  # ~ set above for now
        verbose=True,  # Enable detailed debugging output
        calibrated_cap_maturities=[0.5, 1.5, 2.5, 3.5, 4.5],  # Maturities to calibrate cap volatilities
        calibrated_swaption_exp_mats=[
            (1.0, 2), (2.0, 2), (3.0, 2), (5.0, 2), (7.0, 2),
            (1.0, 3), (2.0, 3), (3.0, 3), (5.0, 3), (7.0, 3),
            (1.0, 5), (2.0, 5), (3.0, 5), (5.0, 5), (7.0, 5),
            (1.0, 10), (2.0, 10), (3.0, 10), (5.0, 10), (7.0, 10),
            (10.0, 2), (10.0, 3), (10.0, 5), (10.0, 10)
        ],  # Swaption expiration and maturity pairs
        curve_df=sofr_curve,  # Curve to calibrate (e.g., test_curve, agency_curve, or sofr_curve)
        curve_string='SOFR',  # Identifier for curve type (e.g., 'TEST', 'SOFR', or 'AGENCY')
        curve_col_name='SOFR',  # Column in the curve to use for calibration (e.g. SOFR or Agency Spot)
        corr_initial_decay_rate=0.10,  # Initial decay rate for the correlation matrix
        initial_guess=np.array([
                                   0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.01,
                                   # A(x) parameters for TTM 0-15
                                   0.90, 0.3,  # B(T) parameters: base level and lambda
                                   0.90, 0.3,  # C(t) parameters: base level and lambda
                               ] + [0.5, 0.5] * len(selected_maturities)),  # Theta parameters for correlations
        bounds=[
                   (1 / 10000, 1000 / 10000)] * 8 + [(0, 0.15)] +  # Bounds for A(x)
               [(0.8, 1)] + [(0.05, 0.3)] +  # Bounds for B(T)
               [(0.8, 1)] + [(0.05, 0.3)] +  # Bounds for C(t)
               [(-np.pi / 2, np.pi / 2)] * 2 * len(selected_maturities),  # Bounds for Theta
        a_breakpoints=[1, 2, 3, 5, 7, 9, 12, 15],  # Breakpoints for A(x) interpolation
        a_param_smoothness_penalty=True,  # Enable penalty for A(x) smoothness
        a_penalty_threshold=0.0025,  # Threshold for penalizing A(x) parameter jumps to prevent overfitting and exploitation of averages by optimizer
        a_penalty_size=0.003,  # Size of penalty for A(x) parameter jumps
        optimizer_method='L-BFGS-B',  # Optimization method 'TNC' or 'L-BFGS-B'. 'TNC' takes much longer to calibrate to.
        optimizer_options_dict={
            'maxiter': 2000,  # Maximum iterations
            'maxfun': 4000,  # Maximum function evaluations
            'disp': True,  # Display optimization progress
            'ftol': 1e-7,  # Function tolerance
            'gtol': 1e-7,  # Gradient norm tolerance
            'eps': 1e-7,  # Step size for gradient approximation
        },
        atm_strikes={
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
    )


    """
    Set parameters for BGM below:
    """

    bgm_model = BGMModel.BGMModel(
        calibration_types=['AGENCY', 'SOFR', 'TEST'],  # File header information (pulls calibration file), sofr_curve, agency_curve, test_curve,
        sofr_curve=sofr_curve, # for forecasting a 30Y fixed mortgage rate
        agency_curve=agency_curve, # for discounting
        maturity=10.0,  # Highest term necessary to model. i.e. I only care about SOFR 0-10Y
        time_step=0.25, # time step in years used in calibration
        extend_int_coef=3, # means we will forecast to 30 years but our data is 10 years in year_frac yr increments
        N=256, # Number of monte carlo paths, main loop will only run have of these because the other half are antithetic
        mortgage_principal=100,
        mortgage_interest_rate=0.07, # 7% mortgage rate
        mortgage_term=360, # 30-year mortgage in months
        alpha_cev=0.8, # assuming (1 - alpha) as the exponent in CEV
        mortgage_swap_term_years_dict={2: 0.5, 10: 0.5}, # {term: weight} Pick the forward curves to create weighted swap curve. Make sure the weights sum to 1.
        volatility_scaler=1, # If you want to scale volatility for test purposes
        spread=0.001, # Add to agency to discount cashflows
        swap_rate_factor=1.5, # what you multiply the weighted swap curve by to account for lack of primary secondary spread in simplified model
        preview_index_list=[7, 39], # for 2 and 10 year SOFR
        preview_index_list_agency=[0], # for 0.25 Agency
        seed=uuid.uuid4().int & (2**32 - 1) , # random seed is fixed across simulations for testing purposes
        prepayment_base_rate=0.002,
        use_CEV=True, # Determine whether to use CEV lognormal or standard lognormal
        verbose=True
    )


    if calibrate:
        BGMCal.run_calibration()

    if BGM:
        bgm_model.run()

    pass

if __name__ == "__main__":
    main(calibrate=False, BGM=True)