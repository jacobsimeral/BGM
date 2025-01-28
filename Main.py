import pandas as pd
import numpy as np
import Calibration
from datetime import datetime
import BGMModel



def main(calibrate=True, BGM=True):
    sofr_curve = pd.read_csv('Data/Input/SOFR curve.csv')
    agency_curve = pd.read_csv('Data/Input/Agency curve.csv')
    test_curve = pd.read_csv('Data/Input/test_curve.csv')
    """Select Curve to Calibrate On"""
    curve_df = test_curve  # agency_curve or sofr_curve
    curve_string = 'TEST'  # AGENCY or SOFR
    curve_col_name = 'Agency Spot'  # SOFR or Agency Spot
    # Grabbed from PolyPaths screenshot
    atm_strikes = {
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

    BGMCal = Calibration.BGMCalibration(
        max_maturity=20,  # Highest maturity to calibrate (e.g., 20 years)
        time_step=1 / 2,  # Semi-annual time step
        verbose=True,  # Enable detailed debugging output
        calibrated_cap_maturities=[0.5, 1.5, 2.5, 3.5, 4.5],  # Maturities to calibrate cap volatilities
        calibrated_swaption_exp_mats=[
            (1.0, 2), (2.0, 2), (3.0, 2), (5.0, 2), (7.0, 2),
            (1.0, 3), (2.0, 3), (3.0, 3), (5.0, 3), (7.0, 3),
            (1.0, 5), (2.0, 5), (3.0, 5), (5.0, 5), (7.0, 5),
            (1.0, 10), (2.0, 10), (3.0, 10), (5.0, 10), (7.0, 10),
            (10.0, 2), (10.0, 3), (10.0, 5), (10.0, 10)
        ],  # Swaption expiration and maturity pairs
        curve_df=test_curve,  # Curve to calibrate (e.g., test_curve, agency_curve, or sofr_curve)
        curve_string='TEST',  # Identifier for curve type (e.g., 'TEST', 'SOFR', or 'AGENCY')
        curve_col_name='Agency Spot',  # Column in the curve to use for calibration
        corr_initial_decay_rate=0.10,  # Initial decay rate for the correlation matrix
        initial_guess=np.array([
                                   0.006, 0.002, 0.004, 0.005, 0.002, 0.005, 0.004, 0.007, 0.01,
                                   # A(x) parameters for TTM 0-15
                                   0.90, 0.3,  # B(T) parameters: base level and lambda
                                   0.90, 0.3,  # C(t) parameters: base level and lambda
                               ] + [0.5, 0.5] * 41),  # Theta parameters for correlations
        bounds=[
                   (1 / 10000, 1000 / 10000)] * 8 + [(0, 0.15)] +  # Bounds for A(x)
               [(0.8, 1)] + [(0.05, 0.3)] +  # Bounds for B(T)
               [(0.8, 1)] + [(0.05, 0.3)] +  # Bounds for C(t)
               [(-np.pi / 2, np.pi / 2)] * 82,  # Bounds for Theta
        a_breakpoints=[1, 2, 3, 5, 7, 9, 12, 15],  # Breakpoints for A(x) interpolation
        a_param_smoothness_penalty=True,  # Enable penalty for A(x) smoothness
        a_penalty_threshold=0.0020,  # Threshold for penalizing A(x) parameter jumps
        a_penalty_size=0.03,  # Size of penalty for A(x) parameter jumps
        optimizer_method='L-BFGS-B',  # Optimization method
        optimizer_options_dict={
            'maxiter': 2000,  # Maximum iterations
            'maxfun': 4000,  # Maximum function evaluations
            'disp': True,  # Display optimization progress
            'ftol': 1e-7,  # Function tolerance
            'gtol': 1e-7,  # Gradient norm tolerance
            'eps': 1e-7,  # Step size for gradient approximation
        },
        atm_strikes=atm_strikes  # ATM strikes for interpolation
    )


    """
    Set parameters for BGM below:
    """
    calibration_types = ['AGENCY', 'SOFR', 'TEST']  # File header information (pulls calibration file)

    bgm_model = BGMModel.BGMModel(calibration_types, sofr_curve, agency_curve, test_curve,
                         maturity=10.0, time_step=0.25, extend_int_coef=3, N=256,
                         mortgage_principal=100, mortgage_interest_rate=0.07,
                         mortgage_term=360, alpha_cev=0.85,
                         mortgage_swap_term_years_dict={2: 0.5, 10: 0.5},
                         volatility_scaler=1, spread=0.001, swap_rate_factor=1.5,
                         preview_index_list=[7, 39], preview_index_list_agency=[0], seed=42, prepayment_base_rate=0.002, use_CEV=True)


    if calibrate:
        BGMCal.run_calibration(curve_df, curve_string, curve_col_name)

    if BGM:
        bgm_model.run()

    pass

if __name__ == "__main__":
    main(calibrate=False, BGM=True)