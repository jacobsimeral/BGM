import numpy as np
from scipy.interpolate import interp1d

def construct_correlation_matrix(theta):
    """
    :param theta: set parameters containing 2 angles for each maturity
    :return: correlation matrix based on cos(theta[i0])cos(theta[j0]) + sin(theta[i0])sin(theta[j0])cos(theta[i1] - theta[j2])
    where j is the current time and i is maturity and i - j is the difference between maturity and current time
    """
    num_maturities = len(theta) // 2
    theta = theta.reshape((num_maturities, 2))
    correlation_matrix = np.zeros((num_maturities, num_maturities))
    for i in range(num_maturities):
        for j in range(num_maturities):
            correlation_matrix[i, j] = (np.cos(theta[i, 0]) * np.cos(theta[j, 0]) +
                                        np.sin(theta[i, 0]) * np.sin(theta[j, 0]) * np.cos(theta[i, 1] - theta[j, 1]))
    return correlation_matrix

def A_function(x, a_params, breakpoints=[1, 2, 3, 5, 7, 9, 10, 15]):
    """
    :param x:  T - t, or maturity T - current time t
    :param a_params: 8 breakpoints for x between 0 and 15 inclusive and then 2 parameters for exponential for x > 15
    :return: interpolated x if 0 <= 0 <= 15 or base * e^(-lambda * (x-15)) if x > 15
    """
    a_values = a_params[:len(a_params) - 1]
    x_breakpoints = np.array(breakpoints)
    try:
        if x < 0:
            print("X values for A param must be non-negative")
            raise Exception
        if x <= 15:
            return np.interp(x, x_breakpoints, a_values)
        else:
            # return a_params[-2] * np.exp(-a_params[-1] * (x - 15)) # this is the method poly uses in documentation but we chose to keep at the A(15) value
            return a_params[-2]
    except Exception as e:
        print(e)
        exit()


def B_function(T, b_params):
    """
    :param T: maturity
    :param b_params: a base and a lambda for the exponential decay
    :return: 1 + (1 - b_params[0]) * np.exp(-T * b_params[1])
    """
    return 1 + (1 - b_params[0]) * np.exp(-T * b_params[1])

def C_function(t, c_params):
    """
    :param t: current time
    :param c_params: a base and a lambda for the exponential growth
    :return: c_params[0] + (1 - c_params[0]) * (1 - np.exp(-t * c_params[1]))
    """
    return c_params[0] + (1 - c_params[0]) * (1 - np.exp(-t * c_params[1]))

def create_zero_curve(curve_used, curve, max_maturity, time_step, calibration=False, alternate=False):
    """
    :param curve_used: for determining the column name in the dataframe and for naming conventions elsewhere
    :param curve: the curve dataframe with a column for term and for rate
    :param max_maturity: if you want to forecast 10 yrs put 10
    :param time_step: in years. For half-year increments put 0.5
    :param calibration: boolean to determine whether you are calibrating or running model
    :return: zero curve interpolated and filtered
    """
    new_ttm = np.arange(0, 30 + time_step, time_step)
    selected_maturities = np.arange(0, max_maturity + time_step, time_step)
    selected_indices = [i for i, ttm in enumerate(selected_maturities) if ttm in selected_maturities]
    # This is some really garbage code. This could easily be improved.
    if alternate:
        col_name = "AGENCY"
    else:
        if curve_used == "SOFR":
            if calibration:
                col_name = "Agency Spot"
            else:
                col_name = "SOFR"
        elif curve_used == "GOVT":
            col_name = 'Govt Spot'
        elif curve_used == 'AGENCY':
            if calibration:
                col_name = 'Agency Spot'
            else:
                col_name = 'SOFR'
    terms = curve['Term'].values
    rates = curve[col_name].values / 100
    zero_curve_filtered = np.interp(new_ttm[selected_indices], terms, rates)
    return zero_curve_filtered