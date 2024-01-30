import inspect
import numpy as np

def check_params(Q, r, J=None):
    _, _, _, _, caller_name, _ = inspect.stack()[1]

    # ---------- Check Q ----------
    if not isinstance(Q, (int, float)):
        raise ValueError(f'Error in {caller_name}: Q must be numeric')

    if Q < 1:
        raise ValueError(f'Error in {caller_name}: Q must be greater than or equal to 1.0')

    # ---------- Check r ----------
    if not isinstance(r, (int, float)):
        raise ValueError(f'Error in {caller_name}: r must be numeric')

    if not np.isfinite(r):
        raise ValueError(f'Error in {caller_name}: r must be greater than 1.0')

    if r <= 1:
        raise ValueError(f'Error in {caller_name}: r must be greater than 1.0')

    # ---------- Check J ----------
    if J is not None:
        if not isinstance(J, int):
            raise ValueError(f'Error in {caller_name}: J must be an integer')

        if J < 1:
            raise ValueError(f'Error in {caller_name}: J must be a positive integer')

# Example usage:
# Q_val = 4
# r_val = 3
# J_val = 3
# check_params(Q_val, r_val, J_val)
