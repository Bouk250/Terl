import numba
import numpy as np

@numba.njit(fastmath=True)
def random_index(min_index:int, max_index:int) -> int:
    return np.random.randint(min_index, max_index,size=1)[0]