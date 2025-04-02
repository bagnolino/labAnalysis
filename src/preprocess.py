import numpy as np
import numpy.typing as npt
from typing import Tuple


def clean_data(data:npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    '''From all the data selecet only the column we are interested in.

    Args:
        data (npt.NDArray[np.float64]): The raw data

    Returns:
        Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: Two arrays, corresponding to x_data and y_data
        
    '''
    data= data.T
    x_data = data[3]
    y_data = data[4]
    return x_data, y_data
