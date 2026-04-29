import numpy as np
from collections.abc import Callable
from numpy.typing import NDArray

def rational_slope(center: float, height: float, slope_at_center: float) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate a rational function whose graph has a sigmoidal shape.

    Parameters
    ----------
    center : float
        Coordinate value for the center of the curve along the independent axis
    height : float
        The approximate distance from the bottom to top of the curve in the dependent direction; 
        asymptotic behavior means the output will not have this exact height
    slope_at_center : float
        Real number describing instantaneous slope of curve at its center; 
        impacts 'length' of the sigmoid - lower slope => more stretched along independent axis

    Returns
    -------
    callable
        A function that takes an array as input and returns an array of values on the curve as output
    """

    def slope(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute values along a sigmoidal curve whose parameters were previously specified.

        Parameters
        ----------
        x : ndarray
            Input array of coordinate values along independent axis

        Returns
        -------
        ndarray
            Output array of heights on curve
        """

        arg = (x - center) / (height / (2 * slope_at_center))
        return 0.5 * height * (1 + arg / np.sqrt(1 + arg ** 2))
    
    return slope