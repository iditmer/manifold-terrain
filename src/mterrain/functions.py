import numpy as np
from collections.abc import Callable
from numpy.typing import NDArray
from typing import Sequence

def rational_slope(center: float | Sequence[float], height: float | Sequence[float], slope_at_center: float | Sequence[float]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
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

        arg = (2 * slope_at_center / height) * (x - center)
        return 0.5 * height * (1 + arg / np.sqrt(1 + arg ** 2))
    
    return slope

def sum_of_rational_slopes(centers: Sequence[float], heights: Sequence[float], slope_at_centers: Sequence[float]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate a function that is the sum of rational slope functions.

    Parameters
    ----------
    centers : float
        Coordinate values for the centers of the component curves
    heights : float
        The approximate distance from the bottom to top of each component curve
    slope_at_centers : float
        Instantaneous slope of curves at their respective centers

    Returns
    -------
    callable
        A function that takes an array as input and returns an array of values on the curve as output
    """

    param_lens = set([len(centers), len(heights), len(slope_at_centers)])
    if len(param_lens) > 1:
        raise ValueError("An equal number of each parameter is required to define component curves.")
    if 0 in param_lens:
        raise ValueError("A non-zero number of each parameter is required to define component curves.")
    
    def slope(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute values along a sum of sigmoidal curves whose parameters were previously specified.

        Parameters
        ----------
        x : ndarray
            Input array of coordinate values along independent axis

        Returns
        -------
        ndarray
            Output array of heights on curve
        """
        output = np.zeros_like(x)
        for (c, h, s) in zip(centers, heights, slope_at_centers):
            arg = (2 * s / h) * (x - c)
            output = output + 0.5 * h * (1 + arg / np.sqrt(1 + arg ** 2))
        return output
    
    return slope

def lorentzian_peak(height: float, center: float, width: float) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate a scaled Lorentzian function whose graph has a single peak.

    Parameters
    ----------
    height : float
        Maximum height of curve peak (occurs at center)
    center : float
        Coordinate value for the center of the peak along the independent axis
    width : float
        Full width of curve at half its max height ("FWHM")

    Returns
    -------
    callable
        A function that takes an array as input and returns an array of values on the curve as output
    """
    def peak(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute dependent values along scaled Lorentzian curve whose parameters were previously specified

        Parameters
        ----------
        x : ndarray
            Input array of coordinate values along independent axis

        Returns
        -------
        ndarray
            Output array of heights on curve
        """
        return height * ((0.5 * width) ** 2) / ((x - center) ** 2 + (0.5 * width) ** 2)
    
    return peak