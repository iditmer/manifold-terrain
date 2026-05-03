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

def lorentzian_peak(height: float | Sequence[float], center: float | Sequence[float], width: float | Sequence[float]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate scaled Lorentzian function whose graph is a single peak; if parameters provided
    as sequences, output function returns a sum of constituent Lorentzian functions.    

    Parameters
    ----------
    height : float
        Maximum height(s) of curve peak(s) (occurs at x = center)
    center : float
        Coordinate value(s) for center(s) of peak(s) along the independent axis
    width : float
        Full width(s) of curve(s) at half max height(s) ("FWHM")

    Returns
    -------
    callable
        Computes heights on the resulting curve given an array of coordinate values
    """
    if isinstance(height, (int, float)):
        height =  [height]
    if isinstance(center, (int, float)):
        center = [center]
    if isinstance(width, (int, float)):
        width = [width]
    
    for h in height:
        if h == 0.0:
            raise ValueError(f"Invalid peak parameter. Expects non-zero height. Input: {h}")
    for w in width:
        if w == 0.0:
            raise ValueError(f"Invalid peak parameter. Expects non-zero width. Input: {w}")
        if w < 0.0:
            raise ValueError(f"Invalid peak parameter. Expects positive width. Input: {w}")
    
    param_lens = set([len(height), len(center), len(width)])
    if 0 in param_lens:
        raise ValueError("A non-zero number of each parameter is required to define component curves.")
    if len(param_lens) > 1:
        raise ValueError("An equal number of each parameter is required to define component curves.")
        
    def peak(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes values along scaled Lorentzian curve (or sum of curves)

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along independent axis

        Returns
        -------
        ndarray
            Output array of heights on curve (or sum of curves)
        """
        output = np.zeros_like(x)
        for (h, c, w) in zip(height, center, width):
            output += h * ((0.5 * w) ** 2) / ((x - c) ** 2 + (0.5 * w) ** 2)
        return output
    
    return peak