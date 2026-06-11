"""Core functions for surface feature definition."""

import numpy as np
from collections.abc import Callable
from numpy.typing import NDArray
from typing import Sequence

class univariate_linear:
    """
    Represents a linear function in one dimension.

    Parameters
    ----------
    slope : float
        Slope of resultant line described by output function
    intercept : float
        Vertical intercept of resultant line described by output function
    """
     
    def __init__(self, 
                 slope: float,
                 intercept: float):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute values on line in one dimension.

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along independent axis

        Returns
        -------
        ndarray
            Array of coordinate values along dependent axis
        """
        return self.intercept + self.slope * x

def bivariate_linear(
    x_slope: float,
    y_slope: float,
    intercept: float,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate a linear function in two dimensions with specified parameters.

    Parameters
    ----------
    x_slope : float
        Slope of resultant plane described by output function in x-direction
    y_slope : float
        Slope of resultant plane described by output function in y-direction
    intercept : float
        Vertical intercept (of z-axis) of resultant plane described by output function
    
    Returns
    -------
    callable
        Computes heights on the resulting plane given an array of coordinate values
    """
    def linear_func(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute values on line in one dimension.

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along x-axis
        y : ndarray
            Array of coordinate values along y-axis

        Returns
        -------
        ndarray
            Output array of heights on plane described by slopes & intercept
        """
        return intercept + x_slope * x + y_slope * y
    return linear_func

def bivariate_peak(
    height: float | Sequence[float],
    center: tuple[float, float] | Sequence[tuple[float, float]],
    width: float | Sequence[float],
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate a function describing a 2D peak with specified parameters.

    Parameters
    ----------
    height : float
        Maximum height of surface peak (occurs at (x,y) = center)
    center : float
        Coordinate values for center of peak in (x,y) plane
    width : float
        Full width of peak at half max height ("FWHM")

    Returns
    -------
    callable
        Computes heights on the resulting surface given an array of coordinate values
    """
    if isinstance(height, (int, float)):
        height =  [height]
    if isinstance(center, tuple):
        center = [center]
    if isinstance(width, (int, float)):
        width = [width]

    for h in height:
        if h == 0.0:
            raise ValueError(f"Invalid surface parameter. Expects non-zero height.")
    for c in center:
        if len(c) != 2:
            raise ValueError(f"Invalid surface parameter. Expects 2D peak center.")
    for w in width:
        if w == 0.0:
            raise ValueError(f"Invalid surface parameter. Expects non-zero width.")
        if w < 0.0:
            raise ValueError(f"Invalid surface parameter. Expects positive width.")
        
    param_lens = set([len(height), len(center), len(width)])
    if 0 in param_lens:
        raise ValueError("A non-zero number of each parameter is required to define component surfaces.")
    if len(param_lens) > 1:
        raise ValueError("An equal number of each parameter is required to define component surfaces.")
    
    def peak_func(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute values on scaled Lorentzian peak surface.

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along x-axis

        y : ndarray
            Array of coordinate values along y-axis

        Returns
        -------
        ndarray
            Output array of heights on surface
        """
        output = np.zeros_like(x)
        for (h, c, w) in zip(height, center, width):
            output += h * ((0.5 * w) ** 2) / ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (0.5 * w) ** 2)
        return output    
    
    return peak_func
    
def univariate_peak(
    height: float | Sequence[float], 
    center: float | Sequence[float], 
    width: float | Sequence[float],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate a function describing a 1D peak with specified parameters.

    Lorentzian function generated has graph that is a single peak; if sequence of
    parameters provided, output function returns a sum of constituent functions.    

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
            raise ValueError(f"Invalid curve parameter. Expects non-zero height. Input: {h}")
    for w in width:
        if w == 0.0:
            raise ValueError(f"Invalid curve parameter. Expects non-zero width. Input: {w}")
        if w < 0.0:
            raise ValueError(f"Invalid curve parameter. Expects positive width. Input: {w}")
    
    param_lens = set([len(height), len(center), len(width)])
    if 0 in param_lens:
        raise ValueError("A non-zero number of each parameter is required to define component curves.")
    if len(param_lens) > 1:
        raise ValueError("An equal number of each parameter is required to define component curves.")
        
    def peak_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute values along scaled Lorentzian curve (or sum of curves).

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
    
    return peak_func

def univariate_slope(
    height: float | Sequence[float], 
    center: float | Sequence[float], 
    slope: float | Sequence[float],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate a function describing a 1D slope with specified parameters.

    Irrational function generated has graph that is a sigmoidal slope; if sequence of
    parameters provided, output function returns a sum of constituent functions.

    Parameters
    ----------
    height : float
        Approximate total height(s) spanned by curve(s) asymptotically
    center : float
        Coordinate value(s) for center(s) inflection point(s) of curve(s) along the independent axis
    slope : float
        Maximum slope(s) of the curve(s); occurs at center(s)

    Returns
    -------
    callable
        Computes heights on the resulting curve given an array of coordinate values
    """
    if isinstance(height, (int, float)):
        height =  [height]
    if isinstance(center, (int, float)):
        center = [center]
    if isinstance(slope, (int, float)):
        slope = [slope]
    
    for h in height:
        if h == 0.0:
            raise ValueError(f"Invalid curve parameter. Expects non-zero height. Input: {h}")
        if h < 0.0:
            raise ValueError(f"Invalid curve parameter. Expects positive height. Input: {h}")
    for s in slope:
        if s == 0.0:
            raise ValueError(f"Invalid curve parameter. Expects non-zero slope. Input: {s}")
    
    param_lens = set([len(height), len(center), len(slope)])
    if 0 in param_lens:
        raise ValueError("A non-zero number of each parameter is required to define component curves.")
    if len(param_lens) > 1:
        raise ValueError("An equal number of each parameter is required to define component curves.")
        
    def slope_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute values along irrational sigmoidal curve (or sum of curves).

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
        for (h, c, s) in zip(height, center, slope):
            arg = (2 * s / h) * (x - c)
            output = output + 0.5 * h * (1 + arg / np.sqrt(1 + arg ** 2))
        return output
    
    return slope_func