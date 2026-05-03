import numpy as np
from collections.abc import Callable
from numpy.typing import NDArray
from typing import Sequence

def lorentzian_peak(height: float | Sequence[float], center: float | Sequence[float], width: float | Sequence[float]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate scaled Lorentzian function whose graph is a single peak; if parameters provided
    as sequences, output function returns a sum of constituent functions.    

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

def irrational_slope(height: float | Sequence[float], center: float | Sequence[float], slope: float | Sequence[float]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate irrational function whose graph is a sigmoidal slope; if parameters provided
    as sequences, output function returns a sum of constituent functions.    

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
        
    def sloped_curve(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes values along irrational sigmoidal curve (or sum of curves)

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
    
    return sloped_curve