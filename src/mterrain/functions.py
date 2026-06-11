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
        Slope of line along independent axis.
    intercept : float
        Height of line along depenedent axis when independent variable = 0.0
    """
     
    def __init__(self, 
                 slope: float,
                 intercept: float):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute heights on a line.

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along independent axis

        Returns
        -------
        ndarray
            Array of height values
        """
        return self.intercept + self.slope * x

class bivariate_linear:
    """
    Represents a linear function in two dimensions.

    Parameters
    ----------
    x_slope : float
        Slope of plane along x-direction
    x_slope : float
        Slope of plane along y-direction
    intercept : float
        Height of plane at point (0,0)
    """

    def __init__(self,
        x_slope: float,
        y_slope: float,
        intercept: float,):
        self.x_slope = x_slope
        self.y_slope = y_slope
        self.intercept = intercept

    def __call__(self,
                 x: NDArray[np.float64], 
                 y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute heights on a plane.

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along x-axis
        y : ndarray
            Array of coordinate values along y-axis

        Returns
        -------
        ndarray
            Array of height values
        """
        return self.intercept + self.x_slope * x + self.y_slope * y

class bivariate_peak:    
    """
    Represents symmetric peaks in two dimensions.

    If single values are provided for parameters, surface is a single
    bell-like peak with a maximum height equal to the height parameter.

    If sequences of parameter values are provided, surface is a sum
    of constituent peaks.

    Parameters
    ----------
    height : float
        Height of constituent peak [at (x,y) = center]
    center : float
        Coordinate values of center of constituent peak in (x,y) plane
    width : float
        Diameter of constituent peak at half its maximum height
    """

    def __init__(self,
                height: float | Sequence[float],
                center: tuple[float, float] | Sequence[tuple[float, float]],
                width: float | Sequence[float]):
        if isinstance(height, (int, float)):
            height =  [height]
        if isinstance(center, tuple):
            center = [center]
        if isinstance(width, (int, float)):
            width = [width]
            
        param_lens = set([len(height), len(center), len(width)])
        if 0 in param_lens:
            raise ValueError("A non-zero number of each parameter is required to define component surfaces.")
        if len(param_lens) > 1:
            raise ValueError("An equal number of each parameter is required to define component surfaces.")

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
            
        self.h = height
        self.c = center
        self.w = width
        
    def __call__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute heights on symmetric 2D peak surface.

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along x-axis

        y : ndarray
            Array of coordinate values along y-axis

        Returns
        -------
        ndarray
            Array of heights on surface
        """
        output = np.zeros_like(x)
        for (h, c, w) in zip(self.h, self.c, self.w):
            output += h * ((0.5 * w) ** 2) / ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (0.5 * w) ** 2)
        return output 

class univariate_peak:
    """
    Represents symmetric peaks in one dimension.

    If single values are provided for parameters, curve is a single
    bell-like peak with a maximum height equal to the height parameter.

    If sequences of parameter values are provided, curve is a sum
    of constituent peaks.

    Parameters
    ----------
    height : float
        Height of constituent peak [at x = center]
    center : float
        Coordinate value of center of constituent peak along independent axis
    width : float
        Width of constituent peak at half its maximum height
    """

    def __init__(self,
                height: float | Sequence[float], 
                center: float | Sequence[float], 
                width: float | Sequence[float]):
        
        if isinstance(height, (int, float)):
            height =  [height]
        if isinstance(center, (int, float)):
            center = [center]
        if isinstance(width, (int, float)):
            width = [width]
        
        param_lens = set([len(height), len(center), len(width)])
        if 0 in param_lens:
            raise ValueError("A non-zero number of each parameter is required to define component curves.")
        if len(param_lens) > 1:
            raise ValueError("An equal number of each parameter is required to define component curves.")
        
        for h in height:
            if h == 0.0:
                raise ValueError(f"Invalid curve parameter. Expects non-zero height. Input: {h}")
        for w in width:
            if w == 0.0:
                raise ValueError(f"Invalid curve parameter. Expects non-zero width. Input: {w}")
            if w < 0.0:
                raise ValueError(f"Invalid curve parameter. Expects positive width. Input: {w}")
            
        self.h = height
        self.c = center
        self.w = width
    
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: 
        """
        Compute heights on peak (or sum of peaks) in one dimension.

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along independent axis

        Returns
        -------
        ndarray
            Array of heights on curve
        """       
        output = np.zeros_like(x)
        for (h, c, w) in zip(self.h, self.c, self.w):
            output += h * ((0.5 * w) ** 2) / ((x - c) ** 2 + (0.5 * w) ** 2)
        return output

class univariate_slope:
    """
    Represents symmetric slopes in one dimension.

    If single values are provided for parameters, curve is a single
    sigmoidal shape spanning the specified height. Behavior is 
    asymptotic and horizontal span of slope is dictated by its
    steepness at the center.

    If sequences of parameter values are provided, curve is a sum
    of constituent sigmoidal slopes.

    Parameters
    ----------
    height : float
        Height spanned by constituent curve
    center : float
        Location of inflection point of constituent curve
    slope : float
        Maximal slope of curve (attained at inflection point)
    """

    def __init__(self, 
                height: float | Sequence[float], 
                center: float | Sequence[float], 
                slope: float | Sequence[float]):
        if isinstance(height, (int, float)):
            height =  [height]
        if isinstance(center, (int, float)):
            center = [center]
        if isinstance(slope, (int, float)):
            slope = [slope]
        
        param_lens = set([len(height), len(center), len(slope)])
        if 0 in param_lens:
            raise ValueError("A non-zero number of each parameter is required to define component curves.")
        if len(param_lens) > 1:
            raise ValueError("An equal number of each parameter is required to define component curves.")
        
        for h in height:
            if h == 0.0:
                raise ValueError(f"Invalid curve parameter. Expects non-zero height. Input: {h}")
            if h < 0.0:
                raise ValueError(f"Invalid curve parameter. Expects positive height. Input: {h}")
        for s in slope:
            if s == 0.0:
                raise ValueError(f"Invalid curve parameter. Expects non-zero slope. Input: {s}")
            
        self.h = height
        self.c = center
        self.s = slope

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute heights on slope (or sum of slopes) in one dimension.

        Parameters
        ----------
        x : ndarray
            Array of coordinate values along independent axis

        Returns
        -------
        ndarray
            Array of heights on curve
        """
        output = np.zeros_like(x)
        for (h, c, s) in zip(self.h, self.c, self.s):
            arg = (2 * s / h) * (x - c)
            output = output + 0.5 * h * (1 + arg / np.sqrt(1 + arg ** 2))
        return output