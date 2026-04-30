import pytest
from manifold_terrain import base_functions

@pytest.mark.parametrize("input, expected", [((45.0, 12.0, 1.0), 6.0), ((75.0, -8.0, 2.0), -4.0)])
def test_rational_slope_height_at_center(input, expected):    
    center, height, slope = input
    f = base_functions.rational_slope(center, height, slope)
    assert f(center) == expected