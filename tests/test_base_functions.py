from manifold_terrain import base_functions

def test_rational_slope_height_at_center():
    
    f = base_functions.rational_slope(45.0, 12.0, 1.0)
    assert f(45.0) == 6.0

    f = base_functions.rational_slope(65.0, -10.0, 1.0)
    assert f(65.0) == -5.0