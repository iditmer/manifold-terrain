import pytest
from mterrain import functions

@pytest.mark.parametrize("parameters, message", [
                         ((0.0, 0.0, 1.0), "non-zero height"),
                         ((1.0, 0.0, 0.0), "non-zero width"),
                         ((1.0, 0.0, -1.0), "positive width"),
                         (([1.0, 1.0], [0.0], [2.0, 2.0]), "equal number"),
                         (([1.0, 1.0], [0.0, 0.0], []), "non-zero number")
])
def test_lorentzian_parameter_validation(parameters, message):
    with pytest.raises(ValueError) as err:
        f = functions.lorentzian_peak(parameters[0], parameters[1], parameters[2])
    assert message in str(err.value)

@pytest.mark.parametrize("parameters", [ 
    (25.0, -50.0, 12.0), (36.0, 25.0, 40.0) 
])
def test_lorentzian_key_values(parameters):
    
    height, center, width = parameters
    f = functions.lorentzian_peak(height, center, width)
    
    assert f(center) == pytest.approx(height)
    assert f(center - 0.5 * width) == pytest.approx(0.5 * height)
    assert f(center + 0.5 * width) == pytest.approx(0.5 * height)

@pytest.mark.parametrize("parameters, message", [
                         ((0.0, 0.0, 1.0), "non-zero height"),
                         ((-1.0, 0.0, 1.0), "positive height"),
                         ((1.0, 0.0, 0.0), "non-zero slope"),
                         (([1.0, 1.0], [0.0], [2.0, 2.0]), "equal number"),
                         (([1.0, 1.0], [0.0, 0.0], []), "non-zero number")
])
def test_irrational_parameter_validation(parameters, message):
    with pytest.raises(ValueError) as err:
        f = functions.irrational_slope(parameters[0], parameters[1], parameters[2])
    assert message in str(err.value)

@pytest.mark.parametrize("parameters", [ 
    (10.0, 0.0, 2.0), (10.0, 0.0, -1.2), 
    (12.0, -8.0, 2.0), (12.0, -8.0, -1.2),
    (8.0, 5.0, 1.0), (8.0, 5.0, -2.0)
])
def test_irrational_key_values(parameters):    
    
    height, center, slope = parameters
    f = functions.irrational_slope(height, center, slope)    
    
    assert f(center) == pytest.approx(0.5 * height)        
    
    if slope > 0:
      assert f(-1e25) == pytest.approx(0.0)
      assert f(1e25) == pytest.approx(height)
    else: 
      assert f(-1e25) == pytest.approx(height)
      assert f(1e25) == pytest.approx(0.0)
    
    dy = f(center + 1e-6) - f(center - 1e-6)
    assert (dy / 2e-6) == pytest.approx(slope)