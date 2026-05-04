import numpy as np
import pytest
from mterrain import functions

@pytest.mark.parametrize(
    "parameters, message", [
        ((0.0, 0.0, 1.0), "non-zero height"),
        ((1.0, 0.0, 0.0), "non-zero width"),
        ((1.0, 0.0, -1.0), "positive width"),
        (([1.0, 1.0], [0.0], [2.0, 2.0]), "equal number"),
        (([1.0, 1.0], [0.0, 0.0], []), "non-zero number"),
])
def test_lorentzian_parameter_validation(parameters, message):
    with pytest.raises(ValueError) as err:
        f = functions.lorentzian_peak(parameters[0], parameters[1], parameters[2])
    assert message in str(err.value)

@pytest.mark.parametrize(
    "parameters", [ 
    (25.0, -50.0, 12.0), 
    (36.0, 25.0, 40.0),
])
def test_lorentzian_key_values(parameters):    
    height, center, width = parameters
    f = functions.lorentzian_peak(height, center, width)
    
    assert f(center) == pytest.approx(height)
    assert f(center - 0.5 * width) == pytest.approx(0.5 * height)
    assert f(center + 0.5 * width) == pytest.approx(0.5 * height)

@pytest.mark.parametrize(
    "parameters, values", [
        ((2.0, -5.0, 4.5), [
            0.096371, 0.130962, 0.187283,
            0.286726, 0.480712, 0.895028,
            1.670103, 1.905882, 1.117241,
            0.584838, 0.336798, 0.214003,
            0.146606, 0.106230, 0.080317,
            0.062766, 0.050357, 0.041274,
            0.034431, 0.029152, 0.024996,
        ]),
        ((-3.0, 4.0, 4.5), [
            -0.041489, -0.048785, -0.058176,
            -0.070537, -0.087253, -0.110605,
            -0.144557, -0.196443, -0.280925,
            -0.430088, -0.721068, -1.342541,
            -2.505155, -2.858824, -1.675862,
            -0.877256, -0.505198, -0.321004,
            -0.219910, -0.159344, -0.120476,
        ]),
        (((2.5, 4.5), (-5.0, 2.0), (7.0, 11.0)), [
            0.699218, 0.865661, 1.101657,
            1.450481, 1.984071, 2.785073,
            3.755618, 4.327586, 4.348416,
            4.452941, 4.796600, 5.025041,
            4.757639, 4.028233, 3.173074,
            2.431751, 1.864724, 1.449336,
            1.146765, 0.924082, 0.757474,
        ]),
])
def test_lorentzian_values(parameters, values):
    f = functions.lorentzian_peak(parameters[0], parameters[1], parameters[2])
    for x, y in zip(np.arange(-15, 16, 1.5), values):
        assert f(x) == pytest.approx(y, abs=1e-5)

@pytest.mark.parametrize(
    "parameters, message", [
        ((0.0, 0.0, 1.0), "non-zero height"),
        ((-1.0, 0.0, 1.0), "positive height"),
        ((1.0, 0.0, 0.0), "non-zero slope"),
        (([1.0, 1.0], [0.0], [2.0, 2.0]), "equal number"),
        (([1.0, 1.0], [0.0, 0.0], []), "non-zero number"),
])
def test_irrational_parameter_validation(parameters, message):
    with pytest.raises(ValueError) as err:
        f = functions.irrational_slope(parameters[0], parameters[1], parameters[2])
    assert message in str(err.value)

@pytest.mark.parametrize(
    "parameters", [ 
        (10.0, 0.0, 2.0), 
        (10.0, 0.0, -1.2), 
        (12.0, -8.0, 2.0), 
        (12.0, -8.0, -1.2),
        (8.0, 5.0, 1.0), 
        (8.0, 5.0, -2.0),
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


@pytest.mark.parametrize(
    "parameters, values", [
        ((12.0, 4.0, -1.2), [
            11.802447, 11.769144, 11.726880,
            11.672238, 11.600074, 11.502420,
            11.366563, 11.171605, 10.882401,
            10.439640, 9.748170, 8.683282,
            7.176697, 5.402978, 3.771656,
            2.559226, 1.757359, 1.244256,
            0.912010, 0.690491, 0.537801,
        ]),
        ((4.0, -3.0, 0.95), [
            0.030086, 0.039028, 0.052570,
            0.074423, 0.112800, 0.188446,
            0.362887, 0.839450, 2.000000,
            3.160550, 3.637113, 3.811554,
            3.887200, 3.925577, 3.947430,
            3.960972, 3.969914, 3.976116,
            3.980588, 3.983916, 3.986459,
        ]),
        (((4.0, 6.0), (-3.0, 4.0), (1.1, 0.7)), [
            0.096094, 0.115435, 0.141760,
            0.179334, 0.236786, 0.334507,
            0.532169, 1.048470, 2.441446,
            3.906365, 4.663441, 5.342746,
            6.232360, 7.291342, 8.229052,
            8.868277, 9.255201, 9.486692,
            9.629907, 9.722576, 9.785207,
        ]),
])
def test_irrational_values(parameters, values):
    f = functions.irrational_slope(parameters[0], parameters[1], parameters[2])
    for x, y in zip(np.arange(-15, 16, 1.5), values):
        assert f(x) == pytest.approx(y, abs=1e-5)