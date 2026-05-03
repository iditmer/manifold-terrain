import pytest
from mterrain import functions

@pytest.mark.parametrize("input, expected", [ ((45.0, 12.0, 1.0), 6.0), ((45.0, -12.0, 1.0), -6.0) ])
def test_rational_slope_height_at_center(input, expected):    
    center, height, slope = input
    f = functions.rational_slope(center, height, slope)
    assert f(center) == expected

@pytest.mark.parametrize("input, expected",[
    (((50.0, 7.0, 1.0), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
    [ 0.0133219887636046, 0.0235790131946497, 0.0523933649550353, 0.196495752718939, 3.5, 
      6.80350424728106, 6.94760663504496, 6.97642098680535, 6.9866780112364 ]),
    (((50.0, 7.0, -0.2), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
    [ 6.70655067215766, 6.52322615314452, 6.13401843147407, 5.23648628424892, 3.5, 
      1.76351371575108, 0.865981568525928, 0.476773846855483, 0.293449327842338 ]),
    (((-12.0, 4.0, 1.2), [ -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0 ]),
     [ 0.0162366706282988, 0.0420391605247898, 0.251685447756925, 3.53644255919475, 
       3.94561242937073, 3.9809845463555, 3.99045705023996, 3.9942853760559, 3.99620046011194 ]),
    (((-12.0, -4.0, 1.2), [ -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0 ]),
     [ -3.9837633293717, -3.95796083947521, -3.74831455224308, -0.463557440805248, -0.0543875706292662, 
       -0.0190154536444969, -0.00954294976004033, -0.00571462394410016, -0.00379953988806303 ]),
])
def test_rational_slope_values(input, expected):
    center, height, slope = input[0]
    f = functions.rational_slope(center, height, slope)
    for i in range(len(input[1])):
        assert f(input[1][i]) == pytest.approx(expected[i])

def test_sum_of_rational_slope_values():
    
    x = [ -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, ]    
    y = [ 0.035332329819115604, 0.045990192574770106, 0.06302452667065472, 0.09356946177848846, 
          0.1598642975653543, 0.3666903199551057, 1.8415165378250853, 10.372723109835936, 11.91812837082345, 
          12.361451302850513, 13.394955961211272, 18.463302408041713, 23.52010023956195, 24.5043596643045 ]
    f = functions.sum_of_rational_slopes( [ 45.0, 90.0 ], [ 12.0, 13.0 ], [ 1.2, 0.8 ] )

    for i in range(len(x)):
        assert f(x[i]) == pytest.approx(y[i])

@pytest.mark.parametrize("parameters, message", [
                         ((0.0, 0.0, 1.0), "non-zero height"),
                         ((1.0, 0.0, 0.0), "non-zero width"),
                         ((1.0, 0.0, -1.0), "positive width"),
                         (([1.0, 1.0], [0.0], [2.0, 2.0]), "equal number"),
                         (([1.0, 1.0], [0.0, 0.0], []), "non-zero number")])
def test_lorentzian_parameter_validation(parameters, message):
    with pytest.raises(ValueError) as err:
        f = functions.lorentzian_peak(parameters[0], parameters[1], parameters[2])
    assert message in str(err.value)

@pytest.mark.parametrize("parameters", [ (25.0, -50.0, 12.0), (36.0, 25.0, 40.0) ])
def test_lorentzian_peak(parameters):
    height, center, width = parameters
    f = functions.lorentzian_peak(height, center, width)
    assert f(center) == pytest.approx(height)
    assert f(center - 0.5 * width) == pytest.approx(0.5 * height)
    assert f(center + 0.5 * width) == pytest.approx(0.5 * height)
    