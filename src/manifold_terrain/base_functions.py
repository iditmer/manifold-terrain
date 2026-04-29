import numpy as np

def rational_slope(center, height, slope_at_center):
    def slope(x, y):
        arg = (x - center) / (height / (2 * slope_at_center))
        return y + (0.5 * height) * (1 + arg / np.sqrt(1 + arg ** 2))
    return slope