import numpy as np

def rational_slope(center, height, width):
    def slope(x, y):
        return y + (0.5 * height) * (1 - ((x - center) / width) / -np.sqrt(1 + ((x - center) / width) ** 2))
    return slope