# copied from Lorenz Rumberger
import gunpowder as gp
import colorsys
import numpy as np


class RandomHue(gp.BatchFilter):
    # int max_change: maximum degree of rotation on the hue color wheel
    # float prob: probability of applying RandomHue transformation
    def __init__(self, array, max_change, prob):
        self.array = array
        self.prob = prob
        self.max_change = max_change
        self.rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
        self.hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    def process(self, batch, request):
        if np.random.rand() < self.prob:
            array = batch.arrays[self.array]
            array.data = np.clip(array.data, 0.0, 1.0)
            
            # transform rgb to hsv color space and add random color shift to hue value
            h, s, v = self.rgb_to_hsv(array.data[0], array.data[1], array.data[2])
            color_offset = np.random.uniform(-self.max_change, self.max_change)
            h += color_offset
            h = h % 1

            # transform back to rgb space
            r, g, b = self.hsv_to_rgb(h, s, v)
            array.data = np.array([r, g, b])
            array.data = np.clip(array.data, 0.0, 1.0)

