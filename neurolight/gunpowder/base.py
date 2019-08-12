import numpy as np
from gunpowder import *
import logging

logger = logging.getLogger(__name__)


class Clip(BatchFilter):

    def __init__(self, array, min=None, max=None):

        self.array = array
        self.min = min
        self.max = max

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]

        if self.min is None:
            self.min = np.min(array.data)
        if self.max is None:
            self.max = np.max(array.data)

        array.data = np.clip(
            array.data, self.min, self.max).astype(array.spec.dtype)


class Convert(BatchFilter):

    def __init__(self, array, dtype):

        self.array = array
        self.dtype = dtype

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]

        try:
            array.data = array.data.astype(self.dtype)
        except:
            logger.warning("Cannot convert from %s to %s!",
                           array.data.dtype,
                           self.dtype)

        array.spec.dtype = self.dtype


class Threshold(BatchFilter):

    def __init__(self, array, threshold):

        self.array = array
        self.threshold = threshold

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.data[array.data <= self.threshold] = 0
