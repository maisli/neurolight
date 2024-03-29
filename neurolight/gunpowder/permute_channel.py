import logging
import numpy as np
from gunpowder import *

logger = logging.getLogger(__name__)


class PermuteChannel(BatchFilter):

    def __init__(self, raw, prob):
        self.raw = raw
        self.prob = prob
        self.dims = None

    def setup(self):
        self.dims = self.spec.get_total_roi().dims()

    def prepare(self, request):
        pass

    def process(self, batch, request):
        if np.random.rand() < self.prob:
            raw = batch[self.raw].data
            assert raw.ndim == self.dims + 1, \
                "Sorry, only one channel dim can be permuted!"

            # heads up: assuming channels first
            raw = np.random.permutation(raw)
            batch[self.raw].data = raw.copy()

