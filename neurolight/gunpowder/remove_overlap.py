import numpy as np
from gunpowder import *


class RemoveOverlap(BatchFilter):

    def __init__(self, gt, gt_cleaned, overlap='remove'):

        self.gt = gt
        self.gt_cleaned = gt_cleaned
        self.overlap = overlap
        self.dims = None
        self.gt_spec = None
        self.grow = None

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        spec = self.spec[self.gt].copy()
        self.provides(self.gt_cleaned, spec)

    def prepare(self, request):

        self.gt_spec = request[self.gt].copy()
        gt_cleaned_spec = request[self.gt_cleaned].copy()

        if gt_cleaned_spec.roi.get_shape() > self.gt_spec.roi.get_shape():
            request[self.gt] = gt_cleaned_spec

    def process(self, batch, request):

        gt_spec = request[self.gt]

        spec = batch[self.gt].spec.copy()
        array = batch[self.gt].data

        num_channels = len(array.shape) - self.dims
        assert num_channels <= 1, \
            "Sorry, don't know what to do with more than one channel dimension."

        cleaned = np.max(array, axis=0)
        if self.overlap == 'remove':
            overlap = np.sum((array > 0).astype(np.uint16), axis=0)
            cleaned[overlap > 1] = 0

        batch[self.gt_cleaned] = Array(data=cleaned.astype(np.uint16),
                                       spec=spec)
        gt = Array(data=batch[self.gt].data.copy(), spec=spec.copy())

        gt = gt.crop(gt_spec.roi)
        batch.arrays[self.gt] = gt
