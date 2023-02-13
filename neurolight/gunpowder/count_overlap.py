import logging
import numpy as np
from gunpowder import *
from scipy import ndimage

logger = logging.getLogger(__name__)


class CountOverlap(BatchFilter):

    def __init__(self, gt, gt_overlap, maxnuminst=None):

        self.gt = gt
        self.gt_overlap = gt_overlap
        self.maxnuminst = maxnuminst
        self.dims = None

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        spec = self.spec[self.gt].copy()
        self.provides(self.gt_overlap, spec)

    def prepare(self, request):

        if self.gt not in request:
            request[self.gt] = request[self.gt_overlap].copy()
            request[self.gt].dtype = self.spec[self.gt].dtype

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        array = batch[self.gt].data

        num_channels = len(array.shape) - self.dims
        assert num_channels <= 1, \
            "Sorry, don't know what to do with more than one channel dimension."

        overlap = np.sum((array > 0).astype('uint16'), axis=0)
        if np.sum(overlap > 1) > 0:
            logger.debug('%i Overlapping pixel with labels %s',
                        np.sum(overlap > 1), np.unique(overlap))

        if self.maxnuminst is not None:
            overlap = np.clip(overlap, 0, self.maxnuminst)
        
        request_dtype = request[self.gt_overlap].dtype
        spec.dtype = request_dtype

        masked = Array(data=overlap.astype(request_dtype), spec=spec)
        masked = masked.crop(request[self.gt_overlap].roi)

        batch[self.gt_overlap] = masked


class MaskOverlap(BatchFilter):

    def __init__(self, gt, gt_overlap):

        self.gt = gt
        self.gt_overlap = gt_overlap
        self.dims = None

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        spec = self.spec[self.gt].copy()
        spec.dtype = np.uint8
        self.provides(self.gt_overlap, spec)

    def prepare(self, request):

        if self.gt not in request:
            request[self.gt] = request[self.gt_overlap].copy()
            request[self.gt].dtype = self.spec[self.gt].dtype

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        array = batch[self.gt].data

        num_channels = len(array.shape) - self.dims
        assert num_channels <= 1, \
            "Sorry, don't know what to do with more than one channel dimension."

        overlap = np.sum((array > 0).astype('uint16'), axis=0)
        overlap = (overlap > 1).astype(np.uint8)

        if np.sum(overlap) == 0:
            logger.info('No overlapping pixel!')

        spec.dtype = np.uint8
        masked = Array(data=overlap.astype(np.uint8), spec=spec)
        masked = masked.crop(request[self.gt_overlap].roi)

        batch[self.gt_overlap] = masked


class MaskCloseDistanceToOverlap(BatchFilter):

    def __init__(self, gt, gt_overlap, min_distance=0, max_distance=10):

        self.gt = gt
        self.gt_overlap = gt_overlap
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.dims = None

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        spec = self.spec[self.gt].copy()
        spec.dtype = np.uint8
        self.provides(self.gt_overlap, spec)

    def prepare(self, request):

        if self.gt not in request:
            request[self.gt] = request[self.gt_overlap].copy()
            request[self.gt].dtype = self.spec[self.gt].dtype

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        array = batch[self.gt].data

        num_channels = len(array.shape) - self.dims
        assert num_channels <= 1, \
            "Sorry, don't know what to do with more than one channel dimension."
        
        if array.shape[0] == 1:
            mask = np.zeros(array.shape[1:], dtype=np.uint8)
        else:
            dst = np.ones(np.max(array, axis=0).shape, dtype=np.float32) * 10000
            # assume one mask in one channel slice
            for c in range(array.shape[0]):
                label_mask = array[c] > 0
                other_label_mask = np.max(np.delete(array, c, axis=0), axis=0) > 0
                dist = ndimage.distance_transform_edt(
                    np.logical_not(other_label_mask)
                )
                dst[label_mask] = np.minimum(dst[label_mask], dist[label_mask])

            mask = np.logical_and(dst >= self.min_distance, dst <
                                  self.max_distance).astype(np.uint8)

        if np.sum(mask) == 0:
            logger.info('WARNING: No overlapping pixel!')

        spec.dtype = np.uint8
        masked = Array(data=mask.astype(np.uint8), spec=spec)
        masked = masked.crop(request[self.gt_overlap].roi)

        batch[self.gt_overlap] = masked
