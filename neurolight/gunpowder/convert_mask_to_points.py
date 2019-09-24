import numpy as np
from gunpowder import *


class ConvertMaskToPoints(BatchFilter):

    def __init__(self, mask, points, max_num_points=None):
        self.mask = mask
        self.points = points
        self.max_num_points = max_num_points
        self.mask_spec = None

    def setup(self):
        print('setup convert mask: ', self.spec[self.mask])
        self.mask_spec = self.spec[self.mask].copy()
        self.provides(self.points, PointsSpec(self.mask_spec.roi))

    def prepare(self, request):
        if self.mask not in request:
            request[self.mask] = self.mask_spec

    def process(self, batch, request):

        if self.points not in request:
            return

        # get points data
        mask = batch[self.mask].data
        idx = np.transpose(np.nonzero(mask))
        print('len points: ', len(idx))
        if self.max_num_points is not None and self.max_num_points < len(idx):
            idx = np.random.permutation(idx)[:self.max_num_points]
        points_data = {
            i: Point(loc) for i, loc in enumerate(idx)
        }

        # points spec
        points_spec = PointsSpec(roi=batch[self.mask].spec.roi)
        batch.points[self.points] = Points(points_data, points_spec)
