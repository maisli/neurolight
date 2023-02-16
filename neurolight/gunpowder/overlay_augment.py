import logging
import numpy as np
from gunpowder import *
import os
import random
import time

logger = logging.getLogger(__name__)


class OverlayAugment(BatchFilter):

    def __init__(self, raw, instances, apply_probability=1., 
            overlay_background=False, numinst=None, max_numinst=2):
        self.raw = raw
        self.instances = instances
        self.apply_probability = apply_probability
        self.overlay_background = overlay_background
        self.numinst = numinst
        self.max_numinst = max_numinst
        self.dims = None

    def setup(self):
        self.dims = self.spec.get_total_roi().dims()

    def prepare(self, request):
        pass

    def process(self, batch, request):
        if not self.overlay_background:
            if np.sum(batch[self.instances].data > 0) == 0:
                logger.info("Skipping overlay augment as batch is background batch")
                return

        if random.random() < self.apply_probability:
            request._random_seed = int(time.time() * 1e6)
            # get current volume data
            raw = batch[self.raw].data.copy()
            instances = batch[self.instances].data.copy()
            if self.numinst:
                numinst = batch[self.numinst].data.copy()

            # get second volume to overlay with
            to_overlay = self.get_upstream_provider().request_batch(request)
            
            to_overlay_raw = to_overlay[self.raw].data
            to_overlay_instances = to_overlay[self.instances].data

            # simply add both volumes together
            overlayed_raw = raw + to_overlay_raw
            overlayed_raw = np.clip(overlayed_raw, 0.0, 1.0)
            overlayed_raw /= float(min(np.max(overlayed_raw), 1.0))

            to_overlay_instances[to_overlay_instances > 0] += np.max(instances)
            overlayed_instances = np.concatenate([instances, to_overlay_instances])

            batch[self.raw].data = overlayed_raw.copy()
            batch[self.instances].data = overlayed_instances.copy()

            if self.numinst:
                to_overlay_numinst = to_overlay[self.numinst].data
                overlayed_numinst = np.clip(
                        numinst + to_overlay_numinst, 0, self.max_numinst)
                batch[self.numinst].data = overlayed_numinst.copy()

