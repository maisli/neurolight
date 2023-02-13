import logging
import numpy as np
from gunpowder import *
import os
import random

logger = logging.getLogger(__name__)


class OverlayAugment(BatchFilter):

    def __init__(self, raw, instances, apply_probability=1.):
        self.raw = raw
        self.instances = instances
        self.apply_probability = apply_probability
        self.dims = None

    def setup(self):
        self.dims = self.spec.get_total_roi().dims()

    def prepare(self, request):
        pass

    def process(self, batch, request):
        if np.sum(batch[self.instances].data > 0) == 0:
            logger.info("Skipping overlay augment as batch is background batch")
            return

        if random.random() < self.apply_probability:

            # get current volume data
            raw = batch[self.raw].data
            instances = batch[self.instances].data

            # get second volume to overlay with
            to_overlay = np.random.choice(self.get_upstream_providers(),
                                          replace=False).request_batch(request)
            to_overlay_raw = to_overlay[self.raw].data
            to_overlay_instances = to_overlay[self.instances].data

            overlayed_raw = raw + to_overlay_raw
            overlayed_raw = np.clip(overlayed_raw, 0.0, 1.0)
            overlayed_raw /= float(min(np.max(overlayed_raw), 1.0))

            to_overlay_instances[to_overlay_instances > 0] += np.max(instances)
            overlayed_instances = np.concatenate([instances, to_overlay_instances])

            batch[self.raw].data = overlayed_raw.copy()
            batch[self.instances].data = overlayed_instances.copy()

