import logging
import numpy as np
from gunpowder import *
import os
import random
import time

logger = logging.getLogger(__name__)


class OverlayAugment(BatchFilter):

    def __init__(
            self, raw, instances, apply_probability=1.,
            overlay_background=False, numinst=None, max_numinst=2,
            loss_mask=None):
        self.raw = raw
        self.instances = instances
        self.apply_probability = apply_probability
        self.overlay_background = overlay_background
        self.numinst = numinst
        self.max_numinst = max_numinst
        self.dims = None
        self.loss_mask = loss_mask

    def setup(self):
        self.dims = self.spec.get_total_roi().dims()

    def prepare(self, request):
        pass

    def process(self, batch, request):
        if not self.overlay_background:
            if not np.any(batch[self.instances].data > 0):
                logger.info("Skipping overlay augment as batch is background batch")
                return

        if random.random() < self.apply_probability:
            request._random_seed = int(time.time() * 1e6) % (2**32)
            # get current volume data
            raw = batch[self.raw].data
            instances = batch[self.instances].data
            if self.numinst:
                numinst = batch[self.numinst].data

            # get second volume to overlay with
            to_overlay = self.get_upstream_provider().request_batch(request)

            to_overlay_raw = to_overlay[self.raw].data
            to_overlay_instances = to_overlay[self.instances].data

            # simply add both volumes together
            overlayed_raw = raw + to_overlay_raw
            overlayed_raw = np.clip(overlayed_raw, 0.0, 1.0)

            to_overlay_instances[to_overlay_instances > 0] += np.max(instances)
            overlayed_instances = np.concatenate([instances, to_overlay_instances])

            batch[self.raw].data = overlayed_raw
            batch[self.instances].data = overlayed_instances

            if self.numinst:
                to_overlay_numinst = to_overlay[self.numinst].data
                overlayed_numinst = np.clip(
                        numinst + to_overlay_numinst, 0, self.max_numinst)
                batch[self.numinst].data = overlayed_numinst

            if self.loss_mask is not None:
                if np.any(batch[self.loss_mask].data == 0):
                    to_overlay_mask = to_overlay[self.loss_mask].data
                    batch[self.loss_mask].data = np.clip(
                         batch[self.loss_mask].data + to_overlay_mask,
                        0, 1)
