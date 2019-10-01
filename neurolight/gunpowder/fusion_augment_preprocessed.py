import logging
import numpy as np
from gunpowder import *
import h5py
from scipy import ndimage
from skimage import io
import os

logger = logging.getLogger(__name__)


class FusionAugmentPreprocessed(BatchFilter):

    def __init__(self, raw, labels, hdf_source, num_insert=1,
                 min_insert_distance=0, max_insert_distance=10,
                 apply_probability=1.
                 ):
        self.raw = raw
        self.labels = labels
        self.hdf_source = hdf_source
        self.num_insert = num_insert
        self.min_insert_distance = min_insert_distance
        self.max_insert_distance = max_insert_distance
        self.apply_probability = apply_probability
        # self.cnt = 0
        self.dims = None

    def setup(self):
        self.dims = self.spec.get_total_roi().dims()

    def prepare(self, request):
        pass

    def process(self, batch, request):

        if np.random.uniform(0, 1) > self.apply_probability:
            logger.info("Skipping fusion augment this time")
            return

        if np.sum(batch[self.labels].data > 0) == 0:
            logger.info("Skipping fusion augment as batch is background batch")
            return

        raw = batch[self.raw].data
        raw_roi = np.array(batch[self.raw].spec.roi.get_shape())
        labels = batch[self.labels].data
        labels_roi = np.array(batch[self.labels].spec.roi.get_shape())

        # save original for checking
        # self.cnt += 1
        # output_folder = '/home/maisl/data/flylight/single_neurons' \
        #                 '/fusion_augment_examples'
        # io.imsave(
        #     os.path.join(output_folder, 'original_%i.tif' % self.cnt),
        #     np.moveaxis((raw * 255).astype(np.uint8), 0, -1)
        # )
        if labels.ndim > self.dims:
            bg = np.max(labels, axis=0) == 0
        else:
            bg = labels == 0
        margin = (raw_roi - labels_roi) // 2
        dist_to_fg = ndimage.distance_transform_edt(bg)
        mask = (dist_to_fg >= self.min_insert_distance) * (
            dist_to_fg <= self.max_insert_distance)
        base_ids = np.transpose(np.nonzero(mask))

        # read random neurons from hdf source
        with h5py.File(self.hdf_source, 'r') as inf:
            instance_cnt = 0
            instances = list(inf.keys())
            while instance_cnt < self.num_insert:
                instance = np.random.choice(instances, replace=False)
                add_mask_shape = inf[instance + '/mask'].shape
                # check if neuron bb > required roi
                if np.any(raw_roi > add_mask_shape):
                    continue
                # chose point in batch mask where to insert new instance
                base_idx = base_ids[np.random.randint(0, len(base_ids))]
                base_end = labels_roi - base_idx
                if np.any(add_mask_shape - base_end < base_idx):
                    continue

                # get valid area of single neuron mask
                slices = tuple([slice(b, e) for b, e in
                                zip(base_idx, add_mask_shape - base_end)])
                add_ids = np.transpose(np.nonzero(
                    np.array(inf[instance + '/mask'][slices])))
                if len(add_ids) == 0:
                    continue
                add_idx = add_ids[np.random.randint(0, len(add_ids))]

                # load mask cutout from source
                add_mask_end = add_idx + base_idx + base_end
                add_mask_slices = tuple([slice(b, e) for b, e in zip(
                    add_idx, add_mask_end)])
                add_mask = np.array(inf[instance + '/mask'][add_mask_slices])

                # load raw and softmask cutout from source
                add_raw_begin = add_idx - margin
                add_raw_end = add_mask_end + margin
                add_raw_end[(raw_roi - labels_roi) % 2 == 1] += 1
                add_raw_slices = tuple([slice(b, e) for b, e in zip(
                    np.maximum(add_raw_begin, np.array([0, 0, 0])),
                    np.minimum(add_raw_end, add_mask_shape)
                )])
                add_raw = np.array(inf[instance + '/raw'][add_raw_slices])
                add_raw = np.clip(add_raw / 1500.0, 0, 1)
                soft_mask = np.array(
                    inf[instance + '/soft_mask'][add_raw_slices])

                # pad raw and softmask if necessary
                if np.any(add_raw_begin < 0) or np.any(
                    add_raw_end > add_mask_shape):
                    begin_pad = np.abs(
                        np.minimum(add_raw_begin, np.array([0, 0, 0])))
                    end_pad = np.abs(np.minimum(
                        add_mask_shape - add_raw_end, np.array([0, 0, 0])))
                    pad = [(b, e) for b, e in zip(begin_pad, end_pad)]
                    soft_mask = np.pad(
                        soft_mask, pad_width=pad, mode='constant')
                    add_raw = np.pad(add_raw, pad_width=pad, mode='constant')

                # add mask and raw to batch volume
                channel = np.random.permutation([0, 1, 2])
                probability = [True, np.random.uniform(0, 1) > 0.5,
                               np.random.uniform(0, 1) > 0.5]
                intensity = [1.0, np.random.uniform(0, 1),
                             np.random.uniform(0, 1)]
                for c, cp, ci in zip(channel, probability, intensity):
                    if cp:
                        c_soft_mask = soft_mask * ci
                        raw[c] = c_soft_mask * add_raw + (1 - c_soft_mask) * \
                                 raw[c]
                add_mask *= np.max(labels) + 1
                labels = np.concatenate(
                    [labels, np.reshape(add_mask, (1,) + add_mask.shape)],
                    axis=0
                )
                instance_cnt += 1
                # save fused raw and labels for checking
                # io.imsave(
                #     os.path.join(output_folder, 'fused_%i.tif' % self.cnt),
                #     np.moveaxis((raw * 255).astype(np.uint8), 0, -1)
                # )
                # show_labels = np.zeros((3,) + add_mask.shape, dtype=np.uint8)
                # show_labels[0] = (
                #         (np.max(labels[:-1], axis=0) > 0) * 255).astype(
                #     np.uint8)
                # show_labels[1] = ((add_mask > 0) * 255).astype(np.uint8)
                # io.imsave(
                #     os.path.join(output_folder, 'labels_%i.tif' % self.cnt),
                #     np.moveaxis(show_labels, 0, -1)
                # )
        batch[self.raw].data = raw.copy()
        batch[self.labels].data = labels.copy()
