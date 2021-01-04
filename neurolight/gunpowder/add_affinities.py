import logging
import numpy as np

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.array import Array

logger = logging.getLogger(__name__)


def seg_to_affgraph_2d(seg, nhood):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]

    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):
        # first == second pixel?
        tt1 = seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
              max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1])]
        tt2 = seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
              max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1])]
        t1 = tt1 == tt2
        # first pixel fg?
        t2 = seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
             max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1])] > 0
        # second pixel fg?
        t3 = seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
             max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1])] > 0
        aff[e, \
        max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
        max(0, -nhood[e, 1]):min(shape[1],
                                 shape[1] - nhood[e, 1])] = t1 * t2 * t3

    return aff


def seg_to_affgraph_2d_multi(seg, nhood):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape[1:]
    nEdge = nhood.shape[0]

    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):

        slice_center_y = slice(max(0, -nhood[e, 0]),
                               min(shape[0], shape[0] - nhood[e, 0]))
        slice_center_x = slice(max(0, -nhood[e, 1]),
                               min(shape[1], shape[1] - nhood[e, 1]))
        slice_offset_y = slice(max(0, nhood[e, 0]),
                               min(shape[0], shape[0] + nhood[e, 0]))
        slice_offset_x = slice(max(0, nhood[e, 1]),
                               min(shape[1], shape[1] + nhood[e, 1]))

        # first == second pixel?
        center = seg[:, slice_center_y, slice_center_x]
        offset = seg[:, slice_offset_y, slice_offset_x]
        t1 = center == offset
        overlap = np.sum(center, axis=0) > 1
        check_for_overlap = np.sum(overlap) > 1
        if check_for_overlap:
            all_same = np.all(t1, axis=0)
        t1[center == 0] = False
        partial_same = np.any(t1, axis=0)

        # first pixel fg?
        t2 = np.any(center, axis=0)
        # second pixel fg?
        t3 = np.any(offset, axis=0)

        aff[e, slice_center_y, slice_center_x] = partial_same * t2 * t3
        #if check_for_overlap:
        #    aff[e, slice_center_y, slice_center_x][overlap] = \
        #        all_same[overlap] * t2[overlap] * t3[overlap]

    return aff


def seg_to_affgraph_3d_multi(seg, nhood):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape[1:]
    nEdge = nhood.shape[0]

    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):

        slice_center_z = slice(max(0, -nhood[e, 0]),
                               min(shape[0], shape[0] - nhood[e, 0]))
        slice_center_y = slice(max(0, -nhood[e, 1]),
                               min(shape[1], shape[1] - nhood[e, 1]))
        slice_center_x = slice(max(0, -nhood[e, 2]),
                               min(shape[2], shape[2] - nhood[e, 2]))

        slice_offset_z = slice(max(0, nhood[e, 0]),
                               min(shape[0], shape[0] + nhood[e, 0]))
        slice_offset_y = slice(max(0, nhood[e, 1]),
                               min(shape[1], shape[1] + nhood[e, 1]))
        slice_offset_x = slice(max(0, nhood[e, 2]),
                               min(shape[2], shape[2] + nhood[e, 2]))

        # first == second pixel?
        center = seg[:, slice_center_z, slice_center_y, slice_center_x]
        offset = seg[:, slice_offset_z, slice_offset_y, slice_offset_x]
        t1 = center == offset
        all_same = np.all(t1, axis=0)
        t1[center == 0] = False
        partial_same = np.any(t1, axis=0)

        # first pixel fg?
        t2 = np.any(center, axis=0)
        overlap = np.sum(center, axis=0) > 1
        # second pixel fg?
        t3 = np.any(offset, axis=0)

        aff[e, slice_center_z, slice_center_y, slice_center_x] = \
            partial_same * t2 * t3
        # aff[e, slice_center_y, slice_center_x][overlap] = \
        #     all_same[overlap] * t2[overlap] * t3[overlap]

    return aff


def seg_to_affgraph(seg, nhood):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
        max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
        max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
        max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = \
            (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
             max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
             max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] == \
             seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
             max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]), \
             max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])]) \
            * (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
               max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
               max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] > 0) \
            * (seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
               max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]), \
               max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])] > 0)

    return aff


class AddAffinities(BatchFilter):
    '''Add an array with affinities for a given label array and neighborhood to 
    the batch. Affinity values are created one for each voxel and entry in the 
    neighborhood list, i.e., for each voxel and each neighbor of this voxel. 
    Values are 1 iff both labels (of the voxel and the neighbor) are equal and 
    non-zero.

    Args:

        affinity_neighborhood (``list`` of array-like):

            List of offsets for the affinities to consider for each voxel.

        labels (:class:`ArrayKey`):

            The array to read the labels from.

        affinities (:class:`ArrayKey`):

            The array to generate containing the affinities.

        labels_mask (:class:`ArrayKey`, optional):

            The array to use as a mask for ``labels``. Affinities connecting at
            least one masked out label will be masked out in
            ``affinities_mask``. If not given, ``affinities_mask`` will contain
            ones everywhere (if requested).

        unlabelled (:class:`ArrayKey`, optional):

            A binary array to indicate unlabelled areas with 0. Affinities from
            labelled to unlabelled voxels are set to 0, affinities between
            unlabelled voxels are masked out (they will not be used for
            training).

        affinities_mask (:class:`ArrayKey`, optional):

            The array to generate containing the affinitiy mask, as derived
            from parameter ``labels_mask``.
    '''

    def __init__(
        self,
        affinity_neighborhood,
        labels,
        affinities,
        multiple_labels=False,
        labels_mask=None,
        unlabelled=None,
        affinities_mask=None):

        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.labels = labels
        self.unlabelled = unlabelled
        self.multiple_labels = multiple_labels
        self.labels_mask = labels_mask
        self.affinities = affinities
        self.affinities_mask = affinities_mask

    def setup(self):

        assert self.labels in self.spec, (
            "Upstream does not provide %s needed by "
            "AddAffinities" % self.labels)

        voxel_size = self.spec[self.labels].voxel_size

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = Coordinate(
            min([0] + [a[d] for a in self.affinity_neighborhood])
            for d in range(dims)
        ) * voxel_size

        self.padding_pos = Coordinate(
            max([0] + [a[d] for a in self.affinity_neighborhood])
            for d in range(dims)
        ) * voxel_size

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        spec = self.spec[self.labels].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        spec.dtype = np.float32

        self.provides(self.affinities, spec)
        if self.affinities_mask:
            self.provides(self.affinities_mask, spec)
        self.enable_autoskip()

    def prepare(self, request):

        if self.labels_mask:
            assert (
                request[self.labels].roi ==
                request[self.labels_mask].roi), (
                "requested GT label roi %s and GT label mask roi %s are not "
                "the same." % (
                    request[self.labels].roi,
                    request[self.labels_mask].roi))

        if self.unlabelled:
            assert (
                request[self.labels].roi ==
                request[self.unlabelled].roi), (
                "requested GT label roi %s and GT unlabelled mask roi %s are not "
                "the same." % (
                    request[self.labels].roi,
                    request[self.unlabelled].roi))

        if self.labels not in request:
            request[self.labels] = request[self.affinities].copy()

        labels_roi = request[self.labels].roi
        context_roi = request[self.affinities].roi.grow(
            -self.padding_neg,
            self.padding_pos)

        # grow labels ROI to accomodate padding
        labels_roi = labels_roi.union(context_roi)
        request[self.labels].roi = labels_roi

        # same for label mask
        if self.labels_mask and self.labels_mask in request:
            request[self.labels_mask].roi = \
                request[self.labels_mask].roi.union(context_roi)

        # and unlabelled mask
        if self.unlabelled and self.unlabelled in request:
            request[self.unlabelled].roi = \
                request[self.unlabelled].roi.union(context_roi)

        logger.debug("upstream %s request: " % self.labels + str(labels_roi))

    def process(self, batch, request):

        affinities_roi = request[self.affinities].roi

        logger.debug("computing ground-truth affinities from labels")
        arr = batch.arrays[self.labels].data.astype(np.int32)
        if arr.shape[0] == 1:
           arr.shape = arr.shape[1:]
        if self.multiple_labels and len(arr.shape) == 3:
            seg_to_affgraph_fun = seg_to_affgraph_2d_multi
        elif len(arr.shape) == 2:
            seg_to_affgraph_fun = seg_to_affgraph_2d
        elif self.multiple_labels and len(arr.shape) == 4:
            seg_to_affgraph_fun = seg_to_affgraph_3d_multi
        else:
            seg_to_affgraph_fun = seg_to_affgraph
        affinities = seg_to_affgraph_fun(
            arr,
            self.affinity_neighborhood
        ).astype(np.uint8)

        # crop affinities to requested ROI
        offset = affinities_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = affinities_roi.shift(shift)
        crop_roi /= self.spec[self.labels].voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        affinities = affinities[(slice(None),) + crop]

        spec = self.spec[self.affinities].copy()
        spec.roi = affinities_roi
        batch.arrays[self.affinities] = Array(affinities, spec)

        if self.affinities_mask and self.affinities_mask in request:

            if self.labels_mask:

                logger.debug("computing ground-truth affinities mask from "
                             "labels mask")
                affinities_mask = seg_to_affgraph_fun(
                    batch.arrays[self.labels_mask].data.astype(np.int32),
                    self.affinity_neighborhood)
                affinities_mask = affinities_mask[(slice(None),) + crop]

            else:

                affinities_mask = np.ones_like(affinities)

            if self.unlabelled:

                # 1 for all affinities between unlabelled voxels
                unlabelled = (1 - batch.arrays[self.unlabelled].data)
                unlabelled_mask = seg_to_affgraph_fun(
                    unlabelled.astype(np.int32),
                    self.affinity_neighborhood)
                unlabelled_mask = unlabelled_mask[(slice(None),) + crop]

                # 0 for all affinities between unlabelled voxels
                unlabelled_mask = (1 - unlabelled_mask)

                # combine with mask
                affinities_mask = affinities_mask * unlabelled_mask

            affinities_mask = affinities_mask.astype(np.float32)
            batch.arrays[self.affinities_mask] = Array(affinities_mask, spec)

        else:

            if self.labels_mask is not None:
                logger.warning("GT labels does have a mask, but affinities "
                               "mask is not requested.")

        # crop labels to original label ROI
        if self.labels in request:
            roi = request[self.labels].roi
            batch.arrays[self.labels] = batch.arrays[self.labels].crop(roi)

        # same for label mask
        if self.labels_mask and self.labels_mask in request:
            roi = request[self.labels_mask].roi
            batch.arrays[self.labels_mask] = \
                batch.arrays[self.labels_mask].crop(roi)

        # and unlabelled mask
        if self.unlabelled and self.unlabelled in request:
            roi = request[self.unlabelled].roi
            batch.arrays[self.unlabelled] = \
                batch.arrays[self.unlabelled].crop(roi)

        batch.affinity_neighborhood = self.affinity_neighborhood
