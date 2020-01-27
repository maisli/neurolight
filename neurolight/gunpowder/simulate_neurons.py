import logging
import numpy as np
from gunpowder import *
from gunpowder.profiling import Timing
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from skimage import io
import h5py
from datetime import datetime

logger = logging.getLogger(__name__)


class SimulateNeurons(BatchProvider):

    def __init__(self, raw, gt, array_specs,
                 size,
                 numinst_min,
                 numinst_max,
                 points_per_skeleton_min=3,
                 points_per_skeleton_max=5,
                 smoothness=1,
                 interpolation="random"
                 ):
        self.raw = raw
        self.gt = gt
        self.array_specs = array_specs
        self.size = size
        self.numinst_min = numinst_min
        self.numinst_max = numinst_max
        self.points_per_skeleton_min = points_per_skeleton_min
        self.points_per_skeleton_max = points_per_skeleton_max
        self.smoothness = smoothness
        self.interpolation = interpolation
        self.dims = 3

    def setup(self):
        offset = Coordinate((0,) * self.dims)
        shape = Coordinate(self.size)

        raw_spec = self.array_specs[self.raw].copy()
        if raw_spec.roi is None:
            raw_spec.roi = Roi(offset, shape)
        self.provides(self.raw, raw_spec)

        gt_spec = self.array_specs[self.gt].copy()
        if gt_spec.roi is None:
            gt_spec.roi = Roi(offset, shape)
        self.provides(self.gt, gt_spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()
        
        # try to import skelerator
        try:
            from skelerator import Tree, Skeleton
        except ImportError:
            logger.info("skelerator cannot be imported, please check!")
        
        batch = Batch()
        n_objects = np.random.randint(self.numinst_min, self.numinst_max + 1)
        raw_spec = request.array_specs[self.raw].copy()
        shape = list(raw_spec.roi.get_shape())

        # create noise
        random_noise = np.random.uniform(0, 0.005, [3] + shape)
        pepper_noise = (np.random.uniform(0, 1, [3] + shape) > 0.9999).astype(
            np.uint8)
        pepper_noise = pepper_noise * np.random.uniform(
            0.4, 1, pepper_noise.shape) * 2
        mixed_noise = random_noise + pepper_noise
        smoothed_noise = gaussian_filter(mixed_noise, sigma=1)

        # sample one tree for each object and generate its skeleton
        seeds = np.zeros(shape, dtype=int)
        instances = []
        for i in range(n_objects):
            points_per_skeleton = np.random.randint(
                self.points_per_skeleton_min, self.points_per_skeleton_max + 1)
            instance = np.zeros(shape, dtype=int)
            points = np.stack(
                [np.random.randint(0, shape[i], points_per_skeleton)
                 for i in range(3)], axis=1)
            tree = Tree(points)
            skeleton = Skeleton(tree, [1, 1, 1], self.interpolation,
                                generate_graph=False)
            instance = skeleton.draw(instance, np.array([0, 0, 0]), 1)
            instances.append(instance)

        # process instances
        for i in range(len(instances)):
            instances[i] = ndimage.binary_dilation(instances[i] > 0).astype(
                np.uint8)
            instance_smoothed = ndimage.gaussian_filter(
                instances[i].astype(np.float32), 1)
            channel = np.random.permutation([0, 1, 2])
            prob = [True, 
                    np.random.uniform(0, 1) > 0.5,
                    np.random.uniform(0, 1) > 0.9]
            intensity = [np.random.uniform(0.5, 1.0),
                         np.random.uniform(0, 1),
                         np.random.uniform(0, 1)]
            mask = instances[i] > 0
            for c, cp, ci in zip(channel, prob, intensity):
                if cp:
                    smoothed_noise[c][mask] += instance_smoothed[mask] * ci

        smoothed_noise = np.clip(smoothed_noise, 0, 1)

        # relabel instance masks
        for i in range(len(instances)):
            instances[i] *= (i + 1)

        instances = np.stack(instances, axis=0).astype(np.uint16)
        smoothed_noise = smoothed_noise.astype(np.float32)
        spec = self.array_specs[self.raw].copy()
        spec.roi = raw_spec.roi.copy()
        batch[self.raw] = Array(data=smoothed_noise, spec=spec)

        gt_spec = self.array_specs[self.gt].copy()
        gt_spec.roi = spec.roi.copy()
        gt_request = request[self.gt]
        gt_array = Array(data=instances, spec=gt_spec)
        batch[self.gt] = gt_array.crop(gt_request.roi)

        timing.stop()
        batch.profiling_stats.add(timing)

        # write data
        #sample_name = datetime.now().strftime("%d_%b_%Y_%H_%M_%S_%f")
        #outfn = "/nrs/saalfeld/maisl/ppp/simulated/" + sample_name
        #f = h5py.File(outfn + '.hdf', "w")
        #f.create_dataset("instances", data=instances.astype(np.uint16))
        #f.create_dataset("raw", data=smoothed_noise.astype(np.float32))
        #f.close()

        #mip = (np.max(smoothed_noise.astype(np.float32), axis=1) * 255).astype(
        #    np.uint8)
        #io.imsave(outfn + '.png', np.moveaxis(mip, 0, -1))

        return batch

