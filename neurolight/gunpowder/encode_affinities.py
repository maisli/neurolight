import json
import os

from gunpowder import *
import numpy as np
import tensorflow as tf


class EncodeAffinities(BatchFilter):
    ''' Use a pretrained autoencoder to encode affinities patches into a dense
    representation
    should be placed after! precache
    '''

    def __init__(self, affinities, code, patchshape,
                 autoencoder_chkpt):

        self.affinities = affinities
        self.code = code
        if patchshape[0] == 1:
            patchshape = patchshape[1:]
        self.patchshape = patchshape
        self.autoencoder_chkpt = autoencoder_chkpt

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.session = tf.Session(
            config=config)

        with tf.device("/gpu:1"):
            self.load_graph()

    def setup(self):

        spec = self.spec[self.affinities].copy()
        spec.dtype = np.float32

        self.provides(self.code, spec)
        self.enable_autoskip()

    def load_graph(self):
        with open(
            self.autoencoder_chkpt.split("checkpoint")[0] + "encoder.json",
                'r') as f:
            encoder = json.load(f)
            print(encoder)
        assert len(encoder['inputs']) == 1
        assert len(encoder['outputs']) == 1
        inputs_op_name = encoder['inputs'][0]
        outputs_op_name = encoder['outputs'][0]
        print(inputs_op_name)

        encoder_scope = "encoder"
        with self.session.graph.as_default() as g:
            with tf.variable_scope(encoder_scope):
                gt_affs_patched = tf.placeholder(
                    tf.float32,
                    shape=[None, np.prod(self.patchshape), 1, 1],
                    name="gt_affs")
                tf.train.import_meta_graph(
                    self.autoencoder_chkpt + "" + '.meta',
                    input_map={inputs_op_name:  gt_affs_patched})
                var_list = {}
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    var_list[v.name.split(encoder_scope+"/")[-1][:-2]] = v
                saver = tf.train.Saver(var_list=var_list)
                saver.restore(self.session, self.autoencoder_chkpt)

                self.code_op = tf.stop_gradient(g.get_tensor_by_name(
                    encoder_scope + "/" + outputs_op_name))
                self.input_pl = gt_affs_patched.name

    def prepare(self, request):
        pass

    def process(self, batch, request):

        data = batch.arrays[self.affinities].data
        input_shape = data.shape[1:]
        data = np.transpose(data, [1, 2, 0])
        data = np.reshape(data, [data.shape[0]*data.shape[1],
                                 data.shape[2], 1, 1])
        sz = data.shape[0]
        cnt = int(np.ceil(sz/1024))
        slices = np.array_split(data, [x*1024 for x in range(1, cnt)])
        codes = []
        for slce in slices:
            codes.append(self.session.run(self.code_op,
                                          feed_dict={self.input_pl: slce}))

        spec = self.spec[self.code].copy()
        spec.roi = request[self.code].roi
        code = np.concatenate(codes)
        code = np.transpose(code, [1, 0])
        code = np.reshape(code, [-1] + list(input_shape))

        batch.arrays[self.code] = Array(code, spec)
