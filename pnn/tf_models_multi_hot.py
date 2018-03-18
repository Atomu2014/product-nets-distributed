from __future__ import division
from __future__ import print_function

import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

WEIGHTS = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS]
EMBEDS = ['EMBEDS']
KERNELS = WEIGHTS + ['KERNELS']
NN_WEIGHTS = WEIGHTS + ['NN_WEIGHTS']
SUB_NN_WEIGHTS = WEIGHTS + ['SUB_NN_WEIGHTS']
BIASES = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES]


def as_model(model_name, **model_param):
    model_name = model_name.lower()
    if model_name == 'lr':
        return LR(**model_param)
    elif model_name == 'fm':
        return FM(**model_param)
    elif model_name == 'ffm':
        return FFM(**model_param)
    elif model_name == 'kfm':
        return KFM(**model_param)
    elif model_name == 'nfm':
        return NFM(**model_param)
    elif model_name == 'fnn':
        return FNN(**model_param)
    elif model_name == 'ccpm':
        return CCPM(**model_param)
    elif model_name == 'deepfm':
        return DeepFM(**model_param)
    elif model_name == 'ipnn':
        return IPNN(**model_param)
    elif model_name == 'kpnn':
        return KPNN(**model_param)
    elif model_name == 'pin':
        return PIN(**model_param)


def get_initializer(init_type='xavier', minval=-0.001, maxval=0.001, mean=0, stddev=0.001, gain=1.):
    if type(init_type) is str:
        init_type = init_type.lower()
    assert init_type in {'xavier', 'orth', 'uniform', 'normal'} if type(init_type) is str \
        else type(init_type) in {int, float}, 'init type: {"xavier", "orth", "uniform", "normal", int, float}'
    if init_type == 'xavier':
        return tf.contrib.layers.xavier_initializer(uniform=True)
    elif init_type == 'orth':
        return tf.orthogonal_initializer(gain=gain)
    elif init_type == 'identity':
        return tf.initializer.identity(gain=gain)
    elif init_type == 'uniform':
        return tf.random_uniform_initializer(minval=minval, maxval=maxval)
    elif init_type == 'normal':
        return tf.truncated_normal_initializer(mean=mean, stddev=stddev)
    elif type(init_type) is int:
        return tf.constant_initializer(value=init_type, dtype=tf.int32)
    else:
        return tf.constant_initializer(value=init_type, dtype=tf.float32)


def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def get_act_func(act_type='relu'):
    if type(act_type) is str:
        act_type = act_type.lower()
    assert act_type in {'sigmoid', 'softmax', 'relu', 'tanh', 'elu', 'selu', 'crelu', 'leacky_relu', None}
    if act_type == 'sigmoid':
        return tf.nn.sigmoid
    elif act_type == 'softmax':
        return tf.nn.softmax
    elif act_type == 'relu':
        return tf.nn.relu
    elif act_type == 'tanh':
        return tf.nn.tanh
    elif act_type == 'elu':
        return tf.nn.elu
    elif act_type == 'selu':
        return selu
    elif act_type == 'crelu':
        return tf.nn.crelu
    elif act_type == 'leaky_relu':
        return tf.nn.leaky_relu
    else:
        return lambda x: x


def unroll_pairwise(xv, num_fields):
    """
    :param xv: batch * n * k
    :return xv_p, xv_q: batch * pair * k
    """
    rows = []
    cols = []
    for i in range(num_fields - 1):
        for j in range(i + 1, num_fields):
            rows.append(i)
            cols.append(j)
    with tf.variable_scope('unroll_pairwise'):
        # batch * pair * k
        xv_p = tf.transpose(
            # pair * batch * k
            # TODO check gather's warning UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape.
            # This may consume a large amount of memory.
            # "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
            tf.gather(
                # num * batch * k
                tf.transpose(
                    xv, [1, 0, 2]),
                rows),
            [1, 0, 2])
        # batch * pair * k
        xv_q = tf.transpose(
            tf.gather(
                tf.transpose(
                    xv, [1, 0, 2]),
                cols),
            [1, 0, 2])
    return xv_p, xv_q


def unroll_field_aware(xv, num_fields):
    """
    :param xv: batch * n * (n - 1) * k
    :return xv_p, xv_q: batch * pair * k
    """
    rows = []
    cols = []
    for i in range(num_fields - 1):
        for j in range(i + 1, num_fields):
            rows.append([i, j - 1])
            cols.append([j, i])
    with tf.variable_scope('unroll_field_aware'):
        # batch * pair * k
        xv_p = tf.transpose(
            # pair * batch * k
            # TODO check gather's warning UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape.
            # This may consume a large amount of memory.
            # "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
            tf.gather_nd(
                # num * (num - 1) * batch * k
                tf.transpose(xv, [1, 2, 0, 3]),
                rows),
            [1, 0, 2])
        xv_q = tf.transpose(
            tf.gather_nd(
                tf.transpose(xv, [1, 2, 0, 3]),
                cols),
            [1, 0, 2])
    return xv_p, xv_q


class Model:
    inputs = None
    indices = None
    values = None
    labels = None
    logits = None
    preds = None
    training = None
    loss = None
    log_loss = None
    l2_loss = None
    train_gen = None
    valid_gen = None
    test_gen = None

    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                output_dim=1, init_type='xavier', l2_embed=0, loss_type='log_loss',
                 pos_weight=1., num_shards=0, input_norm=False, init_sparse=False, init_fused=False, loss_mode='mean'):
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.input_size = input_size or num_fields * 2
        self.field_types = field_types or ['cat'] * num_fields
        self.separator = separator or list(range(1, num_fields))
        self.output_dim = output_dim
        self.init_type = init_type
        self.l2_embed = l2_embed
        self.loss_type = loss_type
        self.pos_weight = pos_weight
        self.num_shards = num_shards
        self.input_norm = input_norm
        self.init_sparse = init_sparse
        self.init_fused = init_fused
        self.loss_mode = loss_mode

        self.embed_size = None
        self.nn_layers = None
        self.nn_input = None
        self.sub_nn_layers = None
        self.sub_nn_input = None

    def __str__(self):
        return self.__class__.__name__

    def def_placeholder(self, train_flag=False):
        with tf.variable_scope('input'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='inputs')
            self.indices, self.values = tf.split(self.inputs, 2, axis=1)
            self.indices = tf.to_int32(self.indices)
            self.values = tf.expand_dims(self.values, axis=2)
            self.values = tf.split(self.values, self.separator, axis=1)
            if self.input_norm:
                # TODO concat cat fields together
                for i in range(self.num_fields):
                    if self.field_types[i] == 'set':
                        field_cnt = tf.reduce_sum(tf.where(tf.greater(self.values[i], 0), 
                                    tf.ones_like(self.values[i]), tf.zeros_like(self.values[i])), axis=1, keep_dims=True)
                        self.values[i] *= tf.sqrt(1. / self.num_fields / tf.maximum(field_cnt, 1))
                    else:
                        self.values[i] *= np.sqrt(1. / self.num_fields)
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim], name='labels')
            if train_flag:
                self.training = tf.placeholder(dtype=tf.bool, name='training')

    def apply_mask(self, inputs, mask, reduce_sum=None, expand=False):
        # for i, flag in enumerate(reduce_sum or [x == 'set' for x in self.field_types]):
        for i in range(len(inputs)):
            inputs[i] *= mask[i]# if not expand else tf.expand_dims(mask[i], 2)
            inputs[i] = tf.reduce_sum(inputs[i], axis=1, keep_dims=True)
        return tf.concat(inputs, axis=1)

    def embedding_lookup(self, weight_flag=True, vector_flag=True, bias_flag=True, field_aware=False, dtype=tf.float32):
        assert hasattr(self, 'embed_size') if vector_flag else True, 'self.embed_size not found'
        partitioner = None if self.num_shards <= 1 else tf.fixed_size_partitioner(num_shards=self.num_shards)
        with tf.variable_scope('embedding' if not field_aware else 'field_aware_embedding', partitioner=partitioner):
            self.xw, self.xv, self.b = None, None, None
            if weight_flag:
                if not self.init_sparse:
                    initializer = get_initializer(init_type=self.init_type)
                    # TODO: check
                    # initializer = get_initializer(init_type='uniform', minval=-0.5, maxval=0.5)
                else:
                    # TODO: check
                    # maxval = np.sqrt(6 / (self.num_fields + 1))
                    # initializer = get_initializer(init_type='uniform', minval=-maxval, maxval=maxval)
                    initializer = get_initializer(init_type='uniform', minval=0, maxval=1)
                w = tf.get_variable(name='w', shape=[self.input_dim, 1], dtype=dtype, initializer=initializer,
                                    collections=WEIGHTS)
                # TODO try pass a list to embedding_lookup
                self.xw = tf.nn.embedding_lookup(w, self.indices)  # , partition_strategy='div')
                self.xw = tf.split(self.xw, self.separator, axis=1)
                self.xw = self.apply_mask(self.xw, self.values)
                tf.add_to_collection('EMBEDS', self.xw)
            if vector_flag:
                if not self.init_sparse:
                    initializer = get_initializer(init_type=self.init_type)
                    # TODO: check
                    # maxval = np.sqrt(1. / self.embed_size)
                    # initializer = get_initializer(init_type='uniform', minval=-maxval, maxval=maxval)
                else:
                    # TODO: check
                    # maxval = np.sqrt(6 / (self.num_fields + self.embed_size))
                    # initializer = get_initializer(init_type='uniform', minval=-maxval, maxval=maxval)
                    initializer = get_initializer(init_type='uniform', minval=0, maxval=np.sqrt(1. / self.embed_size))
                if not field_aware:
                    v = tf.get_variable(name='v', shape=[self.input_dim, self.embed_size], dtype=dtype,
                                        initializer=initializer, collections=WEIGHTS)
                else:
                    v = tf.get_variable(name='v', shape=[self.input_dim, (self.num_fields - 1) * self.embed_size],
                                        dtype=dtype, initializer=initializer, collections=WEIGHTS)
                    v = tf.reshape(v, [self.input_dim, self.num_fields - 1, self.embed_size])
                self.xv = tf.nn.embedding_lookup(v, self.indices)  # , partition_strategy='div')
                self.xv = tf.split(self.xv, self.separator, axis=1)
                self.xv = self.apply_mask(self.xv, self.values, expand=True)
                tf.add_to_collection('EMBEDS', self.xv)
            if bias_flag:
                self.b = tf.get_variable(name='b', shape=[1], dtype=dtype, initializer=get_initializer(init_type=0.),
                                         collections=BIASES)

    def def_log_loss(self):
        self.loss_type = self.loss_type.lower()
        assert self.loss_type in {'log_loss', 'weighted_log_loss'}, 'loss_type in {"log_loss", "weighted_log_loss"}'
        reduce_op = tf.reduce_mean if self.loss_mode == 'mean' else tf.reduce_sum
        if self.loss_type == 'log_loss':
            self.log_loss = reduce_op(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        else:
            self.log_loss = reduce_op(tf.nn.weighted_cross_entropy_with_logits(targets=self.labels, logits=self.logits,
                                                                               pos_weight=self.pos_weight))

    def def_l2_loss(self):
        self.l2_loss = tf.constant(0.)
        if self.l2_embed > 0:
            embeds = tf.get_collection('EMBEDS')
            if len(embeds) > 0:
                self.l2_loss += self.l2_embed * tf.add_n([tf.nn.l2_loss(v) for v in embeds])
        if hasattr(self, 'l2_kernel') and self.l2_kernel > 0:
            kernels = tf.get_collection('KERNELS')
            if len(kernels) > 0:
                self.l2_loss += self.l2_kernel * tf.add_n([tf.nn.l2_loss(v) for v in kernels])
        if hasattr(self, 'l2_sub_nn') and self.l2_sub_nn > 0:
            sub_nn_weights = tf.get_collection('SUB_NN_WEIGHTS')
            if len(sub_nn_weights) > 0:
                self.l2_loss += self.l2_sub_nn * tf.add_n([tf.nn.l2_loss(v) for v in sub_nn_weights])
        if hasattr(self, 'l2_nn') and self.l2_nn > 0:
            nn_weights = tf.get_collection('NN_WEIGHTS')
            if len(nn_weights) > 0:
                self.l2_loss += self.l2_nn * tf.add_n([tf.nn.l2_loss(v) for v in nn_weights])

    def def_inner_product(self, ):
        with tf.variable_scope('inner_product'):
            # batch * 1
            self.p = tf.reduce_sum(
                # batch * k
                tf.square(tf.reduce_sum(self.xv, 1)) -
                tf.reduce_sum(tf.square(self.xv), 1),
                axis=1, keep_dims=True)

    def def_kernel_product(self, kernel=None, dtype=tf.float32, unit_kernel=False, fix_kernel=False, kernel_type='mat'):
        """
        :param kernel: k * pair * k
        :return: batch * pair
        """
        xv_p, xv_q = unroll_pairwise(self.xv, self.num_fields)
        num_pairs = int(self.num_fields * (self.num_fields - 1) / 2)
        with tf.variable_scope('kernel_product'):
            if kernel is None:
                # if self.init_fused:
                #     # TODO: check
                #     # maxval = np.sqrt(3. / num_pairs / self.embed_size)
                #     maxval = 1 / self.embed_size
                #     initializer = get_initializer(init_type='uniform', minval=-maxval, maxval=maxval)
                # else:
                #     # TODO:
                #     maxval = np.sqrt(1. / self.embed_size)
                #     initializer = get_initializer(init_type='uniform', minval=-maxval, maxval=maxval)
                if kernel_type == 'mat':
                    shape = [self.embed_size, num_pairs, self.embed_size]
                    initializer = get_initializer(init_type='identity')
                elif kernel_type == 'vec':
                    shape = [num_pairs, self.embed_size]
                    initializer = get_initializer(init_type=1.)
                else:
                    shape = [num_pairs, 1]
                    initializer = get_initializer(init_type=1.)
                kernel = tf.get_variable(name='kernel', shape=shape, dtype=dtype, initializer=initializer,
                                         collections=KERNELS, trainable=not fix_kernel)
            if kernel_type == 'mat':
                # batch * 1 * pair * k
                xv_pe = tf.expand_dims(xv_p, 1)
                # batch * pair * k
                pd = tf.transpose(
                        # batch * k * pair
                        tf.reduce_sum(
                            # batch * k * pair * k
                            tf.multiply(
                                xv_pe, kernel),
                            -1),
                        [0, 2, 1])
            else:
                pd = xv_p * kernel
            if unit_kernel:
                p_norm = tf.sqrt(tf.reduce_sum(tf.square(xv_p), 2, keep_dims=True))
                pd_norm = tf.sqrt(tf.reduce_sum(tf.square(pd), 2, keep_dims=True))
                pd *= p_norm / pd_norm
            # batch * pair
            self.kp = tf.reduce_sum(
                # batch * pair * k
                tf.multiply(pd, xv_q),
                -1)

    def def_nn_layers(self, nn_input_dim=None, dtype=tf.float32):
        assert hasattr(self, 'nn_input'), 'self.nn_input not found'
        assert hasattr(self, 'nn_layers'), 'self.nn_layers not found'
        with tf.variable_scope('network'):
            if nn_input_dim is None:
                nn_input_dim = self.nn_input.get_shape().as_list()[-1]
            self.h = self.nn_input
            h_dim = nn_input_dim
            n_layer = 0
            for l_type, l_param in self.nn_layers:
                assert l_type in {'full', 'act', 'drop', 'conv', 'flat', 'ln', 'bn'}, \
                    'a layer should be {full, act, drop, conv, flat, ln, bn}'
                if l_type == 'full':
                    with tf.variable_scope('layer_%d' % n_layer) as scope:
                        n_layer += 1
                        wi = tf.get_variable(name='w', shape=[h_dim, l_param], dtype=dtype,
                                             initializer=get_initializer(init_type=self.init_type),
                                             collections=NN_WEIGHTS)
                        bi = tf.get_variable(name='b', shape=[l_param], dtype=dtype,
                                             initializer=get_initializer(init_type=0.), collections=BIASES)
                        self.h = tf.matmul(self.h, wi) + bi
                        h_dim = l_param
                elif l_type == 'act':
                    with tf.variable_scope(scope):
                        self.h = get_act_func(act_type=l_param)(self.h)
                elif l_type == 'drop':
                    with tf.variable_scope(scope):
                        self.h = tf.nn.dropout(self.h, tf.where(self.training, l_param, 1.))
                elif l_type == 'conv':
                    with tf.variable_scope('conv_%d' % n_layer) as scope:
                        n_layer += 1
                        wi = tf.get_variable(name='w', shape=[l_param[0], h_dim, l_param[1]], dtype=dtype,
                                             initializer=get_initializer(init_type=self.init_type),
                                             collections=NN_WEIGHTS)
                        self.h = tf.nn.conv1d(self.h, wi, stride=1, padding='VALID')  # + bi
                        h_dim = l_param[1]
                elif l_type == 'flat':
                    with tf.variable_scope(scope):
                        h_dim = np.prod(np.array(self.h.get_shape().as_list())[list(l_param)])
                        self.h = tf.reshape(self.h, [-1, h_dim])
                elif l_type == 'ln' or l_type == 'bn':
                    with tf.variable_scope(scope):
                        scope_name = 'layer_norm' if l_type == 'ln' else 'batch_norm'
                        axes = [1] if l_type == 'ln' else [0]
                        with tf.variable_scope(scope_name):
                            layer_mean, layer_var = tf.nn.moments(self.h, axes=axes, keep_dims=True)
                            scale = tf.get_variable(name='scale', shape=[h_dim], dtype=dtype,
                                                    initializer=get_initializer(init_type=1.), collections=NN_WEIGHTS)
                            self.h = (self.h - layer_mean) / tf.sqrt(layer_var)
                            self.h = self.h * scale
                            if l_param != 'no_bias':
                                shift = tf.get_variable(name='shift', shape=[h_dim], dtype=dtype,
                                                        initializer=get_initializer(init_type=0.), collections=BIASES)
                                self.h = self.h + shift

    def def_sub_nn_layers(self, sub_nn_input_dim=None, sub_nn_num=None, dtype=tf.float32):
        assert hasattr(self, 'sub_nn_input'), 'self.sub_nn_input not found'
        assert hasattr(self, 'sub_nn_layers'), 'self.sub_nn_layers not found'
        with tf.variable_scope('sub_network'):
            if sub_nn_input_dim is None:
                sub_nn_input_dim = self.sub_nn_input.get_shape().as_list()[-1]
            if sub_nn_num is None:
                sub_nn_num = self.sub_nn_input.get_shape().as_list()[-2]
            # batch * pair * 2k -> pair * batch * 2k
            self.sh = tf.transpose(self.sub_nn_input, [1, 0, 2])
            sh_dim = sub_nn_input_dim
            sh_num = sub_nn_num
            n_layer = 0
            for sl_type, sl_param in self.sub_nn_layers:
                assert sl_type in {'full', 'act', 'drop', 'ln', 'bn'}, 'a layer should be {full, act, drop, ln, bn}'
                if sl_type == 'full':
                    with tf.variable_scope('layer_%d' % n_layer) as scope:
                        n_layer += 1
                        if not self.init_fused:
                            maxval = np.sqrt(6. / (sh_dim + sl_param))
                            initializer = get_initializer(init_type='uniform', minval=-maxval, maxval=maxval)
                        else:
                            initializer = get_initializer(init_type=self.init_type)
                        wi = tf.get_variable(name='w', shape=[sh_num, sh_dim, sl_param], dtype=dtype,
                                             initializer=initializer, collections=SUB_NN_WEIGHTS)
                        bi = tf.get_variable(name='b', shape=[sh_num, 1, sl_param], dtype=dtype,
                                             initializer=get_initializer(init_type=0.), collections=BIASES)
                        self.sh = tf.matmul(self.sh, wi) + bi
                        sh_dim = sl_param
                elif sl_type == 'act':
                    with tf.variable_scope(scope):
                        self.sh = get_act_func(act_type=sl_param)(self.sh)
                elif sl_type == 'drop':
                    with tf.variable_scope(scope):
                        self.sh = tf.nn.dropout(self.sh, tf.where(self.training, sl_param, 1.))
                elif sl_type == 'ln' or sl_type == 'bn':
                    with tf.variable_scope(scope):
                        scope_name = 'layer_norm' if sl_type == 'ln' else 'batch_norm'
                        axes = [0, 2] if sl_type == 'ln' else [0, 1]
                        with tf.variable_scope(scope_name):
                            layer_mean, layer_var = tf.nn.moments(self.sh, axes=axes, keep_dims=True)
                            out_dim = [sh_num, 1, sh_dim] if 'no_share' in sl_param else [sh_dim]
                            scale = tf.get_variable(name='scale', shape=out_dim, dtype=dtype,
                                                    initializer=get_initializer(init_type=1.), collections=SUB_NN_WEIGHTS)
                            self.sh = (self.sh - layer_mean) / tf.sqrt(layer_var)
                            self.sh = self.sh * scale
                            if 'no_bias' not in sl_param:
                                shift = tf.get_variable(name='shift', shape=out_dim, dtype=dtype,
                                                        initializer=get_initializer(init_type=0.), collections=BIASES)
                                self.sh = self.sh + shift
            # pair * batch * m -> batch * pair * m
            self.sh = tf.transpose(self.sh, [1, 0, 2])

    def eval(self, gen, sess):
        labels = []
        preds = []
        start_time = time.time()
        for xs, ys in gen:
            feed = {self.inputs: xs, self.labels: ys}
            if self.training is not None:
                feed[self.training] = False
            _preds_ = sess.run(self.preds, feed_dict=feed)
            labels.append(ys)
            preds.append(_preds_)
        labels = np.vstack(labels)
        preds = np.vstack(preds)
        eps = 1e-6
        preds[preds < eps] = eps
        preds[preds > 1 - eps] = 1 - eps
        _loss_ = log_loss(y_true=labels, y_pred=preds)
        _auc_ = roc_auc_score(y_true=labels, y_score=preds)
        print('%s-Loss: %2.4f, AUC: %2.4f, Elapsed: %s' %
              (gen.gen_type.capitalize(), _loss_, _auc_, str(timedelta(seconds=(time.time() - start_time)))))
        return _loss_, _auc_


class LR(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                output_dim=1, init_type='xavier', l2_embed=0, loss_type='log_loss',
                 pos_weight=1., num_shards=0, input_norm=False, init_sparse=False, init_fused=False, loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)

        self.def_placeholder(train_flag=False)

        self.embedding_lookup(vector_flag=False)

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class FM(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, output_dim=1, init_type='xavier', l2_embed=0,
                 loss_type='log_loss', pos_weight=1., num_shards=0, input_norm=False, init_sparse=False,
                 init_fused=False, loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size

        self.def_placeholder(train_flag=False)

        self.embedding_lookup()

        self.def_inner_product()

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b + 0.5 * self.p

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class FFM(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=2, output_dim=1, init_type='xavier', l2_embed=0,
                 loss_type='log_loss', pos_weight=1., num_shards=0, input_norm=False, init_sparse=False,
                 init_fused=False, loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size

        self.def_placeholder(train_flag=False)

        self.embedding_lookup(field_aware=True)

        xv_p, xv_q = unroll_field_aware(self.xv, self.num_fields)
        with tf.variable_scope('inner_product'):
            # batch * pair
            self.p = tf.reduce_sum(tf.multiply(xv_p, xv_q), 2)

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b + tf.reduce_sum(self.p, axis=1, keep_dims=True)

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class KFM(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, output_dim=1, init_type='xavier', l2_embed=0, l2_kernel=0,
                 unit_kernel=True, loss_type='log_loss', pos_weight=1., num_shards=0, input_norm=False,
                 init_sparse=False, init_fused=False, loss_mode='mean', fix_kernel=False, kernel_type='mat'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size
        self.l2_kernel = l2_kernel

        self.def_placeholder(train_flag=False)

        self.embedding_lookup()

        self.def_kernel_product(unit_kernel=unit_kernel, fix_kernel=fix_kernel, kernel_type=kernel_type)

        self.logits = tf.reduce_sum(self.kp, axis=1, keep_dims=True) + tf.reduce_sum(self.xw, axis=1) + self.b

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class NFM(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, sub_nn_layers=None, output_dim=1, init_type='xavier',
                 l2_embed=0, l2_sub_nn=0, loss_type='log_loss', pos_weight=1., num_shards=0, input_norm=False,
                 init_sparse=False, init_fused=False, loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size
        self.l2_sub_nn = l2_sub_nn
        self.sub_nn_layers = sub_nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup()

        xv_p, xv_q = unroll_pairwise(self.xv, num_fields=self.num_fields)
        # batch * pair * 2k
        self.sub_nn_input = tf.concat([xv_p, xv_q], axis=2)

        self.def_sub_nn_layers()

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b + tf.reduce_sum(self.sh, axis=1)

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class FNN(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_embed=0, l2_nn=0, loss_type='log_loss', pos_weight=1., num_shards=0, input_norm=False,
                 init_sparse=False, init_fused=False, loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size
        self.l2_nn = l2_nn
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(weight_flag=False, bias_flag=False)

        self.nn_input = tf.reshape(self.xv, [-1, self.num_fields * self.embed_size])

        self.def_nn_layers()

        self.logits = self.h

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class CCPM(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_embed=0, l2_nn=0, loss_type='log_loss', pos_weight=1., num_shards=0, input_norm=False,
                 init_sparse=False, init_fused=False, loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size
        self.l2_nn = l2_nn
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(weight_flag=False, bias_flag=False)

        # batch * field * k
        self.nn_input = self.xv

        self.def_nn_layers()

        self.logits = self.h

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class DeepFM(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_embed=0, l2_nn=0, loss_type='log_loss', pos_weight=1., num_shards=0, input_norm=False,
                 init_sparse=False, init_fused=False, loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size
        self.l2_nn = l2_nn
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(bias_flag=False)

        self.def_inner_product()

        self.nn_input = tf.reshape(self.xv, [-1, self.num_fields * self.embed_size])

        self.def_nn_layers()

        self.logits = tf.reduce_sum(self.xw, axis=1) + 0.5 * self.p + self.h

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class IPNN(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_embed=0, l2_nn=0, loss_type='log_loss', pos_weight=1., num_shards=0, input_norm=False,
                 init_sparse=False, init_fused=False, loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size
        self.l2_nn = l2_nn
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(weight_flag=False, bias_flag=False)

        xv_p, xv_q = unroll_pairwise(self.xv, self.num_fields)
        with tf.variable_scope('inner_product'):
            # batch * pair
            self.p = tf.reduce_sum(tf.multiply(xv_p, xv_q), 2)

        self.nn_input = tf.concat([tf.reshape(self.xv, [-1, self.num_fields * self.embed_size]), self.p], axis=1)

        self.def_nn_layers()

        self.logits = self.h

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class KPNN(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_embed=0, l2_kernel=0, unit_kernel=True, l2_nn=0, loss_type='log_loss', pos_weight=1., num_shards=0,
                 input_norm=False, init_sparse=False, init_fused=False, loss_mode='mean', fix_kernel=False, kernel_type='mat'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size
        self.l2_kernel = l2_kernel
        self.unit_kernel = unit_kernel
        self.fix_kernel = fix_kernel
        self.l2_nn = l2_nn
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(weight_flag=False, bias_flag=False)

        self.def_kernel_product(unit_kernel=unit_kernel, fix_kernel=fix_kernel, kernel_type=kernel_type)

        self.nn_input = tf.concat([tf.reshape(self.xv, [-1, self.num_fields * self.embed_size]), self.kp], axis=1)

        self.def_nn_layers()

        self.logits = self.h

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class PIN(Model):
    def __init__(self, input_dim, num_fields, input_size=None, field_types=None, separator=None, 
                embed_size=10, sub_nn_layers=None, nn_layers=None, output_dim=1,
                 init_type='xavier', l2_embed=0, l2_sub_nn=0, l2_nn=0, loss_type='log_loss', pos_weight=1.,
                 num_shards=0, wide=False, prod=True, input_norm=False, init_sparse=False, init_fused=False,
                 loss_mode='mean'):
        Model.__init__(self, input_dim, num_fields, input_size, field_types, separator, 
                        output_dim, init_type, l2_embed, loss_type, pos_weight, num_shards,
                       input_norm, init_sparse, init_fused, loss_mode)
        self.embed_size = embed_size
        self.l2_sub_nn = l2_sub_nn
        self.l2_nn = l2_nn
        self.sub_nn_layers = sub_nn_layers
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        if not wide:
            self.embedding_lookup(weight_flag=False, bias_flag=False)
        else:
            self.embedding_lookup()
            self.def_inner_product()
            self.wide = tf.reduce_sum(self.xw, axis=1) + self.b + 0.5 * self.p

        xv_p, xv_q = unroll_pairwise(self.xv, num_fields=self.num_fields)
        # TODO check this
        if prod:
            # batch * pair * 3k
            self.sub_nn_input = tf.concat([xv_p, xv_q, xv_p * xv_q], axis=2)
        else:
            self.sub_nn_input = tf.concat([xv_p, xv_q], axis=2)

        self.def_sub_nn_layers()

        sh_dim = self.sh.get_shape().as_list()[-1]
        sh_num = self.sh.get_shape().as_list()[-2]
        self.nn_input = tf.reshape(self.sh, [-1, sh_num * sh_dim])

        self.def_nn_layers()

        self.logits = self.h

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss
