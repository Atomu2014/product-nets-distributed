from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

WEIGHTS = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS]
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
    elif model_name == 'deepfm':
        return DeepFM(**model_param)
    elif model_name == 'ipnn':
        return IPNN(**model_param)
    elif model_name == 'kpnn':
        return KPNN(**model_param)
    elif model_name == 'pin':
        return PIN(**model_param)


def get_init_value(init_type='xavier', shape=None, mode='fan_avg',
                   minval=-0.001, maxval=0.001, mean=0, stddev=0.001):
    if type(init_type) is str:
        init_type = init_type.lower()
    assert init_type in {'xavier', 'uniform', 'normal'} if type(init_type) is str \
        else type(init_type) in {int, float}, 'init type: {"xavier", "uniform", "normal", int, float}'
    if init_type == 'xavier':
        mode = mode.lower()
        assert mode in {'fan_avg', 'fan_in', 'fan_out'}
        if len(shape) == 1:
            shape.append(1)
        if mode == 'fan_avg':
            maxval = math.sqrt(6. / (shape[-2] + shape[-1]))
        elif mode == 'fan_in':
            maxval = math.sqrt(3. / shape[-2])
        else:
            maxval = math.sqrt(3. / shape[-1])
        ret_val = tf.random_uniform(shape=shape, minval=-maxval, maxval=maxval)
        return ret_val
    elif init_type == 'uniform':
        return tf.random_uniform(shape=shape, minval=minval, maxval=maxval)
    elif init_type == 'normal':
        return tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    elif type(init_type) is int:
        return tf.constant(value=init_type, dtype=tf.int32, shape=shape)
    else:
        return tf.constant(value=init_type, dtype=tf.float32, shape=shape)


def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def get_act_func(act_type='relu'):
    if type(act_type) is str:
        act_type = act_type.lower()
    assert act_type in {'sigmoid', 'softmax', 'relu', 'tanh', 'elu', 'selu', None}
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
    with tf.name_scope('unroll_pairwise'):
        # batch * pair * k
        xv_p = tf.transpose(
            # pair * batch * k
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
    with tf.name_scope('unroll_field_aware'):
        # batch * pair * k
        xv_p = tf.transpose(
            # pair * batch * k
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
    labels = None
    logits = None
    preds = None
    training = None
    loss = None
    log_loss = None
    l2_loss = None

    def __init__(self, input_dim, num_fields, output_dim=1, init_type='xavier', l2_scale=0, loss_type='log_loss',
                 pos_weight=1.):
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.output_dim = output_dim
        self.init_type = init_type
        self.l2_scale = l2_scale
        self.loss_type = loss_type
        self.pos_weight = pos_weight

        self.embed_size = None
        self.nn_layers = None
        self.nn_input = None

    def __str__(self):
        return self.__class__.__name__

    def def_placeholder(self, train_flag=False):
        with tf.name_scope('input'):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.num_fields], name='inputs')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim], name='labels')
            if train_flag:
                self.training = tf.placeholder(dtype=tf.bool, name='training')

    def embedding_lookup(self, weight_flag=True, vector_flag=True, bias_flag=True, field_aware=False, dtype=tf.float32):
        assert hasattr(self, 'embed_size') if vector_flag else True, 'self.embed_size not found'
        with tf.variable_scope('embedding' if not field_aware else 'field_aware_embedding'):
            self.xw, self.xv, self.b = None, None, None
            if weight_flag:
                w = tf.Variable(initial_value=get_init_value(init_type=self.init_type, shape=[self.input_dim]),
                                name='w', dtype=dtype, collections=WEIGHTS)
                self.xw = tf.gather(w, self.inputs)
            if vector_flag:
                if not field_aware:
                    v = tf.Variable(
                        initial_value=get_init_value(init_type=self.init_type, shape=[self.input_dim, self.embed_size]),
                        name='v', dtype=dtype, collections=WEIGHTS)
                    self.xv = tf.gather(v, self.inputs)
                else:
                    v = tf.Variable(
                        initial_value=get_init_value(init_type=self.init_type,
                                                     shape=[self.input_dim, (self.num_fields - 1) * self.embed_size]),
                        name='v', dtype=dtype, collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
                    v = tf.reshape(v, [self.input_dim, self.num_fields - 1, self.embed_size])
                    self.xv = tf.gather(v, self.inputs)
            if bias_flag:
                self.b = tf.Variable(initial_value=get_init_value(init_type=0., shape=[1]), name='b', dtype=dtype,
                                     collections=BIASES)

    def def_log_loss(self):
        self.loss_type = self.loss_type.lower()
        assert self.loss_type in {'log_loss', 'weighted_log_loss'}, 'loss_type in {"log_loss", "weighted_log_loss"}'
        if self.loss_type == 'log_loss':
            self.log_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        else:
            self.log_loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(targets=self.labels, logits=self.logits,
                                                         pos_weight=self.pos_weight))

    def def_l2_loss(self):
        if self.l2_loss > 0:
            weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
            if len(weights) > 0:
                self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weights])
            else:
                self.l2_loss = tf.constant(0.)
        else:
            self.l2_loss = tf.constant(0.)

    def def_kernel_product(self, kernel=None, dtype=tf.float32):
        """
        :param kernel: k * pair * k
        :return: batch * pair
        """
        xv_p, xv_q = unroll_pairwise(self.xv, self.num_fields)
        num_pairs = int(self.num_fields * (self.num_fields - 1) / 2)
        with tf.name_scope('kernel_product'):
            if kernel is None:
                maxval = math.sqrt(3. / self.embed_size)
                kernel = tf.Variable(
                    get_init_value(init_type='uniform', shape=[self.embed_size, num_pairs, self.embed_size],
                                   minval=-maxval, maxval=maxval), name='kernel', dtype=dtype, collections=WEIGHTS)
            # batch * 1 * pair * k
            xv_p = tf.expand_dims(xv_p, 1)
            # batch * pair
            self.kp = tf.reduce_sum(
                # batch * pair * k
                tf.multiply(
                    # batch * pair * k
                    tf.transpose(
                        # batch * k * pair
                        tf.reduce_sum(
                            # batch * k * pair * k
                            tf.multiply(
                                xv_p, kernel),
                            -1),
                        [0, 2, 1]),
                    xv_q),
                -1)
            # if add_bias:
            #     bias = tf.Variable(get_init_value(init_type=0., shape=[num_pairs]), name='bias', dtype=dtype,
            # collections=BIASES)
            #     self.p += bias
            # if reduce_sum:
            #     self.p = tf.reduce_sum(self.p, 1)

    def def_nn_layers(self, nn_input_dim=None, dtype=tf.float32):
        assert hasattr(self, 'nn_input'), 'self.nn_input not found'
        assert hasattr(self, 'nn_layers'), 'self.nn_layers not found'
        if nn_input_dim is None:
            nn_input_dim = self.nn_input.get_shape().as_list()[-1]
        self.h = self.nn_input
        h_dim = nn_input_dim
        for l_type, l_param in self.nn_layers:
            assert l_type in {'full', 'act', 'drop'}, 'a layer should be {full, act, drop}'
            if l_type == 'full':
                with tf.name_scope('hidden') as scope:
                    init_val = get_init_value(init_type=self.init_type, shape=[h_dim, l_param])
                    wi = tf.Variable(init_val, name='w',
                                     dtype=dtype, collections=WEIGHTS)
                    bi = tf.Variable(get_init_value(init_type=0., shape=[l_param]), name='b', dtype=dtype,
                                     collections=BIASES)
                    self.h = tf.matmul(self.h, wi) + bi
                    h_dim = l_param
            elif l_type == 'act':
                with tf.name_scope(scope):
                    self.h = get_act_func(act_type=l_param)(self.h)
            elif l_type == 'drop':
                with tf.name_scope(scope):
                    self.h = tf.nn.dropout(self.h, tf.where(self.training, l_param, 1.))


class LR(Model):
    def __init__(self, input_dim, num_fields, output_dim=1, init_type='xavier', l2_scale=0, loss_type='log_loss',
                 pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)

        self.def_placeholder(train_flag=False)

        self.embedding_lookup(vector_flag=False)

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class FM(Model):
    def __init__(self, input_dim, num_fields, embed_size=10, output_dim=1, init_type='xavier', l2_scale=0,
                 loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size

        self.def_placeholder(train_flag=False)

        self.embedding_lookup()

        with tf.name_scope('inner_product'):
            self.p = tf.square(tf.reduce_sum(self.xv, 1)) - tf.reduce_sum(tf.square(self.xv), 1)

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b + 0.5 * tf.reduce_sum(self.p, axis=1)

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class FFM(Model):
    def __init__(self, input_dim, num_fields, embed_size=2, output_dim=1, init_type='xavier', l2_scale=0,
                 loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size

        self.def_placeholder(train_flag=False)

        self.embedding_lookup(field_aware=True)

        xv_p, xv_q = unroll_field_aware(self.xv, self.num_fields)
        with tf.name_scope('inner_product'):
            # batch * pair
            self.p = tf.reduce_sum(tf.multiply(xv_p, xv_q), 2)

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b + tf.reduce_sum(self.p, axis=1, keep_dims=True)

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class KFM(Model):
    def __init__(self, input_dim, num_fields, embed_size=10, output_dim=1, init_type='xavier', l2_scale=0,
                 loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size

        self.def_placeholder(train_flag=False)

        self.embedding_lookup()

        self.def_kernel_product()

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b + tf.reduce_sum(self.kp, axis=1, keep_dims=True)

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class NFM(Model):
    def __init__(self, input_dim, num_fields, embed_size=10, output_dim=1, init_type='xavier', l2_scale=0,
                 loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
        # TODO: implement sub-net


class FNN(Model):
    def __init__(self, input_dim, num_fields, embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
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


class DeepFM(Model):
    def __init__(self, input_dim, num_fields, embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(bias_flag=False)

        with tf.name_scope('inner_product'):
            self.p = tf.square(tf.reduce_sum(self.xv, 1)) - tf.reduce_sum(tf.square(self.xv), 1)

        self.nn_input = tf.reshape(self.xv, [-1, self.num_fields * self.embed_size])

        self.def_nn_layers()

        self.logits = tf.reduce_sum(self.xw, axis=1) + 0.5 * tf.reduce_sum(self.p, axis=1) + self.h

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class IPNN(Model):
    def __init__(self, input_dim, num_fields, embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(weight_flag=False, bias_flag=False)

        xv_p, xv_q = unroll_pairwise(self.xv, self.num_fields)
        with tf.name_scope('inner_product'):
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
    def __init__(self, input_dim, num_fields, embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(weight_flag=False, bias_flag=False)

        self.def_kernel_product()

        self.nn_input = tf.concat([tf.reshape(self.xv, [-1, self.num_fields * self.embed_size]), self.kp], axis=1)

        self.def_nn_layers()

        self.logits = self.h

        self.preds = tf.sigmoid(self.logits)

        self.def_log_loss()

        self.def_l2_loss()

        self.loss = self.log_loss + self.l2_loss


class PIN(Model):
    def __init__(self, input_dim, num_fields, embed_size=10, output_dim=1, init_type='xavier', l2_scale=0,
                 loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
        # TODO: implement sub-net
