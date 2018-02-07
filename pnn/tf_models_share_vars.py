from __future__ import division
from __future__ import print_function

import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

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


def get_initializer(init_type='xavier', minval=-0.001, maxval=0.001, mean=0, stddev=0.001):
    if type(init_type) is str:
        init_type = init_type.lower()
    assert init_type in {'xavier', 'uniform', 'normal'} if type(init_type) is str \
        else type(init_type) in {int, float}, 'init type: {"xavier", "uniform", "normal", int, float}'
    if init_type == 'xavier':
        return tf.contrib.layers.xavier_initializer(uniform=True)
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
    with tf.variable_scope('unroll_pairwise'):
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
    with tf.variable_scope('unroll_field_aware'):
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
    train_gen = None
    valid_gen = None
    test_gen = None

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
        self.sub_nn_layers = None
        self.sub_nn_input = None

    def __str__(self):
        return self.__class__.__name__

    def def_placeholder(self, train_flag=False):
        with tf.variable_scope('input'):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.num_fields], name='inputs')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim], name='labels')
            if train_flag:
                self.training = tf.placeholder(dtype=tf.bool, name='training')

    def embedding_lookup(self, weight_flag=True, vector_flag=True, bias_flag=True, field_aware=False, dtype=tf.float32):
        assert hasattr(self, 'embed_size') if vector_flag else True, 'self.embed_size not found'
        with tf.variable_scope('embedding' if not field_aware else 'field_aware_embedding'):
            self.xw, self.xv, self.b = None, None, None
            if weight_flag:
                w = tf.get_variable(name='w', shape=[self.input_dim, 1], dtype=dtype,
                                    initializer=get_initializer(init_type=self.init_type))
                self.xw = tf.gather(w, self.inputs)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.xw)
            if vector_flag:
                if not field_aware:
                    v = tf.get_variable(name='v', shape=[self.input_dim, self.embed_size], dtype=dtype,
                                        initializer=get_initializer(init_type=self.init_type))
                    self.xv = tf.gather(v, self.inputs)
                else:
                    v = tf.get_variable(name='v', shape=[self.input_dim, (self.num_fields - 1) * self.embed_size],
                                        dtype=dtype, initializer=get_initializer(init_type=self.init_type))
                    v = tf.reshape(v, [self.input_dim, self.num_fields - 1, self.embed_size])
                    self.xv = tf.gather(v, self.inputs)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.xv)
            if bias_flag:
                shape = [1]
                self.b = tf.get_variable(name='b', shape=shape, dtype=dtype, initializer=get_initializer(init_type=0.),
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
        if self.l2_scale > 0:
            weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
            if len(weights) > 0:
                self.l2_loss = self.l2_scale * tf.add_n([tf.nn.l2_loss(v) for v in weights])
            else:
                self.l2_loss = tf.constant(0.)
        else:
            self.l2_loss = tf.constant(0.)

    def def_inner_product(self, ):
        with tf.variable_scope('inner_product'):
            # batch * 1
            self.p = tf.reduce_sum(
                # batch * k
                tf.square(tf.reduce_sum(self.xv, 1)) -
                tf.reduce_sum(tf.square(self.xv), 1),
                axis=1, keep_dims=True)

    def def_kernel_product(self, kernel=None, dtype=tf.float32):
        """
        :param kernel: k * pair * k
        :return: batch * pair
        """
        xv_p, xv_q = unroll_pairwise(self.xv, self.num_fields)
        num_pairs = int(self.num_fields * (self.num_fields - 1) / 2)
        with tf.variable_scope('kernel_product'):
            if kernel is None:
                maxval = np.sqrt(3. / self.embed_size)
                shape = [self.embed_size, num_pairs, self.embed_size]
                kernel = tf.get_variable(name='kernel', shape=shape, dtype=dtype, initializer=get_initializer(
                    init_type='uniform', minval=-maxval, maxval=maxval), collections=WEIGHTS)
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
                                             collections=WEIGHTS)
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
                                             initializer=get_initializer(init_type=self.init_type), collections=WEIGHTS)
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
                                                    initializer=get_initializer(init_type=1.), collections=WEIGHTS)
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
                        maxval = np.sqrt(6. / (sh_dim + sl_param))
                        wi = tf.get_variable(name='w', shape=[sh_num, sh_dim, sl_param], dtype=dtype,
                                             initializer=get_initializer(init_type='uniform', minval=-maxval,
                                                                         maxval=maxval), collections=WEIGHTS)
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
                                                    initializer=get_initializer(init_type=1.), collections=WEIGHTS)
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

        self.def_inner_product()

        self.logits = tf.reduce_sum(self.xw, axis=1) + self.b + 0.5 * self.p

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
        with tf.variable_scope('inner_product'):
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
    def __init__(self, input_dim, num_fields, embed_size=10, sub_nn_layers=None, output_dim=1, init_type='xavier',
                 l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
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


class CCPM(Model):
    def __init__(self, input_dim, num_fields, embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
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
    def __init__(self, input_dim, num_fields, embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
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
    def __init__(self, input_dim, num_fields, embed_size=10, nn_layers=None, output_dim=1, init_type='xavier',
                 l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
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
    def __init__(self, input_dim, num_fields, embed_size=10, sub_nn_layers=None, nn_layers=None, output_dim=1,
                 init_type='xavier', l2_scale=0, loss_type='log_loss', pos_weight=1.):
        Model.__init__(self, input_dim, num_fields, output_dim, init_type, l2_scale, loss_type, pos_weight)
        self.embed_size = embed_size
        self.sub_nn_layers = sub_nn_layers
        self.nn_layers = nn_layers

        self.def_placeholder(train_flag=True)

        self.embedding_lookup(weight_flag=False, bias_flag=False)

        xv_p, xv_q = unroll_pairwise(self.xv, num_fields=self.num_fields)
        # batch * pair * 2k
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
