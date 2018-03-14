from __future__ import division
from __future__ import print_function

import json
import os
import sys
import time
from datetime import timedelta, datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

import __init__

sys.path.append(__init__.config['data_path'])
from datasets import as_dataset
from print_hook import PrintHook
from tf_models_share_vars import as_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_shards', 1, 'Number of variable partitions')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of variable partitions')
tf.app.flags.DEFINE_bool('sparse_grad', False, 'Apply sparse gradient')

tf.app.flags.DEFINE_string('logdir', '../log', 'Directory for storing mnist data')
tf.app.flags.DEFINE_string('prefix', '', 'Prefix for logdir')
tf.app.flags.DEFINE_bool('restore', False, 'Restore from logdir')
tf.app.flags.DEFINE_bool('val', False, 'If True, use validation set, else use test set')
tf.app.flags.DEFINE_float('val_ratio', 0., 'Validation ratio')

tf.app.flags.DEFINE_string('optimizer', 'adagrad', 'Optimizer')
tf.app.flags.DEFINE_float('epsilon', 1e-8, 'Epsilon for adam')
tf.app.flags.DEFINE_float('init_val', 0.1, 'Initial accumulator value for adagrad')
tf.app.flags.DEFINE_float('learning_rate', 0.2, 'Learning rate')
tf.app.flags.DEFINE_string('loss_mode', 'sum', 'Loss = mean, sum')

tf.app.flags.DEFINE_integer('batch_size', 16, 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 512, 'Testing batch size')
tf.app.flags.DEFINE_string('dataset', 'criteo_challenge', 'Dataset = ipinyou, avazu, criteo, criteo_9d, criteo_16d"')
tf.app.flags.DEFINE_string('model', 'kfm', 'Model type = lr, fm, ffm, kfm, nfm, fnn, ccpm, deepfm, ipnn, kpnn, pin')

tf.app.flags.DEFINE_bool('input_norm', True, 'Input normalization')
tf.app.flags.DEFINE_bool('init_sparse', True, 'Init sparse layer')
tf.app.flags.DEFINE_bool('init_fused', False, 'Init fused layer')

tf.app.flags.DEFINE_integer('embed_size', 64, 'Embedding size')
tf.app.flags.DEFINE_string('nn_layers', '[' + '["full", 2048],  ["act", "relu"], '*5 + '["full", 1]]', 'Network structure')
tf.app.flags.DEFINE_string('sub_nn_layers', '[["full", 60], ["ln", ""], ["act", "relu"], ["full", 5],  ["ln", ""]]', 'Sub-network structure')
tf.app.flags.DEFINE_float('l2_embed', 2e-5, 'L2 regularization')
tf.app.flags.DEFINE_float('l2_kernel', 1e-5, 'L2 regularization for kernels')
tf.app.flags.DEFINE_bool('wide', True, 'Wide term for pin')
tf.app.flags.DEFINE_bool('prod', True, 'Use product term as sub-net input')
tf.app.flags.DEFINE_string('kernel_type', 'vec', 'Kernel type = mat, vec, num')
tf.app.flags.DEFINE_bool('unit_kernel', False, 'Kernel in unit ball')
tf.app.flags.DEFINE_bool('fix_kernel', False, 'Fix kernel')

tf.app.flags.DEFINE_integer('num_rounds', 4, 'Number of training rounds')
tf.app.flags.DEFINE_integer('eval_level', 0, 'Evaluating frequency level')
tf.app.flags.DEFINE_float('decay', 1., 'Learning rate decay')
tf.app.flags.DEFINE_integer('log_frequency', 10000, 'Logging frequency')


def get_logdir(FLAGS):
    if FLAGS.restore:
        logdir = FLAGS.logdir
    else:
        logdir = '%s/%s/%s/%s' % (
            FLAGS.logdir, FLAGS.dataset, FLAGS.model, FLAGS.prefix + datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = open(logdir + '/log', 'a')
    return logdir, logfile


def redirect_stdout(logfile):
    def MyHookOut(text):
        logfile.write(text)
        logfile.flush()
        return 1, 0, text

    phOut = PrintHook()
    phOut.Start(MyHookOut)


def get_optimizer(opt, lr):
    opt = opt.lower()
    eps = FLAGS.epsilon
    init_val = FLAGS.init_val
    if opt == 'sgd' or opt == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif opt == 'adam':
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)
    elif opt == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=init_val)


def sparse_grads_mean(grads_and_vars):
    indices = []
    values = []
    dense_shape = grads_and_vars[0][0].dense_shape
    n = len(grads_and_vars)
    for g, _ in grads_and_vars:
        indices.append(g.indices)
        values.append(g.values / n)
    indices = tf.concat(indices, axis=0)
    values = tf.concat(values, axis=0)
    return tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)


class Trainer:
    def __init__(self):
        self.config = {}
        self.logdir, self.logfile = get_logdir(FLAGS=FLAGS)
        self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
        self.ckpt_name = 'model.ckpt'
        self.worker_dir = ''
        self.sub_file = os.path.join(self.logdir, 'submission.%d.csv')
        redirect_stdout(self.logfile)
        self.train_data_param = {
            'gen_type': 'train',
            'random_sample': True,
            # TODO
            'batch_size': FLAGS.batch_size * FLAGS.num_gpus,
            'squeeze_output': False,
            'val_ratio': FLAGS.val_ratio,
        }
        self.valid_data_param = {
            'gen_type': 'valid' if FLAGS.val else 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size * FLAGS.num_gpus,
            'squeeze_output': False,
            'val_ratio': FLAGS.val_ratio,
        }
        self.test_data_param = {
            'gen_type': 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size * FLAGS.num_gpus,
            'squeeze_output': False,
        }
        self.train_logdir = os.path.join(self.logdir, 'train', self.worker_dir)
        self.valid_logdir = os.path.join(self.logdir, 'valid', self.worker_dir)
        self.test_logdir = os.path.join(self.logdir, 'test', self.worker_dir)
        self.gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                    gpu_options={'allow_growth': True})

        self.model_param = {'l2_embed': FLAGS.l2_embed, 'num_shards': FLAGS.num_shards, 'input_norm': FLAGS.input_norm,
                            'init_sparse': FLAGS.init_sparse, 'init_fused': FLAGS.init_fused,
                            'loss_mode': FLAGS.loss_mode}
        if FLAGS.model != 'lr':
            self.model_param['embed_size'] = FLAGS.embed_size
        if FLAGS.model in ['fnn', 'ccpm', 'deepfm', 'ipnn', 'kpnn', 'pin']:
            self.model_param['nn_layers'] = [tuple(x) for x in json.loads(FLAGS.nn_layers)]
        if FLAGS.model in ['nfm', 'pin']:
            self.model_param['sub_nn_layers'] = [tuple(x) for x in json.loads(FLAGS.sub_nn_layers)]
        if FLAGS.model == 'pin':
            self.model_param['wide'] = FLAGS.wide
            self.model_param['prod'] = FLAGS.prod
        if FLAGS.model in {'kfm', 'kpnn'}:
            self.model_param['unit_kernel'] = FLAGS.unit_kernel
            self.model_param['fix_kernel'] = FLAGS.fix_kernel
            self.model_param['l2_kernel'] = FLAGS.l2_kernel
            self.model_param['kernel_type'] = FLAGS.kernel_type
        self.dump_config()

        tf.reset_default_graph()
        self.dataset = as_dataset(FLAGS.dataset)

    def build_graph(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope(tf.get_variable_scope()):
                self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                   initializer=tf.constant_initializer(1), trainable=False)
                self.learning_rate = tf.get_variable(name='learning_rate', dtype=tf.float32, shape=[],
                                                     initializer=tf.constant_initializer(
                                                         FLAGS.learning_rate),
                                                     trainable=False)
                self.opt = get_optimizer(FLAGS.optimizer, self.learning_rate)
                self.model = as_model(FLAGS.model, input_dim=self.dataset.num_features,
                                      num_fields=self.dataset.num_fields,
                                      **self.model_param)
                tf.get_variable_scope().reuse_variables()

            self.train_op = self.opt.minimize(self.model.loss, global_step=self.global_step)
            self.saver = tf.train.Saver()

    def build_graph_multi_gpu(self):
        self.tower_grads = []
        self.models = []

        with tf.device('/gpu:0'):
            num_gpus = FLAGS.num_gpus
            with tf.variable_scope(tf.get_variable_scope()):
                self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                    initializer=tf.constant_initializer(1), trainable=False)
                self.learning_rate = tf.get_variable(name='learning_rate', dtype=tf.float32, shape=[],
                                                    initializer=tf.constant_initializer(FLAGS.learning_rate),
                                                    trainable=False)
                self.opt = get_optimizer(FLAGS.optimizer, self.learning_rate)
                for i in xrange(num_gpus):
                    with tf.device('/gpu:%d' % i):
                        print('Deploying gpu:%d ...' % i)
                        with tf.name_scope('tower_%d' % i):
                            model = as_model(FLAGS.model, input_dim=self.dataset.num_features,
                                             num_fields=self.dataset.num_fields,
                                             **self.model_param)
                            self.models.append(model)
                            tf.get_variable_scope().reuse_variables()
                            grads = self.opt.compute_gradients(model.loss)
                            self.tower_grads.append(grads)

            print('###################################')
            average_grads = []
            for grad_and_vars in zip(*self.tower_grads):
                grads = []
                # TODO test this
                if FLAGS.sparse_grad and isinstance(grad_and_vars[0][0], tf.IndexedSlices):
                    grad = sparse_grads_mean(grad_and_vars)
                    grad_shape = grad.dense_shape
                else:
                    for g, _ in grad_and_vars:
                        expanded_g = tf.expand_dims(g, 0)
                        grads.append(expanded_g)
                    grad = tf.concat(axis=0, values=grads)
                    grad = tf.reduce_mean(grad, 0)
                    grad_shape = grad.shape
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                print(type(grad), grad_shape, type(v), v.shape)
                average_grads.append(grad_and_var)
            print('###################################')
            # TODO test this
            # self.grad_op = tf.group([(x[0].op, x[1].op) for x in average_grads])
            self.update_op = self.opt.apply_gradients(average_grads, global_step=self.global_step)
            self.train_op = self.opt.apply_gradients(average_grads, global_step=self.global_step)
            self.saver = tf.train.Saver()

    def sess_op(self):
        return tf.Session(config=self.gpu_config)

    def train(self):
        train_size = int(self.dataset.train_size * (1 - FLAGS.val_ratio))
        self.num_steps = int(np.ceil(train_size / FLAGS.batch_size / FLAGS.num_gpus))
        self.eval_steps = int(np.ceil(self.num_steps / FLAGS.eval_level)) if FLAGS.eval_level else 0

        with self.sess_op() as self.sess:
            print('Train size = %d, Batch size = %d, GPUs = %d' %
                  (self.dataset.train_size, FLAGS.batch_size, FLAGS.num_gpus))
            print('%d rounds in total, One round = %d steps, One evaluation = %d steps' %
                  (FLAGS.num_rounds, self.num_steps, self.eval_steps))

            self.train_gen = self.dataset.batch_generator(self.train_data_param)
            self.valid_gen = self.dataset.batch_generator(self.valid_data_param)
            self.test_gen = self.dataset.batch_generator(self.test_data_param)

            self.train_writer = tf.summary.FileWriter(logdir=self.train_logdir, graph=self.sess.graph, flush_secs=30)
            self.test_writer = tf.summary.FileWriter(logdir=self.test_logdir, graph=self.sess.graph, flush_secs=30)
            self.valid_writer = tf.summary.FileWriter(logdir=self.valid_logdir, graph=self.sess.graph, flush_secs=30)

            if not FLAGS.restore:
                self.sess.run(tf.global_variables_initializer())
            else:
                checkpoint_state = tf.train.get_checkpoint_state(self.ckpt_dir)
                if checkpoint_state and checkpoint_state.model_checkpoint_path:
                    self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
                    print('Restore model from:', checkpoint_state.model_checkpoint_path)
                    print('Run initial evaluation...')
                    self.evaluate(self.test_gen, self.test_writer)
                else:
                    print('Restore failed')

            self.begin_step = self.global_step.eval(self.sess)
            self.step = self.begin_step
            self.start_time = time.time()

            prev_loss = 100000

            for r in range(1, FLAGS.num_rounds + 1):
                print('Round: %d' % r)
                for batch_xs, batch_ys in self.train_gen:
                    self.step, _loss_, _log_loss_, _l2_loss_ = self.train_batch(batch_xs, batch_ys)

                    if self.step % FLAGS.log_frequency == 0:
                        elapsed_time = self.get_elapsed()
                        print('Done step %d, Elapsed: %.2fs, Train-Loss: %.4f, Log-Loss: %.4f, L2-Loss: %g'
                              % (self.step, elapsed_time, _loss_, _log_loss_, _l2_loss_))
                        summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=_loss_),
                                                    tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                                    tf.Summary.Value(tag='l2_loss', simple_value=_l2_loss_)])
                        self.train_writer.add_summary(summary, global_step=self.step)

                    if FLAGS.eval_level and self.step % self.num_steps % self.eval_steps == 0:
                        elapsed_time = self.get_elapsed()
                        eta = FLAGS.num_rounds * self.num_steps / (self.step - self.begin_step) * elapsed_time
                        eval_times = self.step % self.num_steps // self.eval_steps or FLAGS.eval_level
                        print('Round: %d, Eval: %d / %d, AvgTime: %3.2fms, Elapsed: %.2fs, ETA: %s' %
                              (r, eval_times, FLAGS.eval_level, float(elapsed_time * 1000 / self.step),
                               elapsed_time, self.get_timedelta(eta=eta)))
                        if FLAGS.val_ratio > 0:
                            _val_loss_, _ = self.evaluate(self.valid_gen, self.valid_writer)
                        self.learning_rate.assign(self.learning_rate * FLAGS.decay)

                self.saver.save(self.sess, os.path.join(self.logdir, 'checkpoints', 'model.ckpt'), self.step)
                print('Round %d finished, Elapsed: %s' % (r, self.get_timedelta()))
                self.evaluate(self.test_gen, submission=r)
                if FLAGS.val_ratio > 0:
                    if _val_loss_ > prev_loss:
                        print('Early stop at round %d' % r)
                        return
                    else:
                        prev_loss = _val_loss_

    def get_elapsed(self):
        return time.time() - self.start_time

    def get_timedelta(self, eta=None):
        eta = eta or (time.time() - self.start_time)
        return str(timedelta(seconds=eta))

    def dump_config(self):
        for k, v in getattr(FLAGS, '__flags').iteritems():
            self.config[k] = getattr(FLAGS, k)
        for k, v in __init__.config.iteritems():
            if k != 'default':
                self.config[k] = v
        self.config['train_data_param'] = self.train_data_param
        self.config['valid_data_param'] = self.valid_data_param
        self.config['test_data_param'] = self.test_data_param
        self.config['logdir'] = self.logdir
        config_json = json.dumps(self.config, indent=4, sort_keys=True, separators=(',', ':'))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(config_json)
        path_json = os.path.join(self.logdir, 'config.json')
        cnt = 1
        while os.path.exists(path_json):
            path_json = os.path.join(self.logdir, 'config%d.json' % cnt)
            cnt += 1
        print('Config json file:', path_json)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        open(path_json, 'w').write(config_json)

    def train_batch(self, batch_xs, batch_ys):
        if FLAGS.num_gpus == 1:
            fetches = [self.train_op, self.global_step]
            train_feed = {}
            fetches += [self.model.loss, self.model.log_loss, self.model.l2_loss]
            train_feed[self.model.inputs] = batch_xs
            train_feed[self.model.labels] = batch_ys
            if self.model.training is not None:
                train_feed[self.model.training] = True

            _, step, _loss_, _log_loss_, _l2_loss_ = self.sess.run(fetches=fetches, feed_dict=train_feed)
        else:
            fetches = []
            train_feed = {}
            if len(batch_ys) < FLAGS.num_gpus:
                continue
            _split = range(0, len(batch_ys), int(len(batch_ys) / FLAGS.num_gpus))
            batch_xs = np.split(batch_xs, _split)
            batch_ys = np.split(batch_ys, _split)
            for i, model in enumerate(self.models):
                _xs, _ys = batch_xs[i], batch_ys[i]
                fetches += [model.loss, model.log_loss, model.l2_loss]
                train_feed[model.inputs] = _xs
                train_feed[model.labels] = _ys
                if model.training is not None:
                    train_feed[model.training] = True

            ret = self.sess.run(fetches=[self.train_op, self.global_step] + fetches,
                                feed_dict=train_feed, )
            step = ret[1]
            _loss_ = sum([ret[i] for i in range(2, len(ret), 3)]) / FLAGS.num_gpus
            _log_loss_ = sum([ret[i] for i in range(3, len(ret), 3)]) / FLAGS.num_gpus
            _l2_loss_ = sum([ret[i] for i in range(4, len(ret), 3)]) / FLAGS.num_gpus
        return step, _loss_, _log_loss_, _l2_loss_

    def evaluate_batch(self, batch_xs, batch_ys):
        if FLAGS.num_gpus == 1:
            feed_dict = {self.model.inputs: batch_xs, self.model.labels: batch_ys}
            if self.model.training is not None:
                feed_dict[self.model.training] = False
            _preds_ = self.sess.run(fetches=self.model.preds, feed_dict=feed_dict)
            batch_preds = [_preds_.flatten()]
        else:
            fetches = []
            feed_dict = {}
            if len(batch_ys) < FLAGS.num_gpus:
                return [], []
            _split = range(0, len(batch_ys), int(len(batch_ys) / FLAGS.num_gpus))
            batch_xs = np.split(batch_xs, _split)
            batch_ys = np.split(batch_ys, _split)
            for i, model in enumerate(self.models):
                xs, ys = batch_xs[i], batch_ys[i]
                fetches.append(model.preds)
                feed_dict[model.inputs] = xs
                feed_dict[model.labels] = ys
                labels.append(ys.flatten())
                if model.training is not None:
                    feed_dict[model.training] = False
            _preds_ = self.sess.run(fetches=fetches, feed_dict=feed_dict)
            batch_preds = [x.flatten() for x in _preds_]
        return batch_preds

    def evaluate(self, gen, writer=None, eps=1e-6, submission=0):
        labels = []
        preds = []
        start_time = time.time()
        for batch_xs, batch_ys in gen:
            if FLAGS.num_gpus == 1 or len(batch_ys) >= FLAGS.num_gpus:
                labels.append(batch_ys)
                preds.extend(self.evaluate_batch(batch_xs, batch_ys))
        labels = np.hstack(labels)
        preds = np.hstack(preds)
        _min_ = len(np.where(preds < eps)[0])
        _max_ = len(np.where(preds > 1 - eps)[0])
        print('%d samples are evaluated' % len(labels))
        print('EPS: %g, %d (%.2f) < eps, %d (%.2f) > 1-eps, %d (%.2f) are truncated' %
              (eps, _min_, _min_ / len(preds), _max_, _max_ / len(preds), _min_ + _max_, (_min_ + _max_) / len(preds)))
        preds[preds < eps] = eps
        preds[preds > 1 - eps] = 1 - eps
        if not submission:
            _log_loss_ = log_loss(y_true=labels, y_pred=preds)
            _auc_ = roc_auc_score(y_true=labels, y_score=preds)
            print('%s-Loss: %2.4f, AUC: %2.4f, Elapsed: %s' %
                  (gen.gen_type.capitalize(), _log_loss_, _auc_, str(timedelta(seconds=(time.time() - start_time)))))
            if writer:
                summary = tf.Summary(value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                            tf.Summary.Value(tag='auc', simple_value=_auc_)])
                writer.add_summary(summary, global_step=self.step)
            return _log_loss_, _auc_
        else:
            with open(self.sub_file % submission, 'w') as f:
                f.write('Id,Predicted\n')
                for i, p in enumerate(preds):
                    f.write('{0},{1}\n'.format(i + 60000000, p))
            print('Submission file: %s' % (self.sub_file % submission))


def main(_):
    trainer = Trainer()
    if FLAGS.num_gpus == 0:
        trainer.build_graph()
    else:
        trainer.build_graph_multi_gpu()
    trainer.train()


if __name__ == '__main__':
    tf.app.run()
