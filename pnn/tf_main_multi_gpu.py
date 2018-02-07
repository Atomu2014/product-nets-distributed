from __future__ import division
from __future__ import print_function

import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

import __init__

sys.path.append(__init__.config['data_path'])
from datasets import as_dataset
from print_hook import PrintHook
from tf_models_share_vars import as_model

default_values = __init__.default_values_nmz

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_gpus', 2, '# gpus')
tf.app.flags.DEFINE_string('logdir', '../log/avazu/pin/2018-02-07-05-56-44', 'Directory for storing mnist data')
tf.app.flags.DEFINE_bool('restore', False, 'Restore from logdir')
tf.app.flags.DEFINE_bool('val', False, 'If True, use validation set, else use test set')
tf.app.flags.DEFINE_integer('batch_size', 1000, 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 10000, 'Testing batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

tf.app.flags.DEFINE_string('dataset', 'avazu',
                           'Dataset = ipinyou, avazu, criteo, criteo_9d, criteo_16d"')
tf.app.flags.DEFINE_string('model', 'pin',
                           'Model type = lr, fm, ffm, kfm, nfm, fnn, ccpm, deepfm, ipnn, kpnn, pin')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer')
tf.app.flags.DEFINE_float('l2_scale', 1e-4, 'L2 regularization')
tf.app.flags.DEFINE_integer('embed_size', 20, 'Embedding size')
# e.g. [["conv", [5, 10]], ["act", "relu"], ["flat", [1, 2]], ["full", 100], ["act", "relu"], ["drop", 0.5], ["full", 1]]
tf.app.flags.DEFINE_string('nn_layers', default_values['nn_layers'], 'Network structure')
# e.g. [["full", 5], ["act", "relu"], ["drop", 0.9], ["full", 1]]
tf.app.flags.DEFINE_string('sub_nn_layers', default_values['sub_nn_layers'], 'Sub-network structure')

tf.app.flags.DEFINE_integer('max_step', 0, 'Number of max steps')
tf.app.flags.DEFINE_integer('max_data', 0, 'Number of instances')
tf.app.flags.DEFINE_integer('num_rounds', 2, 'Number of training rounds')
tf.app.flags.DEFINE_integer('eval_level', 5, 'Evaluating frequency level')
tf.app.flags.DEFINE_integer('log_frequency', 1000, 'Logging frequency')


def get_logdir(FLAGS):
    if FLAGS.restore:
        logdir = FLAGS.logdir
    else:
        logdir = '%s/%s/%s/%s' % (
            FLAGS.logdir, FLAGS.dataset, FLAGS.model, datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S'))
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


def get_optimizer(opt, lr, **kwargs):
    opt = opt.lower()
    eps = kwargs['epsilon'] if 'epsilon' in kwargs else 1e-8
    if opt == 'sgd' or opt == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif opt == 'adam':
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)
    elif opt == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=lr)


class Trainer:
    def __init__(self):
        self.config = {}
        self.logdir, self.logfile = get_logdir(FLAGS=FLAGS)
        redirect_stdout(self.logfile)
        self.train_data_param = {
            'gen_type': 'train',
            'random_sample': True,
            'batch_size': FLAGS.batch_size,
            'squeeze_output': False,
        }
        self.valid_data_param = {
            'gen_type': 'valid' if FLAGS.val else 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size,
            'squeeze_output': False,
        }
        self.test_data_param = {
            'gen_type': 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size,
            'squeeze_output': False,
        }
        self.model_param = {'l2_scale': FLAGS.l2_scale}
        if FLAGS.model != 'lr':
            self.model_param['embed_size'] = FLAGS.embed_size
        if FLAGS.model in ['fnn', 'ccpm', 'deepfm', 'ipnn', 'kpnn', 'pin']:
            self.model_param['nn_layers'] = [tuple(x) for x in json.loads(FLAGS.nn_layers)]
        if FLAGS.model in ['nfm', 'pin']:
            self.model_param['sub_nn_layers'] = [tuple(x) for x in json.loads(FLAGS.sub_nn_layers)]
        self.dump_config()

        tf.reset_default_graph()
        self.dataset = as_dataset(FLAGS.dataset)
        self.tower_grads = []
        self.models = []
        self.global_step = tf.Variable(1, name='global_step', trainable=False)
        self.opt = get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)
        average_grads = self.tower()
        self.train_op = self.opt.apply_gradients(average_grads, global_step=self.global_step)

        gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, )
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)

        self.train_gen = self.dataset.batch_generator(self.train_data_param)
        self.test_gen = self.dataset.batch_generator(self.test_data_param)
        self.valid_gen = self.dataset.batch_generator(self.valid_data_param)
        self.train_writer = tf.summary.FileWriter(logdir=os.path.join(self.logdir, 'train'), graph=self.sess.graph,
                                                  flush_secs=30)
        self.test_writer = tf.summary.FileWriter(logdir=os.path.join(self.logdir, 'test'), graph=self.sess.graph,
                                                 flush_secs=30)
        self.valid_writer = tf.summary.FileWriter(logdir=os.path.join(self.logdir, 'valid'), graph=self.sess.graph,
                                                  flush_secs=30)
        self.saver = tf.train.Saver()
        self.init_or_restore()

        self.begin_step = self.global_step.eval(self.sess)
        self.step = self.begin_step
        self.start_time = time.time()
        self.num_steps = int(np.ceil(self.dataset.train_size / FLAGS.batch_size / FLAGS.num_gpus))
        self.eval_steps = int(np.ceil(self.num_steps / FLAGS.eval_level)) if FLAGS.eval_level else 0

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
        print(config_json)
        path_json = os.path.join(self.logdir, 'config.json')
        cnt = 1
        while os.path.exists(path_json):
            path_json = os.path.join(self.logdir, 'config%d.json' % cnt)
            cnt += 1
        print('Config json file:', path_json)
        open(path_json, 'w').write(config_json)

    def tower(self):
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    print('Deploying gpu:%d ...' % i)
                    with tf.name_scope('tower_%d' % i):
                        model = as_model(FLAGS.model, input_dim=self.dataset.num_features, num_fields=self.dataset.num_fields,
                                         **self.model_param)
                        self.models.append(model)
                        tf.get_variable_scope().reuse_variables()
                        grads = self.opt.compute_gradients(model.loss)
                        self.tower_grads.append(grads)
        average_grads = []
        for grad_and_vars in zip(*self.tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def init_or_restore(self):
        if not FLAGS.restore:
            tf.global_variables_initializer().run(session=self.sess)
            # tf.local_variables_initializer().run(session=sess)
        else:
            checkpoint_state = tf.train.get_checkpoint_state(os.path.join(self.logdir, 'checkpoints'))
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
                print('Restore model from:', checkpoint_state.model_checkpoint_path)
                print('Run initial evaluation...')
                self.evaluate(self.test_gen)
            else:
                print('Restore failed')

    def evaluate(self, gen, writer=None):
        labels = []
        preds = []
        start_time = time.time()
        _iter = iter(gen)
        fetches = []
        feed_dict = {}
        for model in self.models:
            fetches.append(model.preds)
            try:
                xs, ys = _iter.next()
                feed_dict[model.inputs] = xs
                feed_dict[model.labels] = ys
                labels.append(ys)
                if model.training is not None:
                    feed_dict[model.training] = False
            except StopIteration:
                pass
        _preds_ = self.sess.run(fetches=fetches, feed_dict=feed_dict)
        if type(_preds_) is list:
            preds.extend(_preds_)
        else:
            preds.append(_preds_)
        labels = np.vstack(labels)
        preds = np.vstack(preds)
        eps = 1e-6
        preds[preds < eps] = eps
        preds[preds > 1 - eps] = 1 - eps
        _log_loss_ = log_loss(y_true=labels, y_pred=preds)
        _auc_ = roc_auc_score(y_true=labels, y_score=preds)
        print('%s-Loss: %2.4f, AUC: %2.4f, Elapsed: %s' %
              (gen.gen_type.capitalize(), _log_loss_, _auc_, str(timedelta(seconds=(time.time() - start_time)))))
        if writer:
            summary = tf.Summary(value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                        tf.Summary.Value(tag='auc', simple_value=_auc_)])
            writer.add_summary(summary, global_step=self.step)
        return _log_loss_, _auc_

    def train(self):
        print('%d rounds in total, One round = %d steps, One evaluation = %d steps' %
              (FLAGS.num_rounds, self.num_steps, self.eval_steps))
        for r in range(1, FLAGS.num_rounds + 1):
            print('Round: %d' % r)
            train_iter = iter(self.train_gen)
            while True:
                if (FLAGS.max_data and FLAGS.max_data <= self.step * FLAGS.batch_size) or \
                        (FLAGS.max_step and FLAGS.max_step <= self.step):
                    print('Finish %d steps, Finish %d instances, Elapsed: %.4f' %
                          (self.step, self.step * FLAGS.batch_size, time.time() - self.start_time))
                    exit(0)
                fetches = []
                train_feed = {}
                try:
                    for model in self.models:
                        batch_xs, batch_ys = train_iter.next()
                        fetches += [model.loss, model.log_loss, model.l2_loss]
                        train_feed[model.inputs] = batch_xs
                        train_feed[model.labels] = batch_ys
                        if model.training is not None:
                            train_feed[model.training] = True
                except StopIteration:
                    break
                ret = self.sess.run(fetches=[self.train_op, self.global_step] + fetches, feed_dict=train_feed)
                self.step = ret[1]
                _loss_ = sum([ret[i] for i in range(2, len(ret), 3)]) / FLAGS.num_gpus
                _log_loss_ = sum([ret[i] for i in range(3, len(ret), 3)]) / FLAGS.num_gpus
                _l2_loss_ = sum([ret[i] for i in range(4, len(ret), 3)]) / FLAGS.num_gpus

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
                          (r, eval_times, FLAGS.eval_level, float(elapsed_time * 1000 / self.step), elapsed_time,
                           self.get_timedelta(eta=eta)))
                    self.evaluate(self.valid_gen, self.valid_writer)

            self.saver.save(self.sess, os.path.join(self.logdir, 'checkpoints', 'model.ckpt'), self.step)
            if FLAGS.eval_level < 1:
                print('Round %d finished, Elapsed: %s' % (r, self.get_timedelta()))
        self.evaluate(self.test_gen, self.test_writer)
        print('Total Time: %s, Logdir: %s' % (self.get_timedelta(), self.logdir))


def main(_):
    print(FLAGS.__flags)
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    tf.app.run()
