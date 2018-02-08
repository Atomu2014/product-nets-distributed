from __future__ import division
from __future__ import print_function

import json
import os
import sys
import time
from datetime import timedelta

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
tf.app.flags.DEFINE_string('ps_hosts', 'localhost:12545', 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_hosts', 'localhost:12546,localhost:12547',
                           'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_num_gpus', '2,2', 'Comma-separated list of integers')
tf.app.flags.DEFINE_string('job_name', '', 'One of ps, worker')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_integer('num_ps', 1, 'Number of ps')
tf.app.flags.DEFINE_integer('num_workers', 2, 'Number of workers')
tf.app.flags.DEFINE_bool('distributed', True, 'Distributed training using parameter servers')
tf.app.flags.DEFINE_bool('sync', False, 'Synchronized training')

tf.app.flags.DEFINE_integer('num_gpus', 2, '# gpus')
tf.app.flags.DEFINE_string('logdir', '../log/avazu/pin', 'Directory for storing mnist data')
tf.app.flags.DEFINE_bool('restore', False, 'Restore from logdir')
tf.app.flags.DEFINE_bool('val', False, 'If True, use validation set, else use test set')
tf.app.flags.DEFINE_integer('batch_size', 2000, 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 10000, 'Testing batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')

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
tf.app.flags.DEFINE_integer('log_frequency', 100, 'Logging frequency')


def get_logdir(FLAGS):
    # if FLAGS.restore:
    #     logdir = FLAGS.logdir
    # else:
    #     logdir = '%s/%s/%s/%s' % (
    #         FLAGS.logdir, FLAGS.dataset, FLAGS.model, datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S'))
    logdir = FLAGS.logdir
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


def create_done_queue(i):
    with tf.device('/job:ps/task:%d' % (i)):
        return tf.FIFOQueue(FLAGS.num_workers, tf.int32, shared_name='done_queue' + str(i))


def create_done_queues():
    return [create_done_queue(i) for i in range(FLAGS.num_ps)]


def is_chief_worker():
    return FLAGS.job_name == 'worker' and FLAGS.task_index == 0


def is_monitor():
    return not FLAGS.distributed or is_chief_worker()


class Trainer:
    def __init__(self):
        self.config = {}
        self.logdir, self.logfile = get_logdir(FLAGS=FLAGS)
        self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
        self.ckpt_name = 'model.ckpt'
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
        gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        gpu_config.gpu_options.allow_growth = True
        if FLAGS.distributed:
            self.ps_hosts = FLAGS.ps_hosts.split(',')
            self.worker_hosts = FLAGS.worker_hosts.split(',')
            self.worker_num_gpus = [int(x) for x in FLAGS.worker_num_gpus.split(',')]
            self.cluster = tf.train.ClusterSpec({'ps': self.ps_hosts, 'worker': self.worker_hosts})
            self.server = tf.train.Server(self.cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,
                                          config=gpu_config)
            if FLAGS.job_name == 'ps':
                print(self.server.target)
                # TODO check this
                self.server.join()
                sess = tf.Session(self.server.target)
                queue = create_done_queue(FLAGS.task_index)
                for i in range(FLAGS.num_workers):
                    sess.run(queue.dequeue())
                    print('PS %d received done %d' % (FLAGS.task_index, i))
                print('PS %d: quitting' % (FLAGS.task_index))
                return
            if FLAGS.job_name == 'worker':
                self.train_data_param['num_workers'] = FLAGS.num_workers
                self.train_data_param['task_index'] = FLAGS.task_index

        self.model_param = {'l2_scale': FLAGS.l2_scale}
        if FLAGS.model != 'lr':
            self.model_param['embed_size'] = FLAGS.embed_size
        if FLAGS.model in ['fnn', 'ccpm', 'deepfm', 'ipnn', 'kpnn', 'pin']:
            self.model_param['nn_layers'] = [tuple(x) for x in json.loads(FLAGS.nn_layers)]
        if FLAGS.model in ['nfm', 'pin']:
            self.model_param['sub_nn_layers'] = [tuple(x) for x in json.loads(FLAGS.sub_nn_layers)]
        if is_monitor():
            self.dump_config()

        tf.reset_default_graph()
        self.dataset = as_dataset(FLAGS.dataset)
        self.tower_grads = []
        self.models = []

        def device_op(gpu_index):
            if not FLAGS.distributed:
                return '/gpu:%d' % gpu_index
            else:
                return tf.train.replica_device_setter(worker_device='job:worker/task:%d/gpu:%d' % (FLAGS.task_index, i),
                                                      cluster=self.cluster)

        with tf.device('/gpu:0'):
            num_gpus = FLAGS.num_gpus if not FLAGS.distributed else self.worker_num_gpus[FLAGS.task_index]
            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(num_gpus):
                    with tf.device(device_op(i)):
                        print('Deploying gpu:%d ...' % i)
                        if i == 0:
                            self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                               initializer=tf.constant_initializer(1), trainable=False)
                            self.opt = get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)
                        with tf.name_scope('tower_%d' % i):
                            model = as_model(FLAGS.model, input_dim=self.dataset.num_features,
                                             num_fields=self.dataset.num_fields,
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
            self.train_op = self.opt.apply_gradients(average_grads, global_step=self.global_step)
            self.saver = tf.train.Saver()
            self.enq_ops = []
            for q in create_done_queues():
                qop = q.enqueue(1)
                self.enq_ops.append(qop)

        def sess_op():
            if not FLAGS.distributed:
                return tf.Session(config=gpu_config)
            else:
                return tf.train.MonitoredTrainingSession(master=self.server.target, is_chief=(FLAGS.task_index == 0),
                                                         checkpoint_dir=self.ckpt_dir,
                                                         chief_only_hooks=[tf.train.CheckpointSaverHook(
                                                             checkpoint_dir=self.ckpt_dir,
                                                             save_steps=self.num_steps,
                                                             saver=self.saver,
                                                             checkpoint_basename=self.ckpt_name)])

        if not FLAGS.distributed:
            num_gpus = FLAGS.num_gpus
            total_num_gpus = num_gpus
            self.num_steps = int(np.ceil(self.dataset.train_size / FLAGS.batch_size / num_gpus))
        else:
            num_gpus = self.worker_num_gpus[FLAGS.task_index]
            total_num_gpus = sum(self.worker_num_gpus)
            if FLAGS.sync:
                self.num_steps = int(
                    np.ceil(self.dataset.train_size / FLAGS.num_workers / FLAGS.batch_size / num_gpus))
            else:
                self.num_steps = int(np.ceil(self.dataset.train_size / FLAGS.batch_size / num_gpus))
        self.eval_steps = int(np.ceil(self.num_steps / FLAGS.eval_level)) if FLAGS.eval_level else 0
        print('Train size = %d, Batch size = %d, GPUs = %d/%d' %
              (self.dataset.train_size, FLAGS.batch_size, num_gpus, total_num_gpus))
        print('%d rounds in total, One round = %d steps, One evaluation = %d steps' %
              (FLAGS.num_rounds, self.num_steps, self.eval_steps))

        with sess_op() as self.sess:
            self.begin_step = self.global_step.eval(self.sess)
            self.step = self.begin_step
            self.start_time = time.time()

            self.train_gen = self.dataset.batch_generator(self.train_data_param)
            self.valid_gen = self.dataset.batch_generator(self.valid_data_param)
            self.test_gen = self.dataset.batch_generator(self.test_data_param)
            worker_dir = 'worker_%d' % FLAGS.task_index
            self.train_writer = tf.summary.FileWriter(logdir=os.path.join(self.logdir, 'train', worker_dir),
                                                      graph=self.sess.graph, flush_secs=30)
            self.test_writer = tf.summary.FileWriter(logdir=os.path.join(self.logdir, 'test', worker_dir),
                                                     graph=self.sess.graph, flush_secs=30)
            self.valid_writer = tf.summary.FileWriter(logdir=os.path.join(self.logdir, 'valid', worker_dir),
                                                      graph=self.sess.graph, flush_secs=30)
            if is_monitor():
                self.evaluate(self.test_gen, self.test_writer)
            for r in range(1, FLAGS.num_rounds + 1):
                print('Round: %d' % r)
                train_iter = iter(self.train_gen)
                while True:
                    if (FLAGS.max_data and FLAGS.max_data <= self.step * FLAGS.batch_size) or \
                            (FLAGS.max_step and FLAGS.max_step <= self.step):
                        print('Finish %d steps, Finish %d instances, Elapsed: %.4f' %
                              (self.step, self.step * FLAGS.batch_size, time.time() - self.start_time))
                        if FLAGS.distributed:
                            for op in self.enq_ops:
                                self.sess.run(op)
                        return
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
                        if not FLAGS.distributed or not FLAGS.sync or is_monitor():
                            elapsed_time = self.get_elapsed()
                            eta = FLAGS.num_rounds * self.num_steps / (self.step - self.begin_step) * elapsed_time
                            eval_times = self.step % self.num_steps // self.eval_steps or FLAGS.eval_level
                            print('Round: %d, Eval: %d / %d, AvgTime: %3.2fms, Elapsed: %.2fs, ETA: %s' %
                                  (r, eval_times, FLAGS.eval_level, float(elapsed_time * 1000 / self.step),
                                   elapsed_time, self.get_timedelta(eta=eta)))
                            self.evaluate(self.valid_gen, self.valid_writer)
                if not FLAGS.distributed:
                    self.saver.save(self.sess, os.path.join(self.logdir, 'checkpoints', 'model.ckpt'), self.step)
                print('Round %d finished, Elapsed: %s' % (r, self.get_timedelta()))
            if is_monitor():
                self.evaluate(self.test_gen, self.test_writer)
            print('Total Time: %s, Logdir: %s' % (self.get_timedelta(), self.logdir))
            if FLAGS.distributed:
                for op in self.enq_ops:
                    self.sess.run(op)

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


def main(_):
    Trainer()


if __name__ == '__main__':
    tf.app.run()
