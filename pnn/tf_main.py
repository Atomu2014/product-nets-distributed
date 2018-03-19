from __future__ import division
from __future__ import print_function

import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from tf_models import as_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ps_hosts', '10.58.14.149:12345', 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_hosts', '10.58.14.147:12346', #,10.58.14.150:12347',
                           'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_num_gpus', '4', 'Comma-separated list of integers')
tf.app.flags.DEFINE_string('job_name', '', 'One of ps, worker')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_integer('num_ps', 1, 'Number of ps')
tf.app.flags.DEFINE_integer('num_workers', 1, 'Number of workers')
tf.app.flags.DEFINE_bool('distributed', True, 'Distributed training using parameter servers')
# tf.app.flags.DEFINE_bool('sync', False, 'Synchronized training')
tf.app.flags.DEFINE_integer('lazy_update', 1, 'Number of local steps by which variable update is delayed')

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
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
tf.app.flags.DEFINE_string('loss_mode', 'mean', 'Loss = mean, sum')

tf.app.flags.DEFINE_integer('batch_size', 1024, 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 10240, 'Testing batch size')
tf.app.flags.DEFINE_string('dataset', 'avazu', 'Dataset = ipinyou, avazu, criteo, criteo_9d, criteo_16d"')
tf.app.flags.DEFINE_string('model', 'lr', 'Model type = lr, fm, ffm, kfm, nfm, fnn, ccpm, deepfm, ipnn, kpnn, pin')

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

tf.app.flags.DEFINE_integer('num_rounds', 1, 'Number of training rounds')
tf.app.flags.DEFINE_integer('eval_level', 0, 'Evaluating frequency level')
tf.app.flags.DEFINE_float('decay', 1., 'Learning rate decay')
tf.app.flags.DEFINE_integer('log_frequency', 1000, 'Logging frequency')


def get_logdir(FLAGS):
    # TODO logdir
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


def create_done_queue(i):
    with tf.device('/job:ps/task:%d' % (i)):
        return tf.FIFOQueue(FLAGS.num_workers, tf.int32, shared_name='done_queue' + str(i))


def create_done_queues():
    return [create_done_queue(i) for i in range(FLAGS.num_ps)]


def create_finish_queue(i):
    with tf.device('/job:worker/task:%d' % (i)):
        return tf.FIFOQueue(FLAGS.num_workers - 1, tf.int32, shared_name='done_queue' + str(i))


def create_finish_queues():
    return [create_finish_queue(0)]


class Trainer:
    def __init__(self):
        self.config = {}
        self.logdir, self.logfile = get_logdir(FLAGS=FLAGS)
        self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
        self.ckpt_name = 'model.ckpt'
        self.worker_dir = 'worker_%d' % FLAGS.task_index if FLAGS.distributed else ''
        self.sub_file = os.path.join(self.logdir, 'submission.%d.csv')
        redirect_stdout(self.logfile)

        if not FLAGS.distributed:
            self.num_gpus = FLAGS.num_gpus
            self.total_num_gpus = self.num_gpus
        else:
            self.worker_num_gpus = [int(x) for x in FLAGS.worker_num_gpus.split(',')]            
            self.num_gpus = self.worker_num_gpus[FLAGS.task_index]
            self.total_num_gpus = sum(self.worker_num_gpus)

        self.train_data_param = {
            'gen_type': 'train',
            'random_sample': True,
            # TODO
            'batch_size': FLAGS.batch_size * self.num_gpus,
            'squeeze_output': False,
            'val_ratio': FLAGS.val_ratio,
        }
        if FLAGS.distributed and FLAGS.job_name == 'worker':
            self.train_data_param['num_workers'] = FLAGS.num_workers
            self.train_data_param['task_index'] = FLAGS.task_index
        self.valid_data_param = {
            'gen_type': 'valid' if FLAGS.val else 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size * self.num_gpus,
            'squeeze_output': False,
            'val_ratio': FLAGS.val_ratio,
        }
        self.test_data_param = {
            'gen_type': 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size * self.num_gpus,
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

    def create_cluter(self):
        self.ps_hosts = FLAGS.ps_hosts.split(',')
        self.worker_hosts = FLAGS.worker_hosts.split(',')
        self.cluster = tf.train.ClusterSpec({'ps': self.ps_hosts, 'worker': self.worker_hosts})
        self.server = tf.train.Server(self.cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,
                                    config=self.gpu_config)

    def start_ps(self):
        print(self.server.target)
        sess = tf.Session(self.server.target)
        queue = create_done_queue(FLAGS.task_index)
        print('PS %d: waiting %d workers to finish' % (FLAGS.task_index, FLAGS.num_workers))
        for i in range(FLAGS.num_workers):
            sess.run(queue.dequeue())
            print('PS %d received done %d' % (FLAGS.task_index, i))
        print('PS %d: quitting' % (FLAGS.task_index))

    def device_op(self, gpu_index, local=False):
        if not FLAGS.distributed:
            return '/gpu:%d' % gpu_index
        else:
            worker_device = '/job:worker/task:%d/gpu:%d' % (FLAGS.task_index, gpu_index)
            if local:
                return worker_device
            else:
                return tf.train.replica_device_setter(worker_device=worker_device, cluster=self.cluster)

    def build_graph(self):
        tf.reset_default_graph()
        self.dataset = as_dataset(FLAGS.dataset)

        with tf.device(self.device_op(0)):
            with tf.variable_scope(tf.get_variable_scope()):
                self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                   initializer=tf.constant_initializer(0), trainable=False)
                self.learning_rate = tf.get_variable(name='learning_rate', dtype=tf.float32, shape=[],
                                                     initializer=tf.constant_initializer(
                                                         FLAGS.learning_rate),
                                                     trainable=False)
                self.opt = get_optimizer(FLAGS.optimizer, self.learning_rate)
                self.model = as_model(FLAGS.model, input_dim=self.dataset.num_features,
                                      num_fields=self.dataset.num_fields,
                                      **self.model_param)
                tf.get_variable_scope().reuse_variables()
                self.grads = self.opt.compute_gradients(self.model.loss)
        
        with tf.device(self.device_op(0, local=True)):    
            if FLAGS.lazy_update > 1:
                local_grads = []
                accumulate_op = []
                reset_op = []
                self.local_grads = []
                for grad, v in self.grads:
                    zero_grad = tf.zeros_like(v)
                    local_grad = tf.Variable(zero_grad, dtype=tf.float32, trainable=False,
                                            name=v.name.split(':')[0] + '_local_grad',
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
                    self.local_grads.append(local_grad)
                    reset_grad = local_grad.assign(zero_grad)
                    if FLAGS.sparse_grad and isinstance(grad, tf.IndexedSlices):
                        accumulate_grad = local_grad.scatter_sub(-grad)
                    else:
                        accumulate_grad = local_grad.assign_add(grad)
                    local_grads.append((local_grad, v))
                    accumulate_op.append(accumulate_grad)
                    reset_op.append(reset_grad)
            if FLAGS.lazy_update > 1:
                self.update_op = self.opt.apply_gradients(local_grads, global_step=self.global_step)
                self.accumulate_op = tf.group(*accumulate_op)
                self.reset_op = tf.group(*reset_op)
            else:
                self.train_op = self.opt.minimize(self.model.loss, global_step=self.global_step)
            self.saver = tf.train.Saver()

    def build_graph_multi_gpu(self):
        tf.reset_default_graph()
        self.dataset = as_dataset(FLAGS.dataset)
        self.tower_grads = []
        self.models = []

        with tf.device(self.device_op(0)):
            with tf.variable_scope(tf.get_variable_scope()):
                self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                    initializer=tf.constant_initializer(0), trainable=False)
                self.learning_rate = tf.get_variable(name='learning_rate', dtype=tf.float32, shape=[],
                                                    initializer=tf.constant_initializer(FLAGS.learning_rate),
                                                    trainable=False)
                self.opt = get_optimizer(FLAGS.optimizer, self.learning_rate)
                for i in xrange(self.num_gpus):
                    with tf.device(self.device_op(i)):
                        print('Deploying gpu:%d ...' % i)
                        with tf.name_scope('tower_%d' % i):
                            model = as_model(FLAGS.model, input_dim=self.dataset.num_features,
                                             num_fields=self.dataset.num_fields,
                                             **self.model_param)
                            self.models.append(model)
                            tf.get_variable_scope().reuse_variables()
                            grads = self.opt.compute_gradients(model.loss)
                            self.tower_grads.append(grads)

        with tf.device(self.device_op(0, local=True)):
            print('###################################')
            average_grads = []
            if FLAGS.lazy_update > 1:
                local_grads = []
                accumulate_op = []
                reset_op = []
                self.local_grads = []
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

                if FLAGS.lazy_update > 1:
                    zero_grad = tf.zeros_like(v)
                    local_grad = tf.Variable(zero_grad, dtype=tf.float32, trainable=False, 
                                            name=v.name.split(':')[0] + '_local_grad',
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
                    self.local_grads.append(local_grad)                    
                    reset_grad = local_grad.assign(zero_grad)
                    if FLAGS.sparse_grad and isinstance(grad, tf.IndexedSlices):
                        accumulate_grad = local_grad.scatter_sub(-grad)
                    else:
                        accumulate_grad = local_grad.assign_add(grad)
                    local_grads.append((local_grad, v))
                    accumulate_op.append(accumulate_grad)
                    reset_op.append(reset_grad)
            print('###################################')
            # TODO test this
            # self.grad_op = tf.group([(x[0].op, x[1].op) for x in average_grads])
            if FLAGS.lazy_update > 1:
                self.update_op = self.opt.apply_gradients(local_grads, global_step=self.global_step)
                # self.grad_op = tf.group(average_grads)
                # TODO tf.ver < 1.5 need *inputs 
                self.accumulate_op = tf.group(*accumulate_op)
                self.reset_op = tf.group(*reset_op)
            else:
                self.train_op = self.opt.apply_gradients(average_grads, global_step=self.global_step)
            self.saver = tf.train.Saver()

    def start_worker(self):
        with tf.device(self.device_op(0)):
            self.enq_ops = []
            for q in create_done_queues():
                qop = q.enqueue(1)
                self.enq_ops.append(qop)
            # if not FLAGS.sync:
            if FLAGS.task_index != 0:
                for q in create_finish_queues():
                    qop = q.enqueue(1)
                    self.enq_ops.insert(0, qop)
            else:
                chief_queue = create_finish_queue(FLAGS.task_index)
                self.deq_op = []
                for i in range(FLAGS.num_workers - 1):
                    self.deq_op.append(chief_queue.dequeue())

    def sess_op(self):
        if not FLAGS.distributed:
            return tf.Session(config=self.gpu_config)
        else:
            return tf.Session(self.server.target)
            # return tf.train.MonitoredTrainingSession(master=self.server.target, 
            #                                         is_chief=(FLAGS.task_index == 0),
            #                                         # TODO
            #                                         hooks=None,
            #                                         chief_only_hooks=None)

    def get_nake_sess(self):
        sess = self.sess
        while type(sess).__name__ != 'Session':
            sess = sess._sess
        return sess

    def train_batch(self, batch_xs, batch_ys):
        if self.num_gpus == 1:
            train_op = self.train_op if FLAGS.lazy_update <= 1 else self.accumulate_op
            fetches = [train_op]
            train_feed = {}
            fetches += [self.model.loss, self.model.log_loss, self.model.l2_loss]
            train_feed[self.model.inputs] = batch_xs
            train_feed[self.model.labels] = batch_ys
            if self.model.training is not None:
                train_feed[self.model.training] = True

            _, _loss_, _log_loss_, _l2_loss_ = self.sess.run(fetches=fetches, feed_dict=train_feed)
        else:
            fetches = []
            train_feed = {}
            _batch = int(len(batch_ys) / self.num_gpus)
            _split = [_batch * i for i in range(1, self.num_gpus)]
            batch_xs = np.split(batch_xs, _split)
            batch_ys = np.split(batch_ys, _split)
            for i, model in enumerate(self.models):
                _xs, _ys = batch_xs[i], batch_ys[i]
                fetches += [model.loss, model.log_loss, model.l2_loss]
                train_feed[model.inputs] = _xs
                train_feed[model.labels] = _ys
                if model.training is not None:
                    train_feed[model.training] = True

            train_op = self.train_op if FLAGS.lazy_update <= 1 else self.accumulate_op
            ret = self.sess.run(fetches=[train_op] + fetches,
                                feed_dict=train_feed, )
            _loss_ = np.mean([ret[i] for i in range(1, len(ret), 3)])
            _log_loss_ = np.mean([ret[i] for i in range(2, len(ret), 3)])
            _l2_loss_ = np.mean([ret[i] for i in range(3, len(ret), 3)])
        return _loss_, _log_loss_, _l2_loss_

    def train(self):
        with self.sess_op() as self.sess:
        # with tf.Session(self.server.target) as self.sess:
            if not FLAGS.distributed or FLAGS.task_index == 0:
                self.sess.run([tf.global_variables_initializer(),
                                tf.local_variables_initializer()])
            elif FLAGS.lazy_update > 1:
                self.sess.run(tf.variables_initializer(self.local_grads))
            self.train_gen = self.dataset.batch_generator(self.train_data_param)
            self.valid_gen = self.dataset.batch_generator(self.valid_data_param)
            self.test_gen = self.dataset.batch_generator(self.test_data_param)

            self.train_writer = tf.summary.FileWriter(logdir=self.train_logdir, graph=self.sess.graph, flush_secs=30)
            self.test_writer = tf.summary.FileWriter(logdir=self.test_logdir, graph=self.sess.graph, flush_secs=30)
            self.valid_writer = tf.summary.FileWriter(logdir=self.valid_logdir, graph=self.sess.graph, flush_secs=30)

            # train_size = int(self.dataset.train_size * (1 - FLAGS.val_ratio))
            # self.num_steps = int(np.ceil(train_size / FLAGS.batch_size / FLAGS.num_gpus))
            print('checking dataset...')
            self.train_size = 0
            self.num_steps = 0
            for _x, _y in self.train_gen:
                self.train_size += len(_y)
                self.num_steps += 1
            self.eval_steps = int(np.ceil(self.num_steps / FLAGS.eval_level)) if FLAGS.eval_level else 0
               
            print('Train size = %d, Batch size = %d, GPUs = %d / %d' %
                  (self.train_size, FLAGS.batch_size, self.num_gpus, self.total_num_gpus))
            print('%d rounds in total, One round = %d steps, One evaluation = %d steps' %
                  (FLAGS.num_rounds, self.num_steps, self.eval_steps))

            if not FLAGS.distributed:
                if not FLAGS.restore:
                    self.sess.run([tf.global_variables_initializer(),
                                    tf.local_variables_initializer()])
                else:
                    # TODO check restore
                    checkpoint_state = tf.train.get_checkpoint_state(self.ckpt_dir)
                    if checkpoint_state and checkpoint_state.model_checkpoint_path:
                        self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
                        print('Restore model from:', checkpoint_state.model_checkpoint_path)
                        print('Run initial evaluation...')
                        self.evaluate(self.test_gen, self.test_writer)
                    else:
                        print('Restore failed')
            else:
                # TODO check restore                
                if FLAGS.restore:
                    print('Restore model from:', self.ckpt_dir)
                    print('Run initial evaluation...')
                    self.evaluate(self.test_gen, self.test_writer)
                else:
                    if FLAGS.task_index == 0:
                        self.sess.run([tf.global_variables_initializer(),
                                        tf.local_variables_initializer()])
                    elif FLAGS.lazy_update > 1:
                        self.sess.run(tf.variables_initializer(self.local_grads))

            # TODO: initial evaluation
            self.begin_step = self.global_step.eval(self.sess)
            self.step = self.begin_step
            self.start_time = time.time()
            self.local_step = 0

            for r in range(1, FLAGS.num_rounds + 1):
                print('Round: %d' % r)
                for batch_xs, batch_ys in self.train_gen:
                    # TODO: check
                    if len(batch_ys) < self.num_gpus:
                        break

                    _loss_, _log_loss_, _l2_loss_ = self.train_batch(batch_xs, batch_ys)
                    self.step = self.sess.run(self.global_step)
                    self.local_step += 1

                    if FLAGS.lazy_update > 1:
                        if self.local_step % FLAGS.lazy_update == 0:
                            self.sess.run(self.update_op)
                            self.sess.run(self.reset_op)
                            # elapsed_time = self.get_elapsed()
                            # print('Local step %d, Elapsed: %.2fs, Lazy update' % (self.local_step, elapsed_time))

                        if self.local_step % FLAGS.log_frequency == 0:
                            self.step = self.sess.run(self.global_step)
                            elapsed_time = self.get_elapsed()
                            print('Local step: %d, Global step %d, Elapsed: %.2fs, Train-Loss: %.6f, Log-Loss: %.6f, L2-Loss: %g'
                                % (self.local_step, self.step, elapsed_time, _loss_, _log_loss_, _l2_loss_))
                    else:
                        if self.step % FLAGS.log_frequency == 0:
                            elapsed_time = self.get_elapsed()
                            print('Done step %d, Elapsed: %.2fs, Train-Loss: %.6f, Log-Loss: %.6f, L2-Loss: %g'
                                % (self.step, elapsed_time, _loss_, _log_loss_, _l2_loss_))
                            summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=_loss_),
                                                        tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                                        tf.Summary.Value(tag='l2_loss', simple_value=_l2_loss_)])
                            self.train_writer.add_summary(summary, global_step=self.step)

                        if FLAGS.eval_level and self.step % self.num_steps % self.eval_steps == 0:
                            elapsed_time = self.get_elapsed()
                            eta = FLAGS.num_rounds * self.num_steps / (self.step - self.begin_step) * elapsed_time
                            eval_times = self.step % self.num_steps // self.eval_steps or FLAGS.eval_level
                            print('Round: %d, Eval: %d / %d, AvgTime: %.2fms, Elapsed: %.2fs, ETA: %s' %
                                (r, eval_times, FLAGS.eval_level, float(elapsed_time * 1000 / self.step),
                                elapsed_time, self.get_timedelta(eta=eta)))
                            if not FLAGS.distributed:# or FLAGS.task_index == 0:
                                self.evaluate(self.valid_gen, self.valid_writer)
                                # TODO implement decay
                                # self.learning_rate.assign(self.learning_rate * FLAGS.decay)

                print('Round %d finished, Elapsed: %s' % (r, self.get_timedelta()))
                if not FLAGS.distributed:# or FLAGS.task_index == 0:
                    self.saver.save(self.get_nake_sess(), os.path.join(self.logdir, 'checkpoints', 'model.ckpt'), self.step)
                    # TODO check evaluate if lazy_update > num_steps
                    if FLAGS.eval_level == 0 or (FLAGS.lazy_update > 1):
                        self.evaluate(self.valid_gen, self.valid_writer)                    
            if FLAGS.distributed:
                if FLAGS.task_index == 0:
                    self.evaluate(self.test_gen, self.test_writer)                                    
                self.stop_worker()

    def stop_worker(self):
        if FLAGS.task_index == 0:
            print('Chief worker: waiting %d workers to finish' % len(self.deq_op))
            for q in self.deq_op:
                self.sess.run(q)
                print('Chief worker received done')
            print('Chief worker: quitting')
        print('Total Time: %s, Logdir: %s' % (self.get_timedelta(), self.logdir))
        for op in self.enq_ops:
            self.sess.run(op)
        print('Worker %d can not exit, should be killed' % FLAGS.task_index)

    def evaluate_batch(self, batch_xs, batch_ys):
        if self.num_gpus == 1:
            feed_dict = {self.model.inputs: batch_xs, self.model.labels: batch_ys}
            if self.model.training is not None:
                feed_dict[self.model.training] = False
            _preds_ = self.sess.run(fetches=self.model.preds, feed_dict=feed_dict)
            batch_preds = [_preds_.flatten()]
        else:
            fetches = []
            feed_dict = {}
            _batch = int(len(batch_ys) / self.num_gpus)
            _split = [_batch * i for i in range(1, self.num_gpus)]
            batch_xs = np.split(batch_xs, _split)
            batch_ys = np.split(batch_ys, _split)
            for i, model in enumerate(self.models):
                xs, ys = batch_xs[i], batch_ys[i]
                fetches.append(model.preds)
                feed_dict[model.inputs] = xs
                feed_dict[model.labels] = ys
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
            if len(batch_ys) < self.num_gpus:
                break
            labels.append(batch_ys.flatten())
            preds.extend(self.evaluate_batch(batch_xs, batch_ys))
        labels = np.hstack(labels)
        preds = np.hstack(preds)
        _min_ = len(np.where(preds < eps)[0])
        _max_ = len(np.where(preds > 1 - eps)[0])
        print('%d samples are evaluated' % len(labels))
        if _min_ + _max_ > 0:
            print('EPS: %g, %d (%.2f) < eps, %d (%.2f) > 1-eps, %d (%.2f) are truncated' %
                (eps, _min_, _min_ / len(preds), _max_, _max_ / len(preds), _min_ + _max_, (_min_ + _max_) / len(preds)))
        preds[preds < eps] = eps
        preds[preds > 1 - eps] = 1 - eps
        if not submission:
            _log_loss_ = log_loss(y_true=labels, y_pred=preds)
            _auc_ = roc_auc_score(y_true=labels, y_score=preds)
            print('%s-Loss: %.6f, AUC: %.6f, Elapsed: %s' %
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
    if FLAGS.distributed:
        trainer.create_cluter()
        if FLAGS.job_name == 'ps':
            trainer.start_ps()
            return
    if trainer.num_gpus == 1:
        trainer.build_graph()
    else:
        trainer.build_graph_multi_gpu()
    if FLAGS.distributed:
        trainer.start_worker()
    trainer.train()


if __name__ == '__main__':
    tf.app.run()
