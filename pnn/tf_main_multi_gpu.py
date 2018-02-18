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
tf.app.flags.DEFINE_string('ps_hosts', '172.16.2.245:50031', 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_hosts', '172.16.2.244:50010',  # ,172.16.2.246:50012',
                           'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_num_gpus', '2', 'Comma-separated list of integers')
tf.app.flags.DEFINE_string('job_name', '', 'One of ps, worker')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_integer('num_ps', 1, 'Number of ps')
tf.app.flags.DEFINE_integer('num_workers', 1, 'Number of workers')
tf.app.flags.DEFINE_bool('distributed', True, 'Distributed training using parameter servers')
tf.app.flags.DEFINE_bool('sync', False, 'Synchronized training')
tf.app.flags.DEFINE_integer('num_shards', 1, 'Number of variable partitions')
tf.app.flags.DEFINE_bool('sparse_grad', True, 'Apply sparse gradient')
tf.app.flags.DEFINE_string('caching_device', '', 'Caching device = "", "cpu", "gpu"')
tf.app.flags.DEFINE_integer('lazy_update', 1, 'Number of local steps where variable update is delayed')

tf.app.flags.DEFINE_integer('num_gpus', 2, '# gpus')
tf.app.flags.DEFINE_string('logdir', '../log/avazu/speed', 'Directory for storing mnist data')
tf.app.flags.DEFINE_bool('restore', False, 'Restore from logdir')
tf.app.flags.DEFINE_bool('val', False, 'If True, use validation set, else use test set')
tf.app.flags.DEFINE_integer('batch_size', 10000, 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 10000, 'Testing batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')

tf.app.flags.DEFINE_string('dataset', 'avazu',
                           'Dataset = ipinyou, avazu, criteo, criteo_9d, criteo_16d"')
tf.app.flags.DEFINE_string('model', 'lr',
                           'Model type = lr, fm, ffm, kfm, nfm, fnn, ccpm, deepfm, ipnn, kpnn, pin')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer')
tf.app.flags.DEFINE_float('l2_scale', 0, 'L2 regularization')
tf.app.flags.DEFINE_integer('embed_size', 2, 'Embedding size')
# e.g. [["conv", [5, 10]], ["act", "relu"], ["flat", [1, 2]], ["full", 100], ["act", "relu"], ["drop", 0.5], ["full", 1]]
tf.app.flags.DEFINE_string('nn_layers', default_values['nn_layers'], 'Network structure')
# e.g. [["full", 5], ["act", "relu"], ["drop", 0.9], ["full", 1]]
tf.app.flags.DEFINE_string('sub_nn_layers', default_values['sub_nn_layers'], 'Sub-network structure')

tf.app.flags.DEFINE_integer('max_step', 0, 'Number of max steps')
tf.app.flags.DEFINE_integer('max_data', 0, 'Number of instances')
tf.app.flags.DEFINE_integer('num_rounds', 1, 'Number of training rounds')
tf.app.flags.DEFINE_integer('eval_level', 0, 'Evaluating frequency level')
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


def create_finish_queue(i):
    with tf.device('/job:worker/task:%d' % (i)):
        return tf.FIFOQueue(FLAGS.num_workers - 1, tf.int32, shared_name='done_queue' + str(i))


def create_finish_queues():
    return [create_finish_queue(0)]


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
        self.worker_dir = 'worker_%d' % FLAGS.task_index if (FLAGS.distributed and not FLAGS.sync) else ''
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
        self.train_logdir = os.path.join(self.logdir, 'train', self.worker_dir)
        self.valid_logdir = os.path.join(self.logdir, 'valid', self.worker_dir)
        self.test_logdir = os.path.join(self.logdir, 'test', self.worker_dir)
        gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                    gpu_options={'allow_growth': True})
        if FLAGS.distributed:
            self.ps_hosts = FLAGS.ps_hosts.split(',')
            self.worker_hosts = FLAGS.worker_hosts.split(',')
            self.worker_num_gpus = [int(x) for x in FLAGS.worker_num_gpus.split(',')]
            self.cluster = tf.train.ClusterSpec({'ps': self.ps_hosts, 'worker': self.worker_hosts})
            self.server = tf.train.Server(self.cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,
                                          config=gpu_config)
            if FLAGS.job_name == 'ps':
                print(self.server.target)
                sess = tf.Session(self.server.target)
                queue = create_done_queue(FLAGS.task_index)
                print('PS %d: waiting %d workers to finish' % (FLAGS.task_index, FLAGS.num_workers))
                for i in range(FLAGS.num_workers):
                    sess.run(queue.dequeue())
                    print('PS %d received done %d' % (FLAGS.task_index, i))
                print('PS %d: quitting' % (FLAGS.task_index))
                return
            if FLAGS.job_name == 'worker':
                self.train_data_param['num_workers'] = FLAGS.num_workers
                self.train_data_param['task_index'] = FLAGS.task_index

        self.model_param = {'l2_scale': FLAGS.l2_scale, 'num_shards': FLAGS.num_shards}
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
            # partitioner = tf.fixed_size_partitioner(num_shards=50)
            # TODO check cache device and partitioner
            if FLAGS.caching_device == '':
                caching_device = None
            elif FLAGS.caching_device == 'cpu':
                caching_device = '/cpu:0'
            elif FLAGS.caching_device == 'gpu':
                caching_device = '/gpu:0'
            with tf.variable_scope(tf.get_variable_scope(), caching_device=caching_device):
                # , partitioner=partitioner):
                for i in xrange(num_gpus):
                    with tf.device(device_op(i)):
                        print('Deploying gpu:%d ...' % i)
                        if i == 0:
                            self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                               initializer=tf.constant_initializer(1), trainable=False)
                            # for j in range(FLAGS.num_workers):
                            #     tf.get_variable(name='local_step_%d' % j, dtype=tf.int32,
                            #                                  shape=[], initializer=tf.constant_initializer(1),
                            #                                  trainable=False)
                            # self.local_step = tf.get_variable(name='local_step_%d' % FLAGS.task_index, dtype=tf.int32,
                            #                                   shape=[], initializer=tf.constant_initializer(1),
                            #                                   trainable=False)
                            self.opt = get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)
                        with tf.name_scope('tower_%d' % i):
                            model = as_model(FLAGS.model, input_dim=self.dataset.num_features,
                                             num_fields=self.dataset.num_fields,
                                             **self.model_param)
                            self.models.append(model)
                            tf.get_variable_scope().reuse_variables()
                            grads = self.opt.compute_gradients(model.loss)
                            self.tower_grads.append(grads)

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

            # def sparse_grads_add(grad1, grad2):
            #     indices = tf.concat([grad1.indices, grad2.indices], axis=0)
            #     values = tf.concat([grad1.values, grad2.values], axis=0)
            #     return tf.IndexedSlices(values=values, indices=indices, dense_shape=grad1.dense_shape)
            #
            # def sparse_grads_addn(grads):
            #     indices = []
            #     values = []
            #     dense_shape = grads[0].dense_shape
            #     for grad in grads:
            #         indices.append(grad.indices)
            #         values.append(grad.values)
            #     return tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)

            average_grads = []
            local_grads = []
            accumulate_op = []
            reset_op = []

            print('###################################')
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
                    local_grad = tf.Variable(zero_grad, dtype=tf.float32, trainable=False)
                    reset_grad = local_grad.assign(zero_grad)
                    if isinstance(grad, tf.IndexedSlices):
                        accumulate_grad = local_grad.scatter_sub(-grad)
                    else:
                        accumulate_grad = local_grad.assign_add(grad)
                    local_grads.append((local_grad, v))
                    accumulate_op.append(accumulate_grad)
                    reset_op.append(reset_grad)
            print('###################################')
            # TODO test op.op
            # self.grad_op = tf.group([(x[0].op, x[1].op) for x in average_grads])
            if FLAGS.lazy_update > 1:
                self.update_op = self.opt.apply_gradients(local_grads, global_step=self.global_step)
                self.grad_op = tf.group(average_grads)
                self.accumulate_op = tf.group(accumulate_op)
                self.reset_op = tf.group(reset_op)

            if not FLAGS.distributed or not FLAGS.sync:
                self.train_op = self.opt.apply_gradients(average_grads, global_step=self.global_step)
                self.hooks = None
                self.chief_only_hooks = None
            else:
                rep_op = tf.train.SyncReplicasOptimizer(opt=self.opt, replicas_to_aggregate=FLAGS.num_workers * 1000,
                                                        total_num_replicas=FLAGS.num_workers, use_locking=True)
                self.train_op = rep_op.apply_gradients(grads_and_vars=average_grads, global_step=self.global_step)
                # init_token_op = rep_op.get_init_tokens_op()
                # chief_queue_runner = rep_op.get_chief_queue_runner()
                # self.chief_only_hooks = [chief_queue_runner, init_token_op]
                self.hooks = [rep_op.make_session_run_hook(is_chief_worker())]
                self.chief_only_hooks = None
            self.saver = tf.train.Saver()
            if FLAGS.distributed:
                self.enq_ops = []
                for q in create_done_queues():
                    qop = q.enqueue(1)
                    self.enq_ops.append(qop)
                # if not FLAGS.sync:
                if not is_chief_worker():
                    for q in create_finish_queues():
                        qop = q.enqueue(1)
                        self.enq_ops.insert(0, qop)
                else:
                    chief_queue = create_finish_queue(FLAGS.task_index)
                    self.deq_op = []
                    for i in range(FLAGS.num_workers - 1):
                        self.deq_op.append(chief_queue.dequeue())

        def sess_op():
            if not FLAGS.distributed:
                return tf.Session(config=gpu_config)
            else:
                return tf.train.MonitoredTrainingSession(master=self.server.target, is_chief=(FLAGS.task_index == 0),
                                                         # TODO
                                                         # checkpoint_dir=self.ckpt_dir,
                                                         # chief_only_hooks=[tf.train.CheckpointSaverHook(
                                                         #     checkpoint_dir=self.ckpt_dir,
                                                         #     save_steps=self.num_steps,
                                                         #     saver=self.saver,
                                                         #     checkpoint_basename=self.ckpt_name)]
                                                         hooks=self.hooks,
                                                         chief_only_hooks=self.chief_only_hooks
                                                         )

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

        with sess_op() as self.sess:
            print('Train size = %d, Batch size = %d, GPUs = %d/%d' %
                  (self.dataset.train_size, FLAGS.batch_size, num_gpus, total_num_gpus))
            print('%d rounds in total, One round = %d steps, One evaluation = %d steps' %
                  (FLAGS.num_rounds, self.num_steps, self.eval_steps))

            self.train_gen = self.dataset.batch_generator(self.train_data_param)
            self.valid_gen = self.dataset.batch_generator(self.valid_data_param)
            self.test_gen = self.dataset.batch_generator(self.test_data_param)

            self.train_writer = tf.summary.FileWriter(logdir=self.train_logdir, graph=self.sess.graph, flush_secs=30)
            self.test_writer = tf.summary.FileWriter(logdir=self.test_logdir, graph=self.sess.graph, flush_secs=30)
            self.valid_writer = tf.summary.FileWriter(logdir=self.valid_logdir, graph=self.sess.graph, flush_secs=30)

            if not FLAGS.distributed:
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
            else:
                if FLAGS.restore:
                    print('Restore model from:', self.ckpt_dir)
                    print('Run initial evaluation...')
                    self.evaluate(self.test_gen, self.test_writer)

            self.begin_step = self.global_step.eval(self.sess)
            self.step = self.begin_step
            self.local_step = self.begin_step
            self.start_time = time.time()

            for r in range(1, FLAGS.num_rounds + 1):
                print('Round: %d' % r)
                train_iter = iter(self.train_gen)
                while True:
                    if (FLAGS.max_data and FLAGS.max_data <= self.step * FLAGS.batch_size) or \
                            (FLAGS.max_step and FLAGS.max_step <= self.step):
                        print('Finish %d steps, Finish %d instances, Elapsed: %.4f' %
                              (self.step, self.step * FLAGS.batch_size, time.time() - self.start_time))
                        # TODO check
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

                    train_op = self.train_op if FLAGS.lazy_update <= 1 else self.accumulate_op
                    ret = self.sess.run(fetches=[train_op, self.global_step] + fetches, feed_dict=train_feed, )
                    # TODO use variable?
                    self.local_step += 1
                    self.step = ret[1]
                    _loss_ = sum([ret[i] for i in range(2, len(ret), 3)]) / FLAGS.num_gpus
                    _log_loss_ = sum([ret[i] for i in range(3, len(ret), 3)]) / FLAGS.num_gpus
                    _l2_loss_ = sum([ret[i] for i in range(4, len(ret), 3)]) / FLAGS.num_gpus

                    if FLAGS.lazy_update <= 1 and self.local_step % FLAGS.log_frequency == 0:
                        elapsed_time = self.get_elapsed()
                        print('Local step %d, Elapsed: %.2fs' % (self.local_step, elapsed_time))

                    if FLAGS.lazy_update > 1 and self.local_step % FLAGS.lazy_update == 0:
                        self.sess.run(self.update_op)
                        self.sess.run(self.reset_op)
                        # self.evaluate(self.test_gen)
                        elapsed_time = self.get_elapsed()
                        print('Local step %d, Elapsed: %.2fs, Lazy update' % (self.local_step, elapsed_time))

                    if self.step % FLAGS.log_frequency == 0 and (FLAGS.lazy_update <= 1 or (
                            FLAGS.lazy_update > 1 and self.local_step % FLAGS.lazy_update == 0)):
                        elapsed_time = self.get_elapsed()
                        # TODO change other code to 6-bit precession
                        print('Done step %d, Elapsed: %.2fs, Train-Loss: %.6f, Log-Loss: %.6f, L2-Loss: %g'
                              % (self.step, elapsed_time, _loss_, _log_loss_, _l2_loss_))
                        summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=_loss_),
                                                    tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                                    tf.Summary.Value(tag='l2_loss', simple_value=_l2_loss_)])
                        self.train_writer.add_summary(summary, global_step=self.step)

                    if FLAGS.eval_level and self.step % self.num_steps % self.eval_steps == 0:
                        pass
                        # if not FLAGS.distributed or not FLAGS.sync or is_monitor():
                        #     elapsed_time = self.get_elapsed()
                        #     eta = FLAGS.num_rounds * self.num_steps / (self.step - self.begin_step) * elapsed_time
                        #     eval_times = self.step % self.num_steps // self.eval_steps or FLAGS.eval_level
                        #     print('Round: %d, Eval: %d / %d, AvgTime: %3.2fms, Elapsed: %.2fs, ETA: %s' %
                        #           (r, eval_times, FLAGS.eval_level, float(elapsed_time * 1000 / self.step),
                        #            elapsed_time, self.get_timedelta(eta=eta)))
                        #     self.evaluate(self.valid_gen, self.valid_writer)

                if not FLAGS.distributed:
                    pass
                    # self.saver.save(self.sess, os.path.join(self.logdir, 'checkpoints', 'model.ckpt'), self.step)
                print('Round %d finished, Elapsed: %s' % (r, self.get_timedelta()))
            if is_monitor():
                pass
                # self.evaluate(self.test_gen, self.test_writer)
            if FLAGS.distributed and is_chief_worker():
                print('Chief worker: waiting %d workers to finish' % len(self.deq_op))
                for q in self.deq_op:
                    self.sess.run(q)
                    print('Chief worker received done')
                print('Chief worker: quitting')
            print('Total Time: %s, Logdir: %s' % (self.get_timedelta(), self.logdir))
            if FLAGS.distributed:
                for op in self.enq_ops:
                    self.sess.run(op)
            print('Worker %d can not exit, should be killed' % FLAGS.task_index)

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

    def evaluate(self, gen, writer=None, eps=1e-6):
        labels = []
        preds = []
        start_time = time.time()
        _iter = iter(gen)
        flag = True
        cnt = 0
        if gen.gen_type == 'test':
            gen_size = self.dataset.test_size
        elif gen.gen_type == 'valid':
            gen_size = int(self.dataset.train_size * gen.val_ratio)
        elif gen.gen_type == 'train':
            gen_size = int(self.dataset.train_size * (1 - gen.val_ratio))
        total_step = gen_size / gen.batch_size
        while flag:
            fetches = []
            feed_dict = {}
            for model in self.models:
                try:
                    xs, ys = _iter.next()
                    cnt += 1
                    fetches.append(model.preds)
                    feed_dict[model.inputs] = xs
                    feed_dict[model.labels] = ys
                    labels.append(ys.flatten())
                    if model.training is not None:
                        feed_dict[model.training] = False
                except StopIteration:
                    flag = False
                    break
            if cnt % FLAGS.log_frequency == 0:
                elapsed = time.time() - start_time
                print('Eval step: %d / %d, Elapsed: %s' % (cnt, total_step, self.get_timedelta(elapsed)))
            if len(feed_dict):
                _preds_ = self.sess.run(fetches=fetches, feed_dict=feed_dict)
                if type(_preds_) is list:
                    preds.extend([x.flatten() for x in _preds_])
                else:
                    preds.append(_preds_.flatten())
        elapsed = time.time() - start_time
        print('Eval step: %d / %d, Elapsed: %s' % (cnt, total_step, self.get_timedelta(elapsed)))
        labels = np.hstack(labels)
        preds = np.hstack(preds)
        _min_ = len(np.where(preds < eps)[0])
        _max_ = len(np.where(preds > 1 - eps)[0])
        print('EPS: %g, %d (%.2f) < eps, %d (%.2f) > 1-eps, %d (%.2f) are truncated' %
              (eps, _min_, _min_ / len(preds), _max_, _max_ / len(preds), _min_ + _max_, (_min_ + _max_) / len(preds)))
        preds[preds < eps] = eps
        preds[preds > 1 - eps] = 1 - eps
        _log_loss_ = log_loss(y_true=labels, y_pred=preds)
        _auc_ = roc_auc_score(y_true=labels, y_score=preds)
        elapsed = time.time() - start_time
        print('%s-Loss: %.6f, AUC: %.6f, Elapsed: %s' %
              (gen.gen_type.capitalize(), _log_loss_, _auc_, self.get_timedelta(elapsed)))
        if not FLAGS.restore and writer:
            summary = tf.Summary(value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                        tf.Summary.Value(tag='auc', simple_value=_auc_)])
            writer.add_summary(summary, global_step=self.step)
        return _log_loss_, _auc_


def main(_):
    Trainer()


if __name__ == '__main__':
    tf.app.run()
