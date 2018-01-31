from __future__ import division
from __future__ import print_function

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import __init__

sys.path.append(__init__.config['data_path'])
from datasets import as_dataset

from tf_models import as_model
from print_hook import PrintHook

FLAGS = tf.app.flags.FLAGS
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string('ps_hosts', 'localhost:12445', 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_hosts', 'localhost:12446,localhost:12447',
                           'Comma-separated list of hostname:port pairs')

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string('job_name', '', 'One of ps, worker')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_integer('ps', 1, 'Number of ps')
tf.app.flags.DEFINE_integer('workers', 2, 'Number of workers')
tf.app.flags.DEFINE_bool('sync', True, 'Synchronized training')

tf.app.flags.DEFINE_string('logdir', '../log', 'Directory for storing mnist data')
tf.app.flags.DEFINE_bool('val', False, 'If True, use validation set, else use test set')
tf.app.flags.DEFINE_integer('batch_size', 2000, 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 10000, 'testing batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate')

tf.app.flags.DEFINE_string('dataset', 'criteo_9d', 'Dataset = ipinyou, avazu, criteo, criteo_9d, criteo_16d"')
tf.app.flags.DEFINE_string('model', 'fnn', 'Model type = lr, fm, ffm, kfm, nfm, fnn, ccpm, deepfm, ipnn, kpnn, pin')
tf.app.flags.DEFINE_float('l2_scale', 0., 'L2 regularization')
tf.app.flags.DEFINE_integer('embed_size', 20, 'Embedding size')
# e.g. [["conv", [5, 10]], ["act", "relu"], ["drop", 0.5], ["flat", [1, 2]], ["full", 100], ["act", "relu"], ["drop", 0.5], ["full", 1]]
tf.app.flags.DEFINE_string('nn_layers', '[["full", 100], ["act", "relu"], ["full", 1]]',
                           'Network structure')
# e.g. [["full", 5], ["act", "relu"], ["drop", 0.9], ["full", 1]]
tf.app.flags.DEFINE_string('sub_nn_layers', '', 'Sub-network structure')

# tf.app.flags.DEFINE_integer('max_step', 1000, 'Number of max steps')
tf.app.flags.DEFINE_integer('max_data', 0, 'Number of instances')
tf.app.flags.DEFINE_integer('num_rounds', 1, 'Number of training rounds')
tf.app.flags.DEFINE_integer('eval_level', 1, 'Evaluating frequency in one round')
tf.app.flags.DEFINE_integer('log_frequency', 100, 'Logging frequency')


def create_done_queue(i):
    with tf.device('/job:ps/task:%d' % (i)):
        return tf.FIFOQueue(FLAGS.workers, tf.int32, shared_name='done_queue' + str(i))


def create_done_queues():
    return [create_done_queue(i) for i in range(FLAGS.ps)]


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)

    _config_ = {}
    logdir = '%s/%s/%s/%s' % (FLAGS.logdir, FLAGS.dataset, FLAGS.model, datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = open(logdir + '/log', 'a')
    _config_['logdir'] = logdir

    def MyHookOut(text):
        logfile.write(text)
        logfile.flush()
        return 1, 0, text

    phOut = PrintHook()
    phOut.Start(MyHookOut)

    if FLAGS.job_name == 'ps':
        # TODO remove log dir to build a new graph
        # shutil.rmtree(logdir, True)
        sess = tf.Session(server.target)
        queue = create_done_queue(FLAGS.task_index)
        # wait until all workers are done
        for i in range(FLAGS.workers):
            sess.run(queue.dequeue())
            print('ps %d received done %d' % (FLAGS.task_index, i))
        print('ps %d: quitting' % (FLAGS.task_index))
    elif FLAGS.job_name == 'worker':
        train_data_param = {
            'gen_type': 'train',
            'random_sample': True,
            'batch_size': FLAGS.batch_size,
            'squeeze_output': False,
            'num_workers': FLAGS.workers,
            'task_index': FLAGS.task_index,
        }
        valid_data_param = {
            'gen_type': 'valid',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size,
            'squeeze_output': False,
        }
        test_data_param = {
            'gen_type': 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size,
            'squeeze_output': False,
        }
        dataset = as_dataset(FLAGS.dataset)
        train_gen = dataset.batch_generator(train_data_param)
        test_gen = dataset.batch_generator(test_data_param)
        valid_gen = dataset.batch_generator(valid_data_param) if FLAGS.val else test_gen
        _config_['train_data_param'] = train_data_param
        _config_['valid_data_param'] = valid_data_param
        _config_['test_data_param'] = test_data_param
        _config_['dataset'] = FLAGS.dataset

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % FLAGS.task_index,
                cluster=cluster)):

            # TODO reset default graph
            model_param = {'l2_scale': FLAGS.l2_scale}
            if FLAGS.model != 'lr':
                model_param['embed_size'] = FLAGS.embed_size
            if FLAGS.model in ['fnn', 'ccpm', 'deepfm', 'ipnn', 'kpnn', 'pin']:
                model_param['nn_layers'] = [tuple(x) for x in json.loads(FLAGS.nn_layers)]
            if FLAGS.model in ['nfm', 'pin']:
                model_param['sub_nn_layers'] = [tuple(x) for x in json.loads(FLAGS.sub_nn_layers)]
            model = as_model(FLAGS.model, input_dim=dataset.num_features, num_fields=dataset.num_fields, **model_param)
            _config_['model'] = FLAGS.model
            _config_['model_param'] = model_param
            _config_['learning_rate'] = FLAGS.learning_rate
            _config_['batch_size'] = FLAGS.batch_size
            _config_['num_rounds'] = FLAGS.num_rounds
            _config_['log_frequency'] = FLAGS.log_frequency
            _config_['eval_level'] = FLAGS.eval_level
            _config_json_ = json.dumps(_config_, indent=4, sort_keys=True, separators=(',', ':'))
            if FLAGS.task_index == 0:
                open(logdir + '/config.json', 'w').write(_config_json_)

            global_step = tf.Variable(1, name='global_step', trainable=False)
            if not FLAGS.sync:
                train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)
            else:
                optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
                grads_and_vars = optimizer.compute_gradients(model.loss)
                rep_op = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=FLAGS.workers,
                                                        total_num_replicas=FLAGS.workers, use_locking=True)
                train_op = rep_op.apply_gradients(grads_and_vars, global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            enq_ops = []
            for q in create_done_queues():
                qop = q.enqueue(1)
                enq_ops.append(qop)

        # Create a 'supervisor', which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=logdir,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step, )
        # save_model_secs=60)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            if FLAGS.sync:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            if FLAGS.task_index == 0:
                train_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'train'), graph=sess.graph,
                                                     flush_secs=30)
                test_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'test'), graph=sess.graph,
                                                    flush_secs=30)
                valid_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'valid'), graph=sess.graph,
                                                     flush_secs=30) if FLAGS.val else test_writer

            # Loop until the supervisor shuts down or 100000 steps have completed.
            step = 1
            start_time = time.time()
            num_steps = int(np.ceil(dataset.train_size / FLAGS.batch_size / FLAGS.workers))
            print('%d rounds, %d steps per round' % (FLAGS.num_rounds, num_steps))
            for r in range(FLAGS.num_rounds):
                for batch_xs, batch_ys in train_gen:
                    if FLAGS.sync:
                        if step * FLAGS.batch_size * FLAGS.workers == 1000000:
                            print('Finish')
                            exit(0)
                    else:
                        if step * FLAGS.batch_size == 1000000:
                            print('Finish')
                            exit(0)
                    train_feed = {model.inputs: batch_xs, model.labels: batch_ys}
                    if model.training is not None:
                        train_feed[model.training] = True

                    _, step, _loss_, _log_loss_, _l2_loss_ = sess.run(
                        [train_op, global_step, model.loss, model.log_loss,
                         model.l2_loss], feed_dict=train_feed)
                    elapsed_time = time.time() - start_time
                    if step % FLAGS.log_frequency == 0:
                        print('Done step %d AvgTime: %3.2fms, Elapsed: %.2fs, Train-Loss: %.4f' % (
                            step, float(elapsed_time * 1000 / step), elapsed_time, _loss_))
                        if FLAGS.task_index == 0:
                            summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=_loss_),
                                                        tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                                        tf.Summary.Value(tag='l2_loss', simple_value=_l2_loss_)])
                            train_writer.add_summary(summary, global_step=step)

                    if FLAGS.task_index == 0:
                        if r < FLAGS.num_rounds - 1 or step % num_steps:
                            if FLAGS.eval_level and (
                                                step % int(
                                                np.ceil(num_steps / FLAGS.eval_level)) == 0 or step % num_steps == 0):
                                _log_loss_, _auc_ = model.eval(valid_gen, sess)
                                summary = tf.Summary(
                                    value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                           tf.Summary.Value(tag='auc', simple_value=_auc_)])
                                valid_writer.add_summary(summary, global_step=step)
                if FLAGS.task_index == 0:
                    saver.save(sess, os.path.join(logdir, 'checkpoints', 'model.ckpt'), step)

            if FLAGS.task_index == 0:
                _log_loss_, _auc_ = model.eval(test_gen, sess)
                summary = tf.Summary(value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                            tf.Summary.Value(tag='auc', simple_value=_auc_)])
                test_writer.add_summary(summary, global_step=step)
            print('Total Time: %3.2fs' % float(time.time() - start_time))
            # signal to ps shards that we are done
            # for q in create_done_queues():
            # sess.run(q.enqueue(1))
            for op in enq_ops:
                sess.run(op)

        # Ask for all the services to stop.
        sv.stop()


if __name__ == '__main__':
    tf.app.run()
