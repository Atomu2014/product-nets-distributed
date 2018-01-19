from __future__ import division
from __future__ import print_function

import shutil
import sys
import time

import tensorflow as tf

import __init__

sys.path.append(__init__.config['data_path'])
from datasets import as_dataset

from tf_models import as_model

# Flags for defining the tf.train.ClusterSpec
# tf.app.flags.DEFINE_string('ps_hosts', '172.16.2.245:12345', 'Comma-separated list of hostname:port pairs')
# tf.app.flags.DEFINE_string('worker_hosts', '172.16.2.245:12346,172.16.2.245:12347',
#                            'Comma-separated list of hostname:port pairs')

tf.app.flags.DEFINE_string('ps_hosts', 'localhost:12345', 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_hosts', 'localhost:12346,localhost:12347',
                           'Comma-separated list of hostname:port pairs')

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string('job_name', '', 'One of ps, worker')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
tf.app.flags.DEFINE_string('log_dir', '../log', 'Directory for storing mnist data')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 1000, 'testing batch size')
tf.app.flags.DEFINE_integer('workers', 2, 'Number of workers')
tf.app.flags.DEFINE_integer('ps', 1, 'Number of ps')
tf.app.flags.DEFINE_integer('max_step', 1000, 'Number of max steps')

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

backend = 'tf'
data_name = 'ipinyou'
model_name = 'ccpm'


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

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)

    if FLAGS.job_name == 'ps':
        # TODO remove log dir to build a new graph
        shutil.rmtree(FLAGS.log_dir, True)

        sess = tf.Session(server.target)
        queue = create_done_queue(FLAGS.task_index)

        # wait until all workers are done
        for i in range(FLAGS.workers):
            sess.run(queue.dequeue())
            print('ps %d received done %d' % (FLAGS.task_index, i))

        print('ps %d: quitting' % (FLAGS.task_index))
    elif FLAGS.job_name == 'worker':
        dataset = as_dataset(data_name)

        train_data_param = {
            'gen_type': 'train',
            'random_sample': True,
            'batch_size': FLAGS.batch_size,
            'squeeze_output': False,
        }
        test_data_param = {
            'gen_type': 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size,
            'squeeze_output': False,
        }
        # TODO implement parallel reading
        train_gen = dataset.batch_generator(train_data_param)
        test_gen = dataset.batch_generator(test_data_param)

        train_gen = iter(train_gen)
        test_gen = iter(test_gen)

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % FLAGS.task_index,
                cluster=cluster)):

            # TODO reset default graph
            model = as_model(model_name,
                             input_dim=dataset.num_features,
                             num_fields=dataset.num_fields,
                             # sub_nn_layers=[
                             #     ('full', 5),
                             #     ('act', 'relu'),
                             #     ('drop', 0.9),
                             #     ('full', 1), ],
                             nn_layers=[
                                 ('conv', (5, 10)),
                                 ('act', 'relu'),
                                 ('drop', 0.5),
                                 ('flat', (1, 2)),
                                 ('full', 100),
                                 ('act', 'relu'),
                                 ('drop', 0.5),
                                 ('full', 1), ],
                             )

            global_step = tf.Variable(0, name='global_step', trainable=False)

            # train_op = tf.train.AdagradOptimizer(0.01).minimize(model.loss, global_step=global_step)

            opt_op = tf.train.AdagradOptimizer(0.01)
            train_op = tf.train.SyncReplicasOptimizer(opt_op, replicas_to_aggregate=FLAGS.workers,
                                                      total_num_replicas=FLAGS.workers).minimize(model.loss,
                                                                                                 global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            enq_ops = []
            for q in create_done_queues():
                qop = q.enqueue(1)
                enq_ops.append(qop)

        # Create a 'supervisor', which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=FLAGS.log_dir,  # + '_' + str(FLAGS.task_index),
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)

        begin_time = time.time()
        frequency = 100
        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            # Loop until the supervisor shuts down or 100000 steps have completed.
            step = 0
            while not sv.should_stop() and step < FLAGS.max_step:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                start_time = time.time()

                batch_xs, batch_ys = train_gen.next()
                train_feed = {model.inputs: batch_xs, model.labels: batch_ys}
                if model.training is not None:
                    train_feed[model.training] = True

                _, step = sess.run([train_op, global_step], feed_dict=train_feed)
                elapsed_time = time.time() - start_time
                if step % frequency == 0:
                    print('Done step %d' % step, ' AvgTime: %3.2fms' % float(elapsed_time * 1000 / frequency))

            # Test trained model
            test_xs, test_ys = test_gen.next()
            test_feed = {model.inputs: test_xs, model.labels: test_ys}
            if model.training is not None:
                test_feed[model.training] = False
            print('Test-Loss: %2.4f' % sess.run(model.loss, feed_dict=test_feed))

            # signal to ps shards that we are done
            # for q in create_done_queues():
            # sess.run(q.enqueue(1))
            for op in enq_ops:
                sess.run(op)
        print('Total Time: %3.2fs' % float(time.time() - begin_time))

        # Ask for all the services to stop.
        sv.stop()


if __name__ == '__main__':
    tf.app.run()
