from __future__ import division
from __future__ import print_function

import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf

import __init__

sys.path.append(__init__.config['data_path'])
from datasets import as_dataset

from tf_models import as_model
from print_hook import PrintHook

default_values = __init__.default_values_nmz
config = {}

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', default_values['logdir'], 'Directory for storing mnist data')
tf.app.flags.DEFINE_bool('restore', default_values['restore'], 'Restore from logdir')
tf.app.flags.DEFINE_bool('val', default_values['val'], 'If True, use validation set, else use test set')
tf.app.flags.DEFINE_integer('batch_size', default_values['batch_size'], 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', default_values['test_batch_size'], 'Testing batch size')
tf.app.flags.DEFINE_float('learning_rate', default_values['learning_rate'], 'Learning rate')

tf.app.flags.DEFINE_string('dataset', default_values['dataset'],
                           'Dataset = ipinyou, avazu, criteo, criteo_9d, criteo_16d"')
tf.app.flags.DEFINE_string('model', default_values['model'],
                           'Model type = lr, fm, ffm, kfm, nfm, fnn, ccpm, deepfm, ipnn, kpnn, pin')
tf.app.flags.DEFINE_string('optimizer', default_values['optimizer'], 'Optimizer')
tf.app.flags.DEFINE_float('l2_scale', default_values['l2_scale'], 'L2 regularization')
tf.app.flags.DEFINE_integer('embed_size', default_values['embed_size'], 'Embedding size')
# e.g. [["conv", [5, 10]], ["act", "relu"], ["drop", 0.5], ["flat", [1, 2]], ["full", 100], ["act", "relu"], ["drop", 0.5], ["full", 1]]
tf.app.flags.DEFINE_string('nn_layers', default_values['nn_layers'], 'Network structure')
# e.g. [["full", 5], ["act", "relu"], ["drop", 0.9], ["full", 1]]
tf.app.flags.DEFINE_string('sub_nn_layers', default_values['sub_nn_layers'], 'Sub-network structure')

tf.app.flags.DEFINE_integer('max_step', default_values['max_step'], 'Number of max steps')
tf.app.flags.DEFINE_integer('max_data', default_values['max_data'], 'Number of instances')
tf.app.flags.DEFINE_integer('num_rounds', default_values['num_rounds'], 'Number of training rounds')
tf.app.flags.DEFINE_integer('eval_level', default_values['eval_level'], 'Evaluating frequency level')
tf.app.flags.DEFINE_integer('log_frequency', default_values['log_frequency'], 'Logging frequency')


def add_config_to_json():
    for k, v in FLAGS.__flags.iteritems():
        config[k] = getattr(FLAGS, k)
    for k, v in __init__.config.iteritems():
        if k != 'default':
            config[k] = v


def get_logdir():
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


def add_param_to_json(params):
    for k, v in params.iteritems():
        config[k] = v


def dump_config(logdir):
    config_json = json.dumps(config, indent=4, sort_keys=True, separators=(',', ':'))
    print(config_json)
    path_json = os.path.join(logdir, 'config.json')
    cnt = 1
    while os.path.exists(path_json):
        path_json = os.path.join(logdir, 'config%d.json' % cnt)
        cnt += 1
    print('Config json file:', path_json)
    open(path_json, 'w').write(config_json)


def get_optimizer(opt, lr, **kwargs):
    opt = opt.lower()
    eps = kwargs['epsilon'] if 'epsilon' in kwargs else 1e-8
    if opt == 'sgd' or opt == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif opt == 'adam':
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)
    elif opt == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=lr)


def main(_):
    logdir, logfile = get_logdir()

    redirect_stdout(logfile)

    train_data_param = {
        'gen_type': 'train',
        'random_sample': True,
        'batch_size': FLAGS.batch_size,
        'squeeze_output': False,
    }
    valid_data_param = {
        'gen_type': 'valid' if FLAGS.val else 'test',
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

    model_param = {'l2_scale': FLAGS.l2_scale}
    if FLAGS.model != 'lr':
        model_param['embed_size'] = FLAGS.embed_size
    if FLAGS.model in ['fnn', 'ccpm', 'deepfm', 'ipnn', 'kpnn', 'pin']:
        model_param['nn_layers'] = [tuple(x) for x in json.loads(FLAGS.nn_layers)]
    if FLAGS.model in ['nfm', 'pin']:
        model_param['sub_nn_layers'] = [tuple(x) for x in json.loads(FLAGS.sub_nn_layers)]

    add_config_to_json()

    add_param_to_json({'train_data_param': train_data_param, 'valid_data_param': valid_data_param,
                       'test_data_param': test_data_param, 'logdir': logdir})

    dump_config(logdir)

    tf.reset_default_graph()

    model = as_model(FLAGS.model, input_dim=dataset.num_features, num_fields=dataset.num_fields, **model_param)

    gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, )
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    global_step = tf.Variable(1, name='global_step', trainable=False)
    opt = get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)
    train_op = opt.minimize(model.loss, global_step=global_step)
    saver = tf.train.Saver()

    train_gen = dataset.batch_generator(train_data_param)
    test_gen = dataset.batch_generator(test_data_param)
    valid_gen = dataset.batch_generator(valid_data_param)
    train_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'train'), graph=sess.graph, flush_secs=30)
    test_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'test'), graph=sess.graph, flush_secs=30)
    valid_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'valid'), graph=sess.graph, flush_secs=30)

    if not FLAGS.restore:
        tf.global_variables_initializer().run(session=sess)
        tf.local_variables_initializer().run(session=sess)
    else:
        checkpoint_state = tf.train.get_checkpoint_state(os.path.join(logdir, 'checkpoints'))
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            saver.restore(sess, checkpoint_state.model_checkpoint_path)
            print('Restore model from:', checkpoint_state.model_checkpoint_path)
            print('Run initial evaluation...')
            model.eval(test_gen, sess)
        else:
            print('Restore failed')

    begin_step = global_step.eval(sess)
    step = begin_step
    start_time = time.time()
    num_steps = int(np.ceil(dataset.train_size / FLAGS.batch_size))
    eval_steps = int(np.ceil(num_steps / FLAGS.eval_level)) if FLAGS.eval_level else 0
    print('%d rounds in total, One round = %d steps, One evaluation = %d steps' %
          (FLAGS.num_rounds, num_steps, eval_steps))
    for r in range(1, FLAGS.num_rounds + 1):
        print('Round: %d' % r)
        for batch_xs, batch_ys in train_gen:
            if (FLAGS.max_data and FLAGS.max_data <= step * FLAGS.batch_size) or \
                    (FLAGS.max_step and FLAGS.max_step <= step):
                print('Finish %d steps, Finish %d instances, Elapsed: %.4f' %
                      (step, step * FLAGS.batch_size, time.time() - start_time))
                exit(0)

            train_feed = {model.inputs: batch_xs, model.labels: batch_ys}
            if model.training is not None:
                train_feed[model.training] = True

            _, step, _loss_, _log_loss_, _l2_loss_ = sess.run([train_op, global_step, model.loss, model.log_loss,
                                                               model.l2_loss], feed_dict=train_feed)
            if step % FLAGS.log_frequency == 0:
                elapsed_time = time.time() - start_time
                print('Done step %d, Elapsed: %.2fs, Train-Loss: %.4f, Log-Loss: %.4f, L2-Loss: %g'
                      % (step, elapsed_time, _loss_, _log_loss_, _l2_loss_))
                summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=_loss_),
                                            tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                            tf.Summary.Value(tag='l2_loss', simple_value=_l2_loss_)])
                train_writer.add_summary(summary, global_step=step)

            if FLAGS.eval_level and step % num_steps % eval_steps == 0:
                elapsed_time = time.time() - start_time
                eta = FLAGS.num_rounds * num_steps / (step - begin_step) * elapsed_time
                eval_times = step % num_steps // eval_steps or FLAGS.eval_level
                print('Round: %d, Eval: %d / %d, AvgTime: %3.2fms, Elapsed: %.2fs, ETA: %s' %
                      (r, eval_times, FLAGS.eval_level, float(elapsed_time * 1000 / step), elapsed_time,
                       str(timedelta(seconds=eta))))
                _log_loss_, _auc_ = model.eval(valid_gen, sess)
                summary = tf.Summary(value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                            tf.Summary.Value(tag='auc', simple_value=_auc_)])
                valid_writer.add_summary(summary, global_step=step)
        saver.save(sess, os.path.join(logdir, 'checkpoints', 'model.ckpt'), step)
        if FLAGS.eval_level < 1:
            print('Round %d finished, Elapsed: %s' % (r, str(timedelta(seconds=(time.time() - start_time)))))
    _log_loss_, _auc_ = model.eval(test_gen, sess)
    summary = tf.Summary(value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                tf.Summary.Value(tag='auc', simple_value=_auc_)])
    test_writer.add_summary(summary, global_step=step)
    print('Total Time: %s, Logdir: %s' % (str(timedelta(seconds=(time.time() - start_time))), logdir))


if __name__ == '__main__':
    tf.app.run()
