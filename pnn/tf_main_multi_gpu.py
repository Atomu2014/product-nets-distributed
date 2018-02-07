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

from tf_models_share_vars import as_model
import tf_utils

default_values = __init__.default_values_nmz
config = {}

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_gpus', 2, '# gpus')
tf.app.flags.DEFINE_string('logdir', '../log', 'Directory for storing mnist data')
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


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
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


def main(_):
    logdir, logfile = tf_utils.get_logdir(FLAGS=FLAGS)

    tf_utils.redirect_stdout(logfile)

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

    tf_utils.add_config_to_json(FLAGS=FLAGS, config=config)

    tf_utils.add_param_to_json({'train_data_param': train_data_param, 'valid_data_param': valid_data_param,
                                'test_data_param': test_data_param, 'logdir': logdir}, config=config)

    tf_utils.dump_config(logdir, config=config)

    tf.reset_default_graph()

    global_step = tf.Variable(1, name='global_step', trainable=False)
    opt = tf_utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)
    saver = tf.train.Saver()

    tower_grads = []
    models = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                print('Deploying gpu:%d ...' % i)
                with tf.name_scope('tower_%d' % i):
                    model = as_model(FLAGS.model, input_dim=dataset.num_features, num_fields=dataset.num_fields,
                                     **model_param)
                    models.append(model)
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(model.loss)
                    tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    # for grad, var in grads:
    #     if grad is not None:
    #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # for var in tf.trainable_variables():
    #     summaries.append(tf.summary.histogram(var.op.name, var))
    train_op = apply_gradient_op

    train_gen = dataset.batch_generator(train_data_param)
    test_gen = dataset.batch_generator(test_data_param)
    valid_gen = dataset.batch_generator(valid_data_param)

    gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, )
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    train_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'train'), graph=sess.graph, flush_secs=30)
    test_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'test'), graph=sess.graph, flush_secs=30)
    valid_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'valid'), graph=sess.graph, flush_secs=30)

    if not FLAGS.restore:
        tf.global_variables_initializer().run(session=sess)
        # tf.local_variables_initializer().run(session=sess)
    else:
        checkpoint_state = tf.train.get_checkpoint_state(os.path.join(logdir, 'checkpoints'))
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            saver.restore(sess, checkpoint_state.model_checkpoint_path)
            print('Restore model from:', checkpoint_state.model_checkpoint_path)
            print('Run initial evaluation...')
            models[0].eval(test_gen, sess)
        else:
            print('Restore failed')

    # tf.train.start_queue_runners(sess=sess)

    def evaluate(gen, sess):
        labels = []
        preds = []
        start_time = time.time()
        _iter = iter(gen)
        fetches = []
        feed_dict = {}
        for model in models:
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
        _preds_ = sess.run(fetches=fetches, feed_dict=feed_dict)
        if type(_preds_) is list:
            preds.extend(_preds_)
        else:
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

    begin_step = global_step.eval(sess)
    step = begin_step
    start_time = time.time()
    num_steps = int(np.ceil(dataset.train_size / FLAGS.batch_size / FLAGS.num_gpus))
    eval_steps = int(np.ceil(num_steps / FLAGS.eval_level)) if FLAGS.eval_level else 0
    print('%d rounds in total, One round = %d steps, One evaluation = %d steps' %
          (FLAGS.num_rounds, num_steps, eval_steps))
    for r in range(1, FLAGS.num_rounds + 1):
        print('Round: %d' % r)
        train_iter = iter(train_gen)
        while True:
            if (FLAGS.max_data and FLAGS.max_data <= step * FLAGS.batch_size) or \
                    (FLAGS.max_step and FLAGS.max_step <= step):
                print('Finish %d steps, Finish %d instances, Elapsed: %.4f' %
                      (step, step * FLAGS.batch_size, time.time() - start_time))
                exit(0)

            fetches = []
            train_feed = {}
            try:
                for model in models:
                    batch_xs, batch_ys = train_iter.next()
                    fetches += [model.loss, model.log_loss, model.l2_loss]
                    train_feed[model.inputs] = batch_xs
                    train_feed[model.labels] = batch_ys
                    if model.training is not None:
                        train_feed[model.training] = True
            except StopIteration:
                break
            ret = sess.run(fetches=[train_op, global_step] + fetches, feed_dict=train_feed)
            step = ret[1]
            _loss_ = sum([ret[i] for i in range(2, len(ret), 3)]) / FLAGS.num_gpus
            _log_loss_ = sum([ret[i] for i in range(3, len(ret), 3)]) / FLAGS.num_gpus
            _l2_loss_ = sum([ret[i] for i in range(4, len(ret), 3)]) / FLAGS.num_gpus

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
                _log_loss_, _auc_ = evaluate(valid_gen, sess)
                summary = tf.Summary(value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                            tf.Summary.Value(tag='auc', simple_value=_auc_)])
                valid_writer.add_summary(summary, global_step=step)
        saver.save(sess, os.path.join(logdir, 'checkpoints', 'model.ckpt'), step)
        if FLAGS.eval_level < 1:
            print('Round %d finished, Elapsed: %s' % (r, str(timedelta(seconds=(time.time() - start_time)))))
    _log_loss_, _auc_ = evaluate(test_gen, sess)
    summary = tf.Summary(value=[tf.Summary.Value(tag='log_loss', simple_value=_log_loss_),
                                tf.Summary.Value(tag='auc', simple_value=_auc_)])
    test_writer.add_summary(summary, global_step=step)
    print('Total Time: %s, Logdir: %s' % (str(timedelta(seconds=(time.time() - start_time))), logdir))


if __name__ == '__main__':
    tf.app.run()
