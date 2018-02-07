import json
import os
from datetime import datetime

import tensorflow as tf

import __init__
from print_hook import PrintHook


def add_config_to_json(FLAGS, config):
    for k, v in FLAGS.__flags.iteritems():
        config[k] = getattr(FLAGS, k)
    for k, v in __init__.config.iteritems():
        if k != 'default':
            config[k] = v


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


def add_param_to_json(params, config):
    for k, v in params.iteritems():
        config[k] = v


def dump_config(logdir, config):
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
