``Note``: Any problems, you can contact me at [kevinqu@apex.sjtu.edu.cn](kevinqu@apex.sjtu.edu.cn),
or [kevinqu16@gmail.com](kevinqu16@gmail.com).
Through email, you will get my rapid response.

This repository provides the experiments code of the paper [Product-based Neural Networks for
User Response Prediction over Multi-field Categorical Data](https://arxiv.org/abs/1807.00311).
This paper extends the [conference paper](https://arxiv.org/abs/1611.00144), and is accepted by TOIS.

In general, the journal paper extends the conference paper in 3 aspects:
- We analyze the ``coupled gradient issue`` of FM-like models, and propose ``kernel product`` to solve it.
We apply 2 types of ``kernel product`` in FM, (Kernel FM (KFM) and Network in FM (NIFM)), to verify this issue.
- We analyze the ``insensitive gradient issue`` of DNN-based models, and propose to use interactive feature extractors to tackle this issue.
Using FM, KFM, and NIFM as feature extractors, we propose Inner PNN (IPNN), Kernel PNN (KPNN),
and Product-network In Network (PIN).
- We study practical issue in training and generalization. We conduct more offline experiments and an online A/B test.

The [demo](https://github.com/Atomu2014/product-nets) of the conference paper implements IPNN, KPNN and some baselines, and it is easier to read/run.
This repository implements all the proposed models as well as baseline models in tensorflow, with large-scale data access, multi-gpu support, and distributed training support.

## How to run

In order to accelerate large-scale data access, we use hdf format to store the processed data.
The datasets and APIs are at [Ads-RecSys-Datasets](https://github.com/Atomu2014/Ads-RecSys-Datasets).
The first step you should download the APIs and datasets at ``path/to/data``.
You can check data access through:

    #!/usr/bin/env python
    import sys
    sys.path.append('path/to/data')
    from __future__ import print_function
    from datasets import iPinYou

    data = iPinYou()
    data.summary()
    train_gen = data.batch_generator('train', batch_size=1000)

    for X, y in train_gen:
        print(X.shape, y.shape)
        exit(0)

The second step you should download this repository and configure ``data_path`` in ``__init__.py``:

    config['data_path'] = 'path/to/data'

This repository contains 6 ``.py`` files, they are:

- ``__init__.py``: store configuration.
- ``criteo_challenge.py``: our solution to the [Criteo Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge). 
- ``kill.py``: find and kill the threads when using distributed training.
- ``print_hook.py``: redirect stdout to logfile.
- ``tf_main.py``: training file.
- ``tf_models.py``: all the models, including LR, FM, FFM, KFM, NIFM, FNN, CCPM, DeepFM, IPNN, KPNN, and PIN.

You can run ``tf_main.py`` in different settings, for example:

Local training, 1 GPU:

    CUDA_VISIBLE_DEVICES="0" python tf_main.py --distributed=False --num_gpus=1 --dataset=criteo --model=fnn --batch_size=2000 --optimizer=adam --learning_rate=1e-3 --embed_size=20 --nn_layers="[[\"full\", 400], [\"act\", \"relu\"], [\"full\", 1]]" --num_rounds=1

Local training, 2 GPUs:
    
    CUDA_VISIBLE_DEVICES="0,1" python tf_main.py --distributed=False --num_gpus=2 --dataset=avazu --model=fm --batch_size=2000 --optimizer=adagrad --learning_rate=0.1 --embed_size=40 --num_rounds=3

Distributed training:

    # ps host
    python tf_main.py --distributed=True --ps_hosts='ps_host0' --worker_hosts='worker_host0,worker_host1' --job_name=ps --task_index=0 --tag=XXX
    
    # worker host 0
    python tf_main.py --distributed=True --ps_hosts='ps_host0' --worker_hosts='worker_host0,worker_host1' --job_name=worker --task_index=0 --worker_num_gpus=2,2 --tag=XXX

    # worker host 1
    python tf_main.py --distributed=True --ps_hosts='ps_host0' --worker_hosts='worker_host0,worker_host1' --job_name=worker --task_index=1 --worker_num_gpus=2,2 --tag=XXX


## Criteo Challenge
``criteo_challenge.py`` contains our solution to the contest.
In this contest, [libFFM](https://github.com/guestwalk/kaggle-2014-criteo) (master branch) was the winning solution with log loss = 0.44506/0.44520 on the private/public leaderboard, and achieves 0.44493/0.44508 after calibration. We train KFM with the same training files as libFFM on one 1080Ti.
KFM achieves 0.44484/0.44492 on the private/public leaderboard, and achieves 0.44484/0.44491 after calibration.