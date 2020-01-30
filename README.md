``Note``: I collect some interesting discusions from emails and put then behind.

``Note``: Any problems, you can contact me at [kevinqu16@gmail.com](kevinqu16@gmail.com).
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


## Interesting Discussions

``Note``: some basic problems are discussed in the short [demo](https://github.com/Atomu2014/product-nets), while more advanced problems are discussed here because they are more relavant to this extended work.

### Overfitting of Adam

I our experiments, we found adam makes the models converge much faster than other optimizers. However, these models would overfit severely in 1-2 epochs, including shallow models (e.g. FM) and deep models.

The first solution is by choosing different optimizers. We find adagrad is more robust when learning rate and L2 regularization are set properly, and is not overfitting after 30+ epochs. However, the performance of adagrad on test set is no better than adam. Thus there is a trade-off between robustness and performance. For some enviroments requiring high robustness (e.g., online experiments), adagrad may be a better choice.

I believe the overfitting of adam starts from the embedding vectors, i.e., the embedding vectors are more likely to overfit. And I notice the variance of embedding vectors grows steadily when using adam even with large L2 penalty. An assumption about the gradients of adam has been discussed in the paper 4.3.

Another solution is regularization. In paper 4.4, we empirically discussed several popular regularization methods. In addition, I discussed those methods with other teams working on this direction through email. The most effective regularization would be ``weight decay`` and ``learning rate schedule``. Many proofs can been found, including the paper [Fixing weight decay regularization in Adam](https://openreview.net/forum?id=rk6qdGgCZ), the performance of adagrad (adagrad naturally decays learning rates during training), and some practices in other tasks (e.g., the training code of Bert by google and GPT by openai).

The last question is why overfitting in recommendation/CTR prediction. A possible reason is the distribution shift between train test sets is much severe in high-dimensional sparse scenarios.

### Performance Gain of Deep Models

Even though more and more complicated deep models are proposed for such high-dimensional sparse data, deeper networks can hardly outperform shallower networks significantly. In another word, the performance gain is less and less when networks grow deeper and deeper.

An assumption is, DNN has much lower sampling complexity than shallow models, yet its approximation ability is not significantly better.

A detailed discussion of sampling complexity on sparse data can be found at [The Sample Complexity of Online One-Class Collaborative Filtering](https://arxiv.org/abs/1706.00061). This paper proposes, there is a lower bound for a specific sparse feature such that CF model can learn this feature very well when sufficient samples are trained. In another word, a good model only requires a certain amount of samples (of a feature). Even though this paper only discusses one-class CF models, if we suppose the conclusion also applies to DNNs, we can explain some experiment results of DNNs.

- DNNs usually perform much better in cold-start problems. However, when we downsample the dataset and filter out low-frequency features, the performance gain of DNNs usually drops. 
- Increasing the depth of DNNs does not significant improve the performance. 

If we assume 1. DNNs require less samples then shallow models to achieve the same performance, 2. DNNs do not significantly outperform shallow models when sufficient samples (much more than the sampling complixity of shallow models) are provided, the above results can be easily explained.

The 2nd assumption relies on the approximation ability of DNNs, which has been discussed in the paper 3.2 and 5.4. And we draw the conclusion, "ideal" dataset is unfair for DNNs, and "downsample" may be harmful to compare the learning ability of shallow and deep models. Instead of increasing the capacity of DNNs and propose more and more complicated models, we should focus more on cold-start problems, which is a bigger and more important challenge in high-dimensional sparse data. Also, extending the sampling complexity theory to DNNs would also be interesting!

Besides, some recent advances in learning combinatorial features with DNNs are also intersting and are worth following.
