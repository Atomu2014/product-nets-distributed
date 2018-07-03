``Note``: Any problems, you can contact me at [kevinqu@apex.sjtu.edu.cn](kevinqu@apex.sjtu.edu.cn),
or [kevinqu16@gmail.com](kevinqu16@gmail.com).
Through email, you will get my rapid response.

This repository provides the experiments code of the paper [Product-based Neural Networks for
User Response Prediction over Multi-field Categorical Data](https://arxiv.org/abs/1807.00311).
This paper extends the [conference paper](https://arxiv.org/abs/1611.00144), and this paper is accepted by TOIS.

In general, the journal paper extends the conference paper in 3 aspects:
- We analyze the ``coupled gradient issue`` of FM-like models, and propose ``kernel product`` to solve it.
We apply 2 types of ``kernel product`` in FM, (namely, Kernel FM (KFM) and Network in FM (NIFM)), to verify this issue.
- We analyze the ``insensitive gradient issue`` of DNN-based models, and propose to use interactive feature extractors
to tackle this issue.
Using FM, KFM, and NIFM as feature extractors, we propose Inner PNN (IPNN), Kernel PNN (KPNN),
and Product-network In Network (PIN).
- We study practical issue in training and generalization. We conduct more offline experiments and an online A/B test.

The [demo](https://github.com/Atomu2014/product-nets) of the conference paper contains IPNN and KPNN and is easier to run.
This repository implements all the proposed models as well as baseline models in tensorflow, with multi-gpu support, and distributed training support.
