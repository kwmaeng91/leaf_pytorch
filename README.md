# LEAF-PyTorch: PyTorch Port of the CMU LEAF Codebase

This is a PyTorch port of the CMU LEAF codebase, which was written in Tensorflow.
Currently, only CelebA, FEMNIST, and Reddit datasets are supported (I may or may not implement others in the future).
Not all features from the original code has been fully ported.
Additionally, I decoupled the client-side learning rate (-lr) and the server-side learning rate (--server-lr).
This code was written as a hobby and was never extensively tested (use at your own risk).

The original README file can be found under README_LEAF.md

## Example Executions (not optimal; increasing num_round and clients_per_round improves accuracy)
  * **CelebA:** ```python3 main.py -dataset celeba -model cnn -lr 0.01 --batch-size 10 --num-epoch 5 --num-round 100 --eval-every 5 --clients-per-round 10```
  * **FEMNIST:** ```python3 main.py -dataset femnist -model cnn -lr 0.01 --batch-size 10 --num-epoch 5 --num-round 100 --eval-every 5 --clients-per-round 10```
  * **Reddit:** ```python3 main.py -dataset reddit -model stacked_lstm --eval-every 5 --num-rounds 100 --clients-per-round 10 --batch-size 5 -lr 2.0```

## Original LEAF Resources

  * **Homepage:** [leaf.cmu.edu](https://leaf.cmu.edu)
  * **Original github (in Tensorflow):** [Original Github](https://github.com/TalwalkarLab/leaf)
  * **Paper:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)

## Modifications from the Original LEAF Codebase

The code is largely unmodified from the original LEAF codebase, except for the following files:

1. These files have been modified.
  * client.py
  * main.py
  * server.py
  * utils/args.py
  * main.py

2. These files have been newly added.
  * celeba/cnn.py 
  * femnist/cnn.py
  * reddit/stacked_lstm.py

## LICENSE

The project is licensed under BSD-2. Please see the attached LICENSE.md.
