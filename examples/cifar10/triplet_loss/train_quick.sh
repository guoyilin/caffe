#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=1 $TOOLS/caffe train \
  --solver=examples/cifar10/triplet_loss/cifar10_quick_solver.prototxt \
  2>&1 | tee examples/cifar10/triplet_loss/log

# reduce learning rate by factor of 10 after 8 epochs
#GLOG_logtostderr=1 $TOOLS/caffe train \
#  --solver=examples/cifar10/triplet_loss/cifar10_quick_solver_lr1.prototxt \
#  --snapshot=examples/cifar10/triplet_loss/cifar10_quick_iter_4000.solverstate.h5 \
#  2>&1 | tee examples/cifar10/triplet_loss/log2
