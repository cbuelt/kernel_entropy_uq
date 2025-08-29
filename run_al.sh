#!/bin/sh
python -m experiments.active_learning.drn -loss "crps"
python -m experiments.active_learning.drn -loss "log"
python -m experiments.active_learning.drn -loss "se"
python -m experiments.active_learning.drn -loss "kernel"
python -m experiments.active_learning.kernel_tuning