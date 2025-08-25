#!/bin/sh
#python -m experiments.active_learning.gnn -c "crps"
#python -m experiments.active_learning.gnn -c "log"
python -m experiments.active_learning.gnn -c "var"
python -m experiments.active_learning.gnn -c "kernel"