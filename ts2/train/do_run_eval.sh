#!/bin/bash

set -x
set -e

for i in  {0..3}; do
    SLURM_ARRAY_TASK_ID=$i python main.py -c=config/chengjia/eval_scsrh7.yaml
done
