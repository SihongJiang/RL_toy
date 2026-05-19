#!/usr/bin/env bash

set -e

cd /home/zhangsihong/Projects/rl_toy

source /home/zhangsihong/miniforge3/etc/profile.d/conda.sh
conda activate /data/zsh/conda_envs/rltoy

python src/run_ppo_cartpole.py