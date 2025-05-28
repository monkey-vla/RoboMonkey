# RoboMonkey

This is the code for RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models.


## Contents
 * [**Installation**](#installation)
 * [**Reproduce SIMPLER Benchmark**](#Reproduce-SIMPLER-Benchmark)


## Installation

First, clone this monorepo. Use `--recurse-submodules` to clone all submodules as well. 

```bash
git clone --recurse-submodules https://github.com/monkey-vla/RoboMonkey.git
```
Alternatively, if you already clone the repo: `git submodule update --init --recursive`.

Install RoboMonkey reward model in a separate conda environment to prevent interference with the base model. The reward model communicates with the base model and the simulator via HTTP endpoints. We provide bash scripts that set up the environments.

```bash
cd RoboMonkey
bash scripts/env.sh
bash scripts/env_reward.sh
```


## Reproduce SIMPLER Benchmark

### Base Policy
Setup the environment for the base policy OpenVLA. We use SGLang for efficient inference.
```bash
bash scripts/env_sglang.sh
```

Spin up the base policy server in a tmux terminal to keep it running.
```bash
tmux
conda activate sglang-vla
cd sglang-vla/serve_vla
CUDA_VISIBLE_DEVICES=1 python vla_server.py
```

### Reward Model

Install the reward model with instructions from the [**Installation**](#installation) section.

Spin up the reward model server.
```bash
tmux
conda activate robomonkey-reward
cd RewardModel/RLHF
bash infer_server.sh
```

### Install SIMPLER benchmark

```bash
bash scripts/env_simpler.sh
tmux
conda activate simpler_env && \
    export PRISMATIC_DATA_ROOT=. && \
    export PYTHONPATH=. && \
    cd openvla-mini
xvfb-run --auto-servernum -s "-screen 0 640x480x24" python experiments/robot/simpler/run_simpler_eval.py \
    --task_suite_name simpler_widowx \
    --center_crop True \
    --use_wandb False \
    --robomonkey_batch_size 5
```

## Run Benchmark without Reward Model

Change the above command-line argument of `robomonkey_batch_size` to `--robomonkey_batch_size 1` to run the benchmark without the reward model.

## Acknowledgements

We thank the authors of [OpenVLA](https://github.com/openvla/openvla), [SGLang](https://github.com/sgl-project/sglang), and [OpenVLA-mini](https://github.com/Stanford-ILIAD/openvla-mini) for their contributions to the open-source community.