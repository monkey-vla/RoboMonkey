# RoboMonkey

<!-- [![arXiv]()]() -->
[![Project Website](https://img.shields.io/badge/Project-Website-blue?style=for-the-badge)](https://robomonkey-vla.github.io/)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/robomonkey-vla/RewardModel)
[![License](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](LICENSE)
 

This is the code for RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models.


## Contents
 * [**Installation**](#installation)
 * [**Getting Started**](#getting-started)
 * [**Reproduce SIMPLER Benchmark**](#Reproduce-SIMPLER-Benchmark)


## Installation

First, clone this monorepo. If you wish to use the reward model on its own, just clone the repo without the submodules. If you wish to reproduce the SIMPLER benchmarks, use `--recurse-submodules` to clone all submodules as well. 

```bash
git clone --recurse-submodules https://github.com/monkey-vla/RoboMonkey.git
```
Alternatively, to clone the submodules after you have cloned this repo: `git submodule update --init --recursive`.

The script installs RoboMonkey reward model in a separate conda environment to prevent environment interference with the base VLA model. The reward model communicates with the base model and the simulator via HTTP endpoints. We provide bash scripts that set up the environments. The scripts have been verified to run on 2x RTX4090 with [this](https://hub.docker.com/layers/nvidia/cuda/11.8.0-cudnn8-devel-ubuntu20.04/images/sha256-0b25e1f1c6f596a6c92b04cb825714be41b4dc8323ba71205dbae8b11bfa672c) image.

```bash
cd RoboMonkey
bash scripts/env.sh         # sets up conda, apt packages, and git LFS
bash scripts/env_reward.sh  # install the robomonkey-reward environment
```

## Getting Started

Spin up the reward model server in a separate terminal.

```bash
conda activate robomonkey-reward
cd RewardModel/RLHF
python infer_server.py
```

You can run `python test/test.py` to test if the reward model is set up correctly.

## Reproduce SIMPLER Benchmark

### Base Policy
Setup the environment for the base policy OpenVLA. We use SGLang for efficient inference.
```bash
bash scripts/env_sglang.sh
```

Spin up the base policy server in a separate terminal to keep it running.
```bash
conda activate sglang-vla
cd sglang-vla/serve_vla
CUDA_VISIBLE_DEVICES=1 python vla_server.py
```

### Reward Model

Install the reward model with instructions from the [**Installation**](#installation) section.

### SIMPLER benchmark

First install the SIMPLER benchmark.

```bash
bash scripts/env_simpler.sh
```

Run the benchmark with:
```bash
conda activate simpler_env
export PRISMATIC_DATA_ROOT=. && \
    export PYTHONPATH=. && \
    cd openvla-mini
xvfb-run --auto-servernum -s "-screen 0 640x480x24" python experiments/robot/simpler/run_simpler_eval.py \
    --task_suite_name simpler_widowx \
    --robomonkey_batch_size 5
```

## Run Benchmark without Reward Model

Change the above command-line argument of `robomonkey_batch_size` to `--robomonkey_batch_size 1` to run the benchmark without the reward model.

## Acknowledgements

We thank the authors of [OpenVLA](https://github.com/openvla/openvla), [SGLang](https://github.com/sgl-project/sglang), [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and [OpenVLA-mini](https://github.com/Stanford-ILIAD/openvla-mini) for their contributions to the open-source community.