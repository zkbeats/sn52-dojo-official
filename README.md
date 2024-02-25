<div align="center">
  <h1 style="border-bottom: 0">Tensorplex Reward Modelling Subnet</h1>
</div>

<div align="center">
  <a href="https://discord.gg/p8tg26HFQQ">
    <img src="https://img.shields.io/discord/1186416652955430932.svg" alt="Discord">
  </a>
  <a href="https://opensource.org/license/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</div>

<br>

<div align="center">
  <a href="https://www.tensorplex.ai/">Website</a>
  ·
  <a href="https://tensorplex.gitbook.io/tensorplex-docs/tensorplex-rlhf">Docs</a>
  ·
  <a href="https://huggingface.co/tensorplex-labs">HuggingFace</a>
  ·  
  <a href="#getting-started">Getting Started</a>
  ·
  <a href="https://twitter.com/TensorplexLabs">Twitter</a>
</div>

---

<details>
<summary>Table of Contents</summary>

- [Introduction](#introduction)
- [Features](#features)
- [Use Cases](#use-cases)
- [Minimum Requirements](#minimum-requirements)
  - [Miner](#miner)
  - [Validator](#validator)
- [Getting Started](#getting-started)
  - [Mining](#mining)
  - [Validating](#validating)
- [Mechanisms](#mechanisms)
- [License](#license)

</details>

---

# Introduction
Reinforcement Learning Human Feedback (RLHF) is based on reward model. The reward model is trained with the preference dataset. These preference datasets and reward models are built by large private companies like OpenAI, Anthropic, Google, Meta, etc. While they release aligned LLMs, they don't release the Reward Model. Hence, we don't have a say in determining which responses are good or bad. Why should big corporations have the power to decide what is good or bad? Let's decentralize that power, by having a decentralized, consensus-based Reward Model.

Introducing the RLHF subnet, where participants in this subnet are given the power to decide what is good or bad, and these results are collectively evaluated using our consensus mechanism. We also introduce the first of its kind by connecting our subnet layer to an external application layer (Amazon Mechanical Turk) to allow the subnet to access a globally available and 24/7 workforce to provide high quality human intelligence task feedback.

# Features
- Open Source Reward Models
- Novel Human Feedback Loop
- Multi-Modality (Coming soon...)

# Use Cases
Our RLHF subnet provides decentralised, consensus-based Reward Modelling that allows applications to be built on top of it. One example use case is fine-tuning of Large Language Models (LLMs), where a model being fine-tuned may query our API to score multiple LLM outputs with respect to a prompt. This may also be used to compare quality of responses among different LLMs.

# Minimum Requirements
- Python 3.11 and above
- [Bittensor](https://github.com/opentensor/bittensor#install)

## Miner
- 8 cores
- 32 GB RAM
- 150 GB SSD

## Validator
- 8 cores
- 32 GB RAM
- 1 TB SSD

# Getting Started
To get started as a miner or validator, these are the common steps both a miner and validator have to go through. 


1. setup python environment

```bash
cd repo_name/
# create new virtual env
python -m venv env_name 
# activate our virtual env
source env_name/bin/activate 
# verify python environment version
python --version
```

2. install requirements.txt

```bash
source env_name/bin/activate 
pip install -r requirements.txt
```

3. setup bittensor wallet
```bash
# create new coldkey
btcli wallet new_coldkey --wallet.name your_coldkey_name
# create new hotkey
btcli wallet new_hotkey --wallet.name your_coldkey_name
# you will be prompted with the following...
Enter hotkey name (default):
```

4. prepare .env file

Copy the `.env.example` into a separate `.env` file. These are supposed to contain certain API keys required for your miner/validator to function as expected.

<font size="6">**Remember, never commit this .env file!**</font>

## Mining

When providing scoring for prompt & completions, there are currently 3 methods:
- using a HuggingFace model
- using a LLM like mistralai/Mixtral-8x7B-Instruct-v0.1 via [TogetherAI's Inference endpoints](https://docs.together.ai/docs/inference-models).
- human feedback via [Amazon Mechanical Turk](https://www.mturk.com/)

Note that when using APIs, make sure to check the supported models. Currently TogetherAI and OpenAI are supported. For more providers, please send us a request via Discord or help contribute to our repository! See the [guidelines for contributing](./contrib/CONTRIBUTING.md).

Note that in order to Amazon Mechanical Turk, there are additional steps to take, see the [Amazon MTURK setup guide](#amazon-mechanical-turk-setup-guide).

To start the miner, run one of the following command(s):
```bash
# using huggingface model
python main_miner.py --netuid 1 --subtensor.network finney --wallet.name your_coldkey --wallet.hotkey your_hotkey --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "hf_model" --model_name "OpenAssistant/reward-model-deberta-v3-large-v2"
# using llm api 
python main_miner.py --netuid 1 --subtensor.network finney --wallet.name your_coldkey --wallet.hotkey your_hotkey --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "llm_api" --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1"
# using aws mturk, --model_name is not used
python main_miner.py --netuid 1 --subtensor.network finney --wallet.name your_coldkey --wallet.hotkey your_hotkey --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "aws_mturk"
```


### Amazon Mechanical Turk Setup Guide
<details>

<summary>Details (TODO)</summary>

- sign up for aws account
- sign up for aws mturk requester account
- link the two
- setup IAM policy to get aws access key id and secret, attach mturk policies
- setup aws sns
- setup aws lambda function
- inside of lambda function, configure environment variables to be

```bash
# set up TARGET_URL so that MTurk can send requests back to your miner
TARGET_URL="https://miner_machine_ip:port/api/human_feedback/callback"
# production
MTURK_ENDPOINT_URL="https://mturk-requester.us-east-1.amazonaws.com"
# sandbox
MTURK_ENDPOINT_URL="https://mturk-requester-sandbox.us-east-1.amazonaws.com"
```
</details>

## Validating
To start the validator, run the following command
```bash
python main_validator.py --netuid 1 --subtensor.network finney --wallet.name your_coldkey --wallet.hotkey your_hotkey --logging.debug --axon.port 9500 --neuron.type validator
```

# Mechanisms

As a miner, you will be evaluated on the classification accuracy on a set of human preference datasets, and this will act as a multiplier towards your consensus score, thus to gain more emissions as a miner you will need to perform better in terms of classification accuracy on some human preference datasets. This is done to incentivise miners to create better reward models.

# Building a reward model
TODO


# License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
