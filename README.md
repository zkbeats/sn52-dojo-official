<p align="center">
    <font size="20">Tensorplex RLHF Subnet</font> 
</p>

<div align="center">
  <a href="https://discord.gg/p8tg26HFQQ">
    <img src="https://img.shields.io/discord/1186416652955430932.svg" alt="Discord">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</div>

<div align="center">
  <a href="https://www.tensorplex.ai/">Website</a>
  ·
  <a href="https://tensorplex.gitbook.io/tensorplex-docs/tensorplex-rlhf">Docs</a>
  ·
  <a href="https://huggingface.co/tensorplex-labs">HuggingFace</a>
  ·  
  <a href="#installation">Installation</a>
  ·
  <a href="https://twitter.com/TensorplexLabs">Twitter</a>
</div>


<!-- toc -->

---
- [Quickstarter template](#quickstarter-template)
- [Introduction](#introduction)
  - [Example](#example)
- [Installation](#installation)
  - [Before you proceed](#before-you-proceed)
  - [Install](#install)
- [Writing your own incentive mechanism](#writing-your-own-incentive-mechanism)
- [License](#license)

---
# Introduction
<TODO>

# Getting started as a miner or validator
To get started, see the `.env.example`, copy and paste this file into a separate `.env` file! <b>Remember, never commit this .env file anywhere!</b>

## Mining

When providing scoring for prompt & completions, there are currently 3 methods, i.e. using a huggingface model, LLM based API (e.g. mistralai/Mixtral-8x7B-Instruct-v0.1) or human feedback via Amazon Mechanical Turk.

Each of them have different combinations of command line arguments available:
- For huggingface models, use the `--scoring_method='hf_model'` option, and `--model_name='OpenAssistant/reward-model-deberta-v3-large-v2'`
- For LLM models via an API, use the `--scoring_method='llm_api'` option, and `--model_name='mistralai/Mixtral-8x7B-Instruct-v0.1'`
- For using AWS Mechanical Turk, use the `--scoring_method='aws_mturk'` option, in this case --model_name will not be used.

To start the miner, run the following command
```bash
# using huggingface model
python main_miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name subnet_miner --wallet.hotkey test01 --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "hf_model" --model_name "OpenAssistant/reward-model-deberta-v3-large-v2"
# using llm api 
python main_miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name subnet_miner --wallet.hotkey test01 --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "llm_api" --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1"
# using aws mturk
python main_miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name subnet_miner --wallet.hotkey test01 --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "aws_mturk"
```

You will be evaluated on the classification accuracy on a set of human preference datasets, and this will act as a multiplier towards your consensus score, thus to gain more emissions as a miner you will need to perform better in terms of classification accuracy on some human preference datasets.

## Validating
To start the validator, run the following command
```bash
python main_validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name subnet_validator --wallet.hotkey test01 --logging.debug --axon.port 9500 --neuron.type validator --neuron.epoch_length 100
```



# Consensus Mechanism
We use a mix of spearman correlation...

- Spearman correlation is used in order to calculate the consensus scores among miners when providing scores to prompt & completions.
- 






## Introduction

**IMPORTANT**: If you are new to Bittensor subnets, read this section before proceeding to [Installation](#installation) section. 

The Bittensor blockchain hosts multiple self-contained incentive mechanisms called **subnets**. Subnets are playing fields in which:
- Subnet miners who produce value, and
- Subnet validators who produce consensus

determine together the proper distribution of TAO for the purpose of incentivizing the creation of value, i.e., generating digital commodities, such as intelligence or data. 

Each subnet consists of:
- Subnet miners and subnet validators.
- A protocol using which the subnet miners and subnet validators interact with one another. This protocol is part of the incentive mechanism.
- The Bittensor API using which the subnet miners and subnet validators interact with Bittensor's onchain consensus engine [Yuma Consensus](https://bittensor.com/documentation/validating/yuma-consensus). The Yuma Consensus is designed to drive these actors: subnet validators and subnet miners, into agreement on who is creating value and what that value is worth. 

This starter template is split into three primary files. To write your own incentive mechanism, you should edit these files. These files are:
1. `template/protocol.py`: Contains the definition of the protocol used by subnet miners and subnet validators.
2. `neurons/miner.py`: Script that defines the subnet miner's behavior, i.e., how the subnet miner responds to requests from subnet validators.
3. `neurons/validator.py`: This script defines the subnet validator's behavior, i.e., how the subnet validator requests information from the subnet miners and determines the scores.

### Example

The Bittensor Subnet 1 for Text Prompting is built using this template. See [Bittensor Text-Prompting](https://github.com/opentensor/text-prompting) for how to configure the files and how to add monitoring and telemetry and support multiple miner types. Also see this Subnet 1 in action on [Taostats](https://taostats.io/subnets/netuid-1/) explorer.

---

## Installation

### Before you proceed
Before you proceed with the installation of the subnet, note the following: 

- Use these instructions to run your subnet locally for your development and testing, or on Bittensor testnet or on Bittensor mainnet. 
- **IMPORTANT**: We **strongly recommend** that you first run your subnet locally and complete your development and testing before running the subnet on Bittensor testnet. Furthermore, make sure that you next run your subnet on Bittensor testnet before running it on the Bittensor mainnet.
- You can run your subnet either as a subnet owner, or as a subnet validator or as a subnet miner. 
- **IMPORTANT:** Make sure you are aware of the minimum compute requirements for your subnet. See the [Minimum compute YAML configuration](./min_compute.yml).
- Note that installation instructions differ based on your situation: For example, installing for local development and testing will require a few additional steps compared to installing for testnet. Similarly, installation instructions differ for a subnet owner vs a validator or a miner. 

### Install

- **Running locally**: Follow the step-by-step instructions described in this section: [Running Subnet Locally](./docs/running_on_staging.md).
- **Running on Bittensor testnet**: Follow the step-by-step instructions described in this section: [Running on the Test Network](./docs/running_on_testnet.md).
- **Running on Bittensor mainnet**: Follow the step-by-step instructions described in this section: [Running on the Main Network](./docs/running_on_mainnet.md).


## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
