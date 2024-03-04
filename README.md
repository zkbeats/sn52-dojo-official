<div align="center">
  <h1 style="border-bottom: 0">Dojo Subnet</h1>
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
    - [Amazon Mechanical Turk Setup Guide](#amazon-mechanical-turk-setup-guide)
      - [MTurk Setup](#mturk-setup)
      - [AWS SNS](#aws-sns)
      - [AWS **Lambda**](#aws-lambda)
  - [Validating](#validating)
- [Types of Requests](#types-of-requests)
- [Mechanisms](#mechanisms)
  - [Human Feedback Loop](#human-feedback-loop)
  - [Validator Scoring](#validator-scoring)
  - [Consensus \& Classification Accuracy](#consensus--classification-accuracy)
- [Building a reward model](#building-a-reward-model)
- [Roadmap](#roadmap)
- [License](#license)

</details>





---

# Introduction
Reinforcement Learning Human Feedback (RLHF) is based on reward models that are trained with preference datasets. These preference datasets and reward models are built by large private companies like OpenAI, Anthropic, Google, Meta, etc. While they release aligned LLMs, they don't release the Reward Model. Hence, we don't have a say in determining which responses are good or bad. _Why should big corporations have the power to decide what is good or bad?_ __Let's decentralize that power, by having a decentralized, consensus-based Reward Model.__

Introducing the Dojo, a subnet where participants are given the power to decide what is good or bad, and these results are collectively evaluated using our consensus mechanism. We also introduce the first of its kind by connecting our subnet layer to an external application layer (Amazon Mechanical Turk) to allow the subnet to access a globally available and 24/7 workforce to provide high quality human intelligence task feedback.

# Features
🤗 Open Source Reward Models<br>
👨‍💻 Human Feedback Loop<br>
📝 Text-based Reward Models<br>
🧧 Low minimum cost to participate<br>

# Use Cases
Our Dojo subnet provides decentralised, consensus-based Reward Modelling that allows applications to be built on top of it. One example use case is fine-tuning of Large Language Models (LLMs), where a model being fine-tuned may query our API to score multiple LLM outputs with respect to a prompt. This may also be used to compare quality of responses among different LLMs.

Other use cases include text-based and image-based evaluations like on other subnets. In each subnet, they typically need to evaluate text based prompt & responses, so each subnet has to write their own versions of these evaluations and our subnet could abstract these evaluations away and serve all of them, where the scalability is only limited by the modalities (text, image, etc.) that are being supported on this subnet.

# Minimum Requirements
- Python 3.11 and above
- PM2

## Miner
- 4 cores
- 16 GB RAM
- 150GB SSD

## Validator
- 8 cores
- 32 GB RAM
- 1 TB SSD

# Getting Started
To get started as a miner or validator, these are the common steps both a miner and validator have to go through. 


1. setup python environment

```bash
cd dojo-subnet/
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

3. install pm2 

```bash
apt-get update
apt-get install -y ca-certificates curl gnupg
mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
NODE_MAJOR=20
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
apt-get update
apt-get install nodejs -y

npm install -g pm2
pm2 install pm2-logrotate
```

4. setup bittensor wallet
```bash
# create new coldkey
btcli wallet new_coldkey --wallet.name your_coldkey_name
# create new hotkey
btcli wallet new_hotkey --wallet.name your_coldkey_name

# you will be prompted with the following...
Enter hotkey name (default):
```

5. prepare .env file

Copy the `.env.miner.example` or `.env.validator.example` file into a separate `.env` file. These are supposed to contain certain API keys required for your miner/validator to function as expected.

<font size="6">**Remember, never commit this .env file!**</font>

## Mining

To start the miner, run one of the following command(s):
```bash
# using huggingface model
python main_miner.py --netuid 1 --subtensor.network finney --wallet.name your_coldkey --wallet.hotkey your_hotkey --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "hf_model" --model_name "OpenAssistant/reward-model-deberta-v3-large-v2"
# using llm api 
python main_miner.py --netuid 1 --subtensor.network finney --wallet.name your_coldkey --wallet.hotkey your_hotkey --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "llm_api" --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1"
# using aws mturk, --model_name is not used
python main_miner.py --netuid 1 --subtensor.network finney --wallet.name your_coldkey --wallet.hotkey your_hotkey --logging.debug --axon.port 9599 --neuron.type miner --scoring_method "aws_mturk"
```

When providing scoring for prompt & completions, there are currently 3 methods:
- using a HuggingFace model
- using a LLM like mistralai/Mixtral-8x7B-Instruct-v0.1 via [TogetherAI's Inference endpoints](https://docs.together.ai/docs/inference-models).
- human feedback via [Amazon Mechanical Turk](https://www.mturk.com/)

If you wish to earn a higher miner trust and incentives, you will need to use different scoring methods such as those with human feedback (Amazon Mechanical Turk), and/or have better classification accuracy on human preference datasets.

Note that in order to Amazon Mechanical Turk, there are additional steps to take, see the [Amazon MTurk setup guide](#amazon-mechanical-turk-setup-guide).

<blockquote class="callout callout_default">
<p>❗NOTE❗ when using APIs, make sure to check the supported models. Currently TogetherAI and OpenAI are supported. For more providers, please send us a request via Discord or help contribute to our repository! 

See the <a href="./contrib/CONTRIBUTING.md">guidelines for contributing</a>.
</p>
</blockquote>

<br>


### Amazon Mechanical Turk Setup Guide

This guide will go through how to setup Amazon Mechanical Turk so that we can send tasks to humans that will contribute to scoring requests.

<details>

<summary><strong>Click me for details!</strong></summary>


#### MTurk Setup

1. Sign up for AWS account at https://portal.aws.amazon.com/billing/signup#/start/email
2. Sign up for AWS MTurk Requester account at https://requester.mturk.com/begin_aws_signin, which will link the two accounts together.
3. Setup an IAM policy to get AWS Access Key ID and Secret
- Go to https://console.aws.amazon.com/console
- Use the search bar to search for "IAM" and click on the first result.
- Click on "Users" in the side bar and then go to "Create User".
<img src="./assets/iam/iam1.jpg">

- Specify this IAM user's name, in this case I am using "lambda_mturk_role"
<img src="./assets/iam/iam6.jpg">

- Set permissions of the IAM user. Click on "Attach policies directly" and search for "AmazonMechanicalTurkFullAccess" using the search bar and select it.
<img src="./assets/iam/iam7.jpg">

- Go on to the next page and click on "Create User".
<img src="./assets/iam/iam9.jpg">

- Now that the user is created, head back to the Users section, and click on the name of the user previously created.
<img src="./assets/iam/iam8.jpg">

- Go to the "Security Credentials" tab, then click on "Create access key".
<img src="./assets/iam/iam2.jpg">

- Select "Application running outside AWS" and set a description to something memorable for future reference. In this case I am using "lambda_mturk_role for subnet" as the description. Then click on "Create access key".
<img src="./assets/iam/iam5.jpg">
<img src="./assets/iam/iam4.jpg">
- Now get the access key credentials and copy it over to your `.env` file!
```bash
AWS_ACCESS_KEY_ID=
AWS_SECRET_KEY=
```
<img src="./assets/iam/iam3.jpg">

<br>

We will now need to setup AWS Simple Notification Service (SNS) and AWS Lambda, where AWS SNS is for subscribing to MTurk events and AWS Lambda executes code that will forward the responses to our miner.

#### AWS SNS

1. Search for "Simple Notification Service" using the search bar.

2. From the dashboard, navigate to "Topics", and click on "Create Topic".

<img src="./assets/sns/sns1.jpg">
<img src="./assets/sns/sns2.jpg">

3. Fill out the topic name, in this case I have assigned "mturk_sns_topic", you must also choose "Standard" type so that AWS Lambda may be used with SNS. 

<img src="./assets/sns/sns4.jpg">

4. Afterwards, click on "Create Topic" all the way at the bottom. 

<img src="./assets/sns/sns3.jpg">

5. Navigate to "Edit" and scroll down to "Access Policy". You will need the "Resource" field which looks like `arn:aws:sns:us-east-1:<your_aws_id>:mturk_sns_topic` so copy it first.

<img src="./assets/sns/sns6.jpg">
<img src="./assets/sns/sns7.jpg">

6. Replace the access policy with the following:

```json
{
  "Version": "2008-10-17",
  "Id": "__default_policy_ID",
  "Statement": [
    {
      "Sid": "__default_statement_ID",
      "Effect": "Allow",
      "Principal": {
        "Service": "mturk-requester.amazonaws.com"
      },
      "Action": "SNS:Publish",
      "Resource": "arn:aws:sns:us-east-1:<your_aws_id>:test_topic"
    }
  ]
}
```

<br>
Almost there!

#### AWS **Lambda**

1. Use the search bar to navigate to AWS Lambda and click on "Create function"
<img src="./assets/lambda/lambda1.jpg">

2. Setup the function name, in this case I have used "mturk_event_forwarder" and set your runtime to Python >= 3.11, and click on "Create function"
<img src="./assets/lambda/lambda2.jpg">
3. Upon successful creation you will now see the following message.
<img src="./assets/lambda/lambda3.jpg">
4. Navigate to "Code" tab and click on the "lambda function" section of the code editor.
<img src="./assets/lambda/lambda4.jpg">
5. Grab the code from  `commons/human_feedback/aws_lambda_function.py` and paste it into the code editor. 
6. You will now see a "Changed not deployed" bubble. Click on "Deploy" so that your changes are saved!
<img src="./assets/lambda/lambda5.jpg">

7. Navigate to the "Configuration" tab at the top, and go to "Environment Variables", and click on "Edit".
<img src="./assets/lambda/lambda6.jpg">

8. You will need to add the following 2 environment variables.
`TARGET_URL` and `MTURK_ENDPOINT_URL`.
<img src="./assets/lambda/lambda7.jpg">

- TARGET_URL can be found from your miner's machine, go to the command line and run `curl ifconfig.me` -> this will allow you to get your external IP, combine this with the `api.port` inside of `config.py` to construct the TARGET_URL.
```bash
TARGET_URL="https://<external_ip>:<api.port>/api/human_feedback/callback
```
- MTURK_ENDPOINT_URL refers to either a production or sandbox requester URL of AWS MTurk.

```bash
# production
MTURK_ENDPOINT_URL="https://mturk-requester.us-east-1.amazonaws.com"
# sandbox
MTURK_ENDPOINT_URL="https://mturk-requester-sandbox.us-east-1.amazonaws.com"
```

<font size="6"> **You are highly encouraged to try out requests on the sandbox environment before trying it out on the production environment!**</font> 

To do this you will need to link your AWS account to the request sandbox at https://requestersandbox.mturk.com, after that try running `scripts/test_aws_mturk.py`

9. Go back to SNS, and set SNS to be a trigger for the Lambda function. Click on "Add Trigger" and select "SNS" from the dropdown. Search for the SNS topic name.
<!-- # TODO remove the screenshot with ARN inside, and run bfg repo cleaner -->
<img src="./assets/lambda/lambda8.jpg">
<img src="./assets/lambda/lambda9.jpg">
<img src="./assets/lambda/lambda10.jpg">
<img src="./assets/lambda/lambda11.jpg">


10. When publishing MTurk tasks in production, you will definitely hit an error. This is because the monthly spending limit by sending a request to them at https://support.aws.amazon.com/#/contacts/aws-mechanical-turk, asking to raise the monthly spending limit on your account.


<img src="./assets/mturk_prod_error.jpg">

<br>
<blockquote class="callout callout_default" theme="🥳">
  <h3>🎉 AWS MTURK SETUP DONE 🎉</h3>
  <p>Your AWS MTurk setup is finally done, you just need to wait 1-3 days for approval to use MTurk as a scoring method to earn higher incentives! Head back to the <a href="#mining">Mining</a> section to start your miner and start earning daily rewards!</p>
</blockquote>

</details>


## Validating

1. Visit https://huggingface.co/settings/tokens to generate a new token if you haven't done so already, and place this in your `.env` file from earlier.

To start the validator, run the following command
```bash
python main_validator.py --netuid 1 --subtensor.network finney --wallet.name your_coldkey --wallet.hotkey your_hotkey --logging.debug --axon.port 9500 --neuron.type validator
```

# Types of Requests
- `RankingRequest`
  - This represents a request from validators, where miners are expected to use tools at their disposal to provide scores to each completion, which is a response to an original prompt.
- `RankingResult`
  - This represents the consensus score across all queried miners, referenced by the completion ID.
- `MTurkResponse`
  - This represents a completed `RankingRequest` sent from the miner to Amazon MTurk workers, and the payload is a dictionary of completion ID to score.

# Mechanisms

## Human Feedback Loop
There may be delays in terms of responses from Amazon MTurk workers, so validators have to serve axons as well in order to receive forwarded MTurk responses from miners once they are completed. This breaks the traditional way of only validators calling miners.

## Validator Scoring
Due to the need to provide ample time for human feedback, the deadline for each `RankingRequest` is currently set to 8 hours. Only after the deadline has been passed, validators will score all participants responses. This deadline is generous and provides plenty of time for the feedback loop.

## Consensus & Classification Accuracy
As a miner, you will be evaluated on the classification accuracy on a set of human preference datasets, and this will act as a multiplier towards your consensus score, thus to gain more emissions as a miner you will need to perform better in terms of classification accuracy on some human preference datasets. This is done to incentivise miners to create better reward models because scoring initially uses Spearman correlation. Thus if someone has a better reward model but steers away from consensus, they may be penalised. We want to incentivise people to build better reward models. These classification scores get reset every 24 hours.


# Building a reward model
Coming soon...

# Roadmap
- 🌏 Release of API service
- 🖼️ Image-based Reward Models
- 🤖 Integrations with other feedback interfaces using basic.ai, scale, argilla
- 🤗 Release of dataset from consensus

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
