# Local Setup

This document summarizes how to setup this repository locally for development.

## 1. Install System Dependencies

Please install the below requirements:

- [Docker](https://www.docker.com/products/docker-desktop)
- [brew](https://brew.sh/)
- [Visual Studio Code](https://code.visualstudio.com/docs/setup/mac). In addition to the
  initial steps, please run through the section called "Launching from the command line".

Afterwards, please install other system dependencies, like so:

```zsh
brew install git
brew install python@3.8
brew install sox # Audio processing
brew install ffmpeg # Audio processing
brew install espeak # Speech synthesizer
brew install rsync lsyncd # File transfer
brew install jq # Bash JSON parser
setopt interactivecomments # Bash style comments
```

## 2. Clone the Repository

Using `git` clone the repository onto your system:

```zsh
git clone --depth=10 --no-single-branch --recurse-submodules https://github.com/wellsaid-labs/Text-to-Speech.git
```

The reason for using `--depth=10 --no-single-branch` is to reduce the size of the `.git` directory.
On May 20th, these flags would reduce the repository size by 80%
(i.e. 58 megabytes to 12 megabytes).

## 3. Install Python Dependencies

Install Python dependencies, like so:

```zsh
# Start a virtual environment, learn more:
# https://realpython.com/python-virtual-environments-a-primer/
python3 -m venv venv
. venv/bin/activate

# Install Python dependencies
python -m pip install pip --upgrade
python -m pip install -r requirements.txt --upgrade
```

## 4. Comet

This repository requires [Comet](https://www.comet.ml) for visualization, and you'll need to ask
a team member to create you an account.

With your new account, you'll need to create a `.comet.config` file in this repositories root
level directory with these configurations:

```
[comet]
workspace=wellsaid-labs
api_key=<api_key>
rest_api_key=<rest_api_key>
disable_auto_logging=1
```

Learn more about where to find the `api_key` and `rest_api_key` configurations on
[this web page](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration).

## 5. Google Cloud

This repository relies on GCP, and you'll need to ask team member to get access to our GCP projects,
"Voice Research" and "Voice Service".

Afterwards, install [Google Cloud SDK](https://cloud.google.com/sdk/) with these
[installation scripts](https://cloud.google.com/sdk/docs/downloads-interactive) and authorize
Google Cloud SDK tools, like so:

```zsh
gcloud init

# NOTE: Authorize Google's command-line-interface, learn more:
# https://cloud.google.com/sdk/gcloud/reference/auth/login
gcloud auth login

# NOTE: Authorize Google client libraries, learn more:
# https://cloud.google.com/sdk/gcloud/reference/auth/application-default
gcloud auth application-default login
```

## 6. Amazon Web Services

You'll need to ask team member to get access to our AWS account via an IAM user.

### Configure the AWS CLI

1. Follow
   [this guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey)
   to create an access key.

1. Set these bash variables with your new access key...

   ```zsh
   AWS_ACCESS_KEY_ID='your-aws-access-key-id'
   AWS_SECRET_ACCESS_KEY='your-aws-secret-access-key'
   ```

1. And run this..

   ```zsh
   mkdir ~/.aws
   echo "[default]
   aws_access_key_id=$AWS_ACCESS_KEY_ID
   aws_secret_access_key=$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials
   ```

### Configure AWS SSH

You'll need an SSH key to use with your AWS account, you can create one like so...

```zsh
AWS_KEY_PAIR_NAME=$USER"_amazon_web_services"
ssh-keygen -t rsa -C $AWS_KEY_PAIR_NAME -f ~/.ssh/$AWS_KEY_PAIR_NAME -N ""
```

## Good job! 🎉
