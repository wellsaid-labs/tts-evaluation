# Train a Model with Amazon Web Services (AWS) Spot Instances

This markdown will walk you through the steps required to train a model on a AWS virtual
machine.

## Prerequisites

1. Setup your local development environment by following [these instructions](LOCAL_SETUP.md).

1. Ask a team member to grant you access to our AWS account.

1. Install these dependencies:

   ```bash
   brew install rsync lsyncd jq
   python -m pip install awscli
   ```

1. You'll need programatic access to our AWS account.

   1. Ask a team member to create you an AWS IAM user and login to it.

   1. Follow [this guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey)
      to create an access key.

   1. Set these bash variables with your new access key...

      ```bash
      AWS_ACCESS_KEY_ID='your-aws-access-key-id'
      AWS_SECRET_ACCESS_KEY='your-aws-secret-access-key'
      ```

      and run this..

      ```bash
      mkdir ~/.aws
      echo "[default]
      aws_access_key_id=$AWS_ACCESS_KEY_ID
      aws_secret_access_key=$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials
      ```

1. You'll need an SSH key to use with your AWS account, you can create one like so...

   ```bash
   AWS_KEY_PAIR_NAME=$USER"_amazon_web_services"
   ssh-keygen -t rsa -C $AWS_KEY_PAIR_NAME -f ~/.ssh/$AWS_KEY_PAIR_NAME -N ""
   ```

## Train a Model with Amazon Web Services (AWS)

### From your local repository

1. Setup your environment variables...

   ```bash
   # Learn more about the AWS Deep Learning Base AMI here:
   # https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-Base-AMI-Ubu/B07Y3VDBNS
   VM_IMAGE_ID=ami-0b98d7f73c7d1bb71
   VM_IMAGE_USER=ubuntu

   AWS_KEY_PAIR_NAME=$USER"_amazon_web_services"
   ```

   Set these variables for training the spectrogram model...

   ```bash
   VM_MACHINE_TYPE=g4dn.12xlarge
   TRAIN_SCRIPT_PATH='src/bin/train/spectrogram_model/__main__.py'
   ```

   Set these variables for training the signal model...

   ```bash
   VM_MACHINE_TYPE=p3.16xlarge
   TRAIN_SCRIPT_PATH='src/bin/train/signal_model/__main__.py'
   ```

   Finally, run this script to determine the region with the least spot instance interruptions...

   ```bash
   python3 docs/train_model_aws_best_region.py --machine_type=$VM_MACHINE_TYPE
   ```

   Then, set these environment variables...

   ```bash
   export AWS_DEFAULT_REGION='your-vm-region' # EXAMPLE: us-east-1
   VM_NAME=$USER"_your-instance-name" # EXAMPLE: michaelp_baseline
   ```

   Related Resources:

   - Learn more about our benchmarks for the available machine types,
     [here](./TRAIN_MODEL_AWS_BENCHMARKS.md).
   - Learn more about the available instance types,
     [here](https://aws.amazon.com/ec2/instance-types/).
   - During any point in this process, you may want to image (AMI) the disk so that you don't have
     to start from scratch every time. In order to do so, please follow the instructions
     [here](https://docs.aws.amazon.com/cli/latest/reference/ec2/create-image.html).

     Note that every AMI is created with a corresponding snapshot. If your curious about the AMI
     progress, you can find that information by viewing the corresponding AMI snapshot. Recently,
     this process took 1.5 hours.

   - Learn more about the available GPU instances for each region,
     [here](https://aws.amazon.com/ec2/pricing/on-demand/).
   - Get a list of all AWS regions,
     [here](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html).

1. Upload your SSH key to the AWS region you plan to train in.

   ```bash
   aws ec2 import-key-pair --key-name=$AWS_KEY_PAIR_NAME \
        --public-key-material=file://$HOME/.ssh/$AWS_KEY_PAIR_NAME.pub
   ```

   Skip this step if the key pair already exists.

1. Create a security group that only allows SSH traffic to the VM, like so...

   ```bash
   SECURITY_GROUP_NAME=only-ssh
   aws ec2 create-security-group --group-name $SECURITY_GROUP_NAME \
      --description "Only allow SSH connections"
   aws ec2 authorize-security-group-ingress --group-name $SECURITY_GROUP_NAME --protocol tcp \
      --port 22 --cidr 0.0.0.0/0
   ```

   Skip this step if the security group already exists.

1. Setup a startup script for the instance,
   [similar to this](https://aws.amazon.com/premiumsupport/knowledge-center/execute-user-data-ec2/),
   like so...

   ```bash
   USER_DATA=$(cat docs/train_model_aws_spot_start_up.sh)
   USER_DATA=${USER_DATA//'$VM_NAME'/\'$VM_NAME\'}
   USER_DATA=${USER_DATA//'$VM_USER'/\'$VM_IMAGE_USER\'}
   USER_DATA=${USER_DATA//'$VM_REGION'/\'$AWS_DEFAULT_REGION\'}
   USER_DATA=${USER_DATA//'$TRAIN_SCRIPT_PATH'/\'$TRAIN_SCRIPT_PATH\'}
   USER_DATA=${USER_DATA//'$AWS_CREDENTIALS'/\'$(cat ~/.aws/credentials)\'}
   USER_DATA="Content-Type: multipart/mixed; boundary=\"//\"
   MIME-Version: 1.0

   --//
   Content-Type: text/cloud-config; charset=\"us-ascii\"
   MIME-Version: 1.0
   Content-Transfer-Encoding: 7bit
   Content-Disposition: attachment; filename=\"cloud-config.txt\"

   #cloud-config
   cloud_final_modules:
   - [scripts-user, always]

   --//
   Content-Type: text/x-shellscript; charset=\"us-ascii\"
   MIME-Version: 1.0
   Content-Transfer-Encoding: 7bit
   Content-Disposition: attachment; filename=\"userdata.txt\"

   #!/bin/bash
   $USER_DATA
   --//"
   USER_DATA=$(echo "$USER_DATA" | base64)
   ```

   The output of the startup script will be saved on the VM here: `/var/log/cloud-init-output.log`

1. Create an EC2 instance for training...

   ```bash
   echo '{
      "SecurityGroups": ["only-ssh"],
      "BlockDeviceMappings": [
        {
          "DeviceName": "/dev/sda1",
          "Ebs": {
            "DeleteOnTermination": true,
            "VolumeSize": 512,
            "VolumeType": "gp2"
          }
        }
      ],
      "ImageId": "'$VM_IMAGE_ID'",
      "InstanceType": "'$VM_MACHINE_TYPE'",
      "KeyName": "'$AWS_KEY_PAIR_NAME'",
      "UserData": "'$USER_DATA'"
   }' > /tmp/launch_specification.json
   SPOT_REQUEST_ID=$(aws ec2 request-spot-instances --type "persistent" \
      --launch-specification file:///tmp/launch_specification.json \
      --instance-interruption-behavior "stop" | \
      jq '.SpotInstanceRequests[0].SpotInstanceRequestId' | xargs)
   aws ec2 create-tags --resources $SPOT_REQUEST_ID --tags Key=Name,Value=$VM_NAME
   ```

   This instance will stay online for seven days or until you cancel the spot request.

1. Wait for the instance status to be 'running'...

   ```bash
   VM_STATUS=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$VM_NAME \
      --query 'Reservations[0].Instances[0].State.Name' --output text)
   echo "The status of VM '$VM_NAME' is '$VM_STATUS'."
   ```

1. SSH into the instance...

   ```bash
   VM_PUBLIC_DNS=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$VM_NAME \
     --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
   ssh -i ~/.ssh/$AWS_KEY_PAIR_NAME -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
     $VM_IMAGE_USER@$VM_PUBLIC_DNS
   ```

   Continue to run this command until it succeeds.

### On the VM instance

1. Install these packages, like so...

   ```bash
   sudo apt-get update
   sudo apt-get install python3-venv sox ffmpeg ninja-build -y
   ```

   If you get a `dkpg` error, wait a minute or so and try again.

1. Create a directory for our software...

   ```bash
   sudo chmod -R a+rwx /opt
   mkdir /opt/wellsaid-labs
   cd /opt/wellsaid-labs
   ```

### From your local repository

1. In a new terminal window, setup your environment variables again...

   ```bash
   export AWS_DEFAULT_REGION='your-vm-region' # EXAMPLE: us-east-1
   VM_NAME=$USER"_your-instance-name" # EXAMPLE: michaelp_baseline
   ```

1. Use `src.bin.cloud.lsyncd` to live sync your repository to your VM instance:

   ```bash
   VM_PUBLIC_DNS=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$VM_NAME \
      --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
   VM_IMAGE_USER=ubuntu
   python3 -m src.bin.cloud.lsyncd --public_dns $VM_PUBLIC_DNS \
                                 --identity_file ~/.ssh/$USER"_amazon_web_services" \
                                 --source $(pwd) \
                                 --destination /opt/wellsaid-labs/Text-to-Speech \
                                 --user $VM_IMAGE_USER
   ```

   When prompted, enter your sudo password.

1. Leave this processing running until you've started training. This will allow you to make any
   hot-fixes to your code in case you run into an error.

### On the VM instance

1. Start a `screen` session:

   ```bash
   screen
   ```

1. Navigate to the repository, activate a virtual environment, and install package requirements...

   ```bash
   cd /opt/wellsaid-labs/Text-to-Speech

   # Note: You will always want to be in an active venv whenever you want to work with python.
   python3 -m venv venv
   . venv/bin/activate

   python -m pip install wheel
   python -m pip install -r requirements.txt --upgrade

   sudo bash src/bin/install_mkl.sh
   ```

1. Set a flag to restart training if the instance is rebooted...

   ```bash
   touch /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT
   ```

1. For [comet](https://www.comet.ml/wellsaid-labs), name your experiment and pick a project...

   ```bash
   COMET_PROJECT='your-comet-project'
   EXPERIMENT_NAME='Your experiment name'
   ```

1. Start training...

   ```bash
   TRAIN_SCRIPT_PATH='your-choosen-training-script-from-earlier'

   # Kill any leftover processes from other runs...
   pkill -9 python; sleep 5s; nvidia-smi; \
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH --project_name $COMET_PROJECT --name "$EXPERIMENT_NAME";
   ```

   💡 TIP: You may want to include the optional `--spectrogram_model_checkpoint` argument.

1. Detach from your screen session by typing `Ctrl-A` then `D`.

1. You can now exit your VM with the `exit` command.

### From your local repository

1. Kill your `lsyncd` process by typing `Ctrl-C`.

## Post Training Clean Up

### From your local repository

1. Setup your environment variables again...

   ```bash
   export AWS_DEFAULT_REGION='your-vm-region' # EXAMPLE: us-east-1
   VM_NAME=$USER"_your-instance-name" # EXAMPLE: michaelp_baseline

   SPOT_REQUEST_ID=$(aws ec2 describe-spot-instance-requests \qaq
      --filters Name=tag:Name,Values=$VM_NAME \
      --query 'SpotInstanceRequests[0].SpotInstanceRequestId' --output text)
   VM_ID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$VM_NAME \
                --query 'Reservations[0].Instances[0].InstanceId' --output text)
   ```

1. Cancel your spot instance request...

   ```bash
   aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $SPOT_REQUEST_ID
   ```

1. Stop your instance...

   ```bash
   aws ec2 stop-instances --instance-ids $VM_ID
   ```

   or delete your instance...

   ```bash
   aws ec2 terminate-instances --instance-ids $VM_ID
   ```
