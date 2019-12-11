# Train a Model with Amazon Web Services (AWS)

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

1. You'll first need programatic access to our AWS account.

   1. Ask a team member to create you an AWS user.

   1. Follow [this guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey)
      to create an access key.

   1. Set these bash variables with your new access key...

      ```bash
      AWS_ACCESS_KEY_ID='your-aws-access-key-id'
      AWS_SECRET_ACCESS_KEY='your-aws-secret-access-key'
      ```

      For `awscli` to progrmatically access your AWS account run this..

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

## From your local repository

1. Setup your environment variables...

   ```bash
   VM_REGION='your-vm-region' # EXAMPLE: us-east-1
   VM_MACHINE_TYPE=g4dn.12xlarge
   VM_IMAGE_ID=ami-0b98d7f73c7d1bb71
   VM_IMAGE_USER=ubuntu  # The default user name for the above image.
   VM_NAME=$USER"_your_instance_name" # EXAMPLE: michaelp_baseline
   KEY_PAIR_NAME=$USER"_amazon_web_services" # EXAMPLE: michaelp_amazon_web_services
   TRAIN_SCRIPT_PATH='src/bin/train/spectrogram_model/__main__.py'
   ```

   If your training the signal model, you'll want instead...

   ```bash
   VM_MACHINE_TYPE=p3.16xlarge
   TRAIN_SCRIPT_PATH='src/bin/train/signal_model/__main__.py'
   ```

   Related Resources:

   - Learn more about the available instance types, [here](https://aws.amazon.com/ec2/instance-types/).
   - Learn more about the above "ami-0b98d7f73c7d1bb71" image, [here](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-Base-AMI-Ubu/B07Y3VDBNS).
   - During any point in this process, you may want to image the disk so that you don't have to start
     from scratch every time. In order to do so, please follow the instructions
     [here](https://docs.aws.amazon.com/cli/latest/reference/ec2/create-image.html).
   - Learn more about the available GPU instances for each region,
     [here](https://aws.amazon.com/ec2/pricing/on-demand/).
   - Get a list of all AWS regions, [here](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html).

1. Upload your SSH key to the AWS region you plan to train in.

   ```bash
   aws --region=$VM_REGION ec2 import-key-pair \
        --key-name=$AWS_KEY_PAIR_NAME \
        --public-key-material=file://$HOME/.ssh/$AWS_KEY_PAIR_NAME.pub
   ```

1. Create a security group to restrict incoming and outgoing VM traffic to only allow SSH, like
   so...

   ```bash
   SECURITY_GROUP_NAME=only-ssh
   aws --region=$VM_REGION ec2 create-security-group --group-name $SECURITY_GROUP_NAME \
      --description "Only allow SSH connections"
   aws --region=$VM_REGION ec2 authorize-security-group-ingress \
      --group-name $SECURITY_GROUP_NAME \
      --protocol tcp \
      --port 22 \
      --cidr 0.0.0.0/0
   ```

   If the security group already exists, you can skip this step.

1. Create an instance startup script that'll run every time this instance boots...

   ```bash
   USER_DATA=$(cat docs/train_model_aws_start_up.sh)
   USER_DATA=${USER_DATA//'$VM_NAME'/\'$VM_NAME\'}
   USER_DATA=${USER_DATA//'$VM_USER'/\'$VM_USER\'}
   USER_DATA=${USER_DATA//'$VM_REGION'/\'$VM_REGION\'}
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

   Learn more about this approach
   [here](https://aws.amazon.com/premiumsupport/knowledge-center/execute-user-data-ec2/).

1. Create an EC2 instance for training.

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
   ```

   ```bash
   SPOT_REQUEST_ID=$(aws --region=$REGION ec2 request-spot-instances \
      --type "persistent" \
      --launch-specification file:///tmp/launch_specification.json \
      --instance-interruption-behavior "stop" | \
      jq '.SpotInstanceRequests[0].SpotInstanceRequestId' | xargs)
   aws --region=$VM_REGION ec2 create-tags \
       --resources $SPOT_REQUEST_ID \
       --tags Key=Name,Value=$VM_NAME
   ```

   By default, AWS will keep this instance alive for the next seven days.

1. When the instance comes online, you'll be able to fetch a URL for SSH...

   ```bash
   VM_ID=$(aws --region=$VM_REGION ec2 describe-instances \
      --filters Name=tag:Name,Values=$VM_NAME \
      --query 'Reservations[0].Instances[0].InstanceId' \
      --output text)
   VM_STATUS=$(aws --region=$VM_REGION ec2 describe-instance-status \
      --instance-id $VM_ID | jq ".InstanceStatuses[0].InstanceState.Name" | xargs)
   if [ "$VM_STATUS" == "running" ] ; then
      VM_PUBLIC_DNS=$(aws --region=$VM_REGION ec2 describe-instances \
          --filters Name=tag:Name,Values=$VM_NAME \
          --query 'Reservations[0].Instances[0].PublicDnsName' \
          --output text)
   else
      echo "ERROR: The instance '$VM_NAME' isn't running, yet."
   fi
   ```

1. Finally, you can ssh into the instance...

   ```bash
   ssh -i ~/.ssh/$AWS_KEY_PAIR_NAME \
      -o UserKnownHostsFile=/dev/null \
      -o StrictHostKeyChecking=no \
      $VM_IMAGE_USER@$VM_PUBLIC_DNS
   ```

   You'll want to run this command multiple times until it works.

### On the VM instance

1. Install these packages, like so...

   ```bash
   sudo apt-get update
   sudo apt-get install python3-venv sox ffmpeg ninja-build -y
   ```

   If you get an error after running `sudo apt-get update`, wait a minute or so and try again.

1. Create a directory for our software.

   ```bash
   sudo chmod -R a+rwx /opt
   mkdir /opt/wellsaid-labs
   cd /opt/wellsaid-labs
   ```

### From your local repository

1. Use `src.bin.cloud.lsyncd` to live sync your repository to your VM instance:

   ```bash
   python3 -m src.bin.cloud.lsyncd --public_dns $VM_PUBLIC_DNS \
                                 --identity_file ~/.ssh/$AWS_KEY_PAIR_NAME \
                                 --source $(pwd) \
                                 --destination /opt/wellsaid-labs/Text-to-Speech \
                                 --user $VM_IMAGE_USER
   ```

1. When prompted, give your local sudo password for your laptop.

   Keep this process running on your local machine until you've started training, it'll
   allow you to make any hot-fixes to your code in case you run into an error.

### On the VM instance

1. Start a screen session:

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

1. Pick or create a comet project [here](https://www.comet.ml/wellsaid-labs). Afterwards set
   this variable...

   ```bash
   COMET_PROJECT="your-comet-project"
   ```

1. Train your model ...

   ```bash
   EXPERIMENT_NAME='your-experiment-name'
   ```

   ```bash
   pkill -9 python; \
   nvidia-smi; \
   PYTHONPATH=. python $TRAIN_SCRIPT_PATH \
       --project_name $COMET_PROJECT \
       --name $EXPERIMENT_NAME
   ```

   Note: You can include an optional `--spectrogram_model_checkpoint` argument is optional
   (for example, see [here](TRAIN_TTS_MODEL_GCP.md#on-the-vm-instance)).

   We run `pkill -9 python` to kill any leftover processes from previous runs and `nvidia-smi`
   to ensure the GPU has no running processes.

1. Detach from your screen session by typing `Ctrl-A` then `D`.

1. To ensure that your VM restarts from the latest checkpoint, in the case of an interruption,
   run this command:

   ```bash
   touch /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT
   ```

1. You can exit your VM with the `exit` command.

### From your local repository

1. Kill your `lsyncd` process by typing `Ctrl-C`.

1. Once training has finished ...

   ... stop your VM

   ```bash
   aws --region=$VM_REGION ec2 stop-instances --instance-ids $VM_ID
   ```

   ... or delete your VM

   ```bash
   aws --region=$VM_REGION ec2 terminate-instances --instance-ids $VM_ID
   ```
