echo 'Running startup script...'

echo 'Adding credentials to AWS instance...'
mkdir ~/.aws
echo $AWS_CREDENTIALS > ~/.aws/credentials

VM_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
SPOT_REQUEST_ID=$(aws --region $VM_REGION ec2 describe-instances --instance-ids $VM_ID \
                      --query 'Reservations[0].Instances[0].SpotInstanceRequestId' --output text)

echo "Transfering tags from spot request $SPOT_REQUEST_ID to instance $VM_ID..."
TAGS=$(aws --region $VM_REGION ec2 describe-spot-instance-requests \
          --spot-instance-request-ids $SPOT_REQUEST_ID --query 'SpotInstanceRequests[0].Tags')
aws --region $VM_REGION ec2 create-tags --resources $VM_ID --tags "$TAGS"

OLD_HOST_NAME=$(hostname)
echo "Updating hostname from $OLD_HOST_NAME to $VM_NAME..."
hostnamectl set-hostname $VM_NAME

if [ -f /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT ] ; then
  echo 'Restarting from the latest checkpoint...'
  runuser -l $VM_USER -c "screen -dmL bash -c \
      'cd /opt/wellsaid-labs/Text-to-Speech;
      . venv/bin/activate;
      PYTHONPATH=. python $TRAIN_SCRIPT_PATH --checkpoint;'"
fi

echo 'Finished startup script! :)'
