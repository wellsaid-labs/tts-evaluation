echo 'Running startup script...'

export AWS_ACCESS_KEY_ID='{access_key_id}'
export AWS_SECRET_ACCESS_KEY='{secret_access_key}'
export AWS_REGION='{vm_region}'
VM_NAME='{vm_name}'

VM_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
SPOT_REQUEST_ID=$(aws --region $AWS_REGION ec2 describe-instances --instance-ids $VM_ID \
  --query 'Reservations[0].Instances[0].SpotInstanceRequestId' --output text)

OLD_HOST_NAME=$(hostname)
echo "Updating hostname from $OLD_HOST_NAME to $VM_NAME..."
hostnamectl set-hostname $VM_NAME

{more_bash_script}

# NOTE: Don't tag instance until the start-up script is done.
echo "Transfering tags from spot request $SPOT_REQUEST_ID to instance $VM_ID..."
TAGS=$(aws --region $AWS_REGION ec2 describe-spot-instance-requests \
  --spot-instance-request-ids $SPOT_REQUEST_ID --query 'SpotInstanceRequests[0].Tags')
aws --region $AWS_REGION ec2 create-tags --resources $VM_ID --tags "$TAGS"

echo 'Finished startup script! :)'
