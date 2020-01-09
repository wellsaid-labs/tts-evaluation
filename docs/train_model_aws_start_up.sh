echo 'Running startup script...'

OLD_HOST_NAME=$(hostname)
echo "Updating hostname from $OLD_HOST_NAME to $VM_NAME..."
hostnamectl set-hostname $VM_NAME

echo "Setting environment variables..."
echo "TRAIN_SCRIPT_PATH=$TRAIN_SCRIPT_PATH" >>/etc/environment

echo 'Finished startup script! :)'
