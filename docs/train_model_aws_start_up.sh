echo 'Running startup script...'

OLD_HOST_NAME=$(hostname)
echo "Updating hostname from $OLD_HOST_NAME to $VM_NAME..."
hostnamectl set-hostname $VM_NAME

echo 'Finished startup script! :)'
