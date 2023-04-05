function get_metadata_value() {
  curl --retry 5 \
    -s \
    -f \
    -H "Metadata-Flavor: Google" \
    "http://metadata/computeMetadata/v1/$1"
}

function get_attribute_value() {
  get_metadata_value "instance/attributes/$1"
}

for line in $(cat /etc/environment); do export $line; done

if [ -z "$TRAIN_SCRIPT_PATH" ]; then
  TRAIN_SCRIPT_PATH=$(get_attribute_value train-script-path)
  echo "Setting environment variables..."
  echo "TRAIN_SCRIPT_PATH=$TRAIN_SCRIPT_PATH" >>/etc/environment
fi

if [ -f /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT ]; then
  # NOTE: The drivers may rarely sometimes fail to initialize.
  echo 'Reloading drivers, just in case...'
  sudo rmmod nvidia_uvm
  sudo modprobe nvidia_uvm

  echo 'Restarting from the latest checkpoint...'
  STARTUP_SCRIPT_USER=$(get_attribute_value startup-script-user)
  runuser -l $STARTUP_SCRIPT_USER -c "screen -dmL bash -c \
      'cd /opt/wellsaid-labs/Text-to-Speech;
      . venv/bin/activate;
      PYTHONPATH=. python $TRAIN_SCRIPT_PATH resume;'"
fi
