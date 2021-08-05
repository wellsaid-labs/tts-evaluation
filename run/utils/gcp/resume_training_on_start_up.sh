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

TRAIN_SCRIPT_PATH=$(get_attribute_value train-script-path)
STARTUP_SCRIPT_USER=$(get_attribute_value startup-script-user)

if [ -f /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT ]; then
  echo 'Restarting from the latest checkpoint...'
  runuser -l $STARTUP_SCRIPT_USER -c "screen -dmL bash -c \
      'cd /opt/wellsaid-labs/Text-to-Speech;
      . venv/bin/activate;
      PYTHONPATH=. python $TRAIN_SCRIPT_PATH resume;'"
fi

echo "Setting environment variables..."
echo "TRAIN_SCRIPT_PATH=$TRAIN_SCRIPT_PATH" >>/etc/environment
