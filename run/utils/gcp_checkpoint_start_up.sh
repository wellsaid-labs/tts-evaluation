# Learn more: https://stackoverflow.com/questions/58733368/gcp-metadata-access-from-startup-script

getMetadata() {
  curl -fs http://metadata/computeMetadata/v1/instance/attributes/$1 \
    -H "Metadata-Flavor: Google"
}

TRAIN_SCRIPT_PATH=$(getMetadata train-script-path)
STARTUP_SCRIPT_USER=$(getMetadata startup-script-user)

if [ -f /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT ]; then
  echo 'Restarting from the latest checkpoint...'
  runuser -l $STARTUP_SCRIPT_USER -c "screen -dmL bash -c \
      'cd /opt/wellsaid-labs/Text-to-Speech;
      . venv/bin/activate;
      PYTHONPATH=. python $TRAIN_SCRIPT_PATH resume;'"
fi

echo "Setting environment variables..."
echo "TRAIN_SCRIPT_PATH=$TRAIN_SCRIPT_PATH" >>/etc/environment

