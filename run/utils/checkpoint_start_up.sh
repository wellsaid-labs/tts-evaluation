if [ -f /opt/wellsaid-labs/AUTO_START_FROM_CHECKPOINT ]; then
  echo 'Restarting from the latest checkpoint...'
  runuser -l $VM_USER -c "screen -dmL bash -c \
      'cd /opt/wellsaid-labs/Text-to-Speech;
      . venv/bin/activate;
      PYTHONPATH=. python $TRAIN_SCRIPT_PATH resume;'"
fi

echo "Setting environment variables..."
echo "TRAIN_SCRIPT_PATH=$TRAIN_SCRIPT_PATH" >>/etc/environment

