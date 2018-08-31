#!/bin/bash
#
# Bash script to setup ``tensorboard`` GCP instance.

run () {
  # Runs a detacted screen session unless its already running.
  # Arguments:
  #    $1: Name of the screen session
  #    $2+: Commands to run in screen session.
  if ! screen -list | grep -q "\b$1\b"; then
      echo "Starting screen session $1..."
      screen -dmS "$1" bash -c "${@:2}"
  fi
}

run 'wave_net_tensorboard' \
    'tensorboard --logdir=experiments/wave_net --port=6006 --window_title=WaveNet'

run 'tacotron_2_tensorboard' \
    'tensorboard --logdir=experiments/tacotron_2 --port=6007 --window_title=Tacotron-2'

run 'wave_rnn_tensorboard' \
    'tensorboard --logdir=experiments/wave_rnn --port=6009 --window_title=WaveRNN'

run 'signal_model_sync_tensorboard' \
    "tensorboard --logdir=sync/signal_model/ --port=6008 --window_title='Signal Model Sync'"

run 'feature_model_sync_tensorboard' \
    "tensorboard --logdir=sync/feature_model/ --port=6010 --window_title='Feature Model Sync'"

echo 'Updating SSH files...'
gcloud compute config-ssh

run 'signal_model_sync_python' \
     "python3 src.bin.periodic_rsync \
        --destination ~/WellSaid-Labs-Text-To-Speech/sync/signal_model/  \
        --source ~/WellSaid-Labs-Text-To-Speech/experiments/signal_model \
        --all;"

run 'feature_model_sync_python' \
     "python3 src.bin.periodic_rsync \
        --destination ~/WellSaid-Labs-Text-To-Speech/sync/feature_model/  \
        --source ~/WellSaid-Labs-Text-To-Speech/experiments/feature_model \
        --all;"

run 'backup' \
    "cd ..; watch -n 60 rsync -av WellSaid-Labs-Text-To-Speech/experiments/ backup/"

echo 'Done!'
