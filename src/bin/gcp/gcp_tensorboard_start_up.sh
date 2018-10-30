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

read -p "Start tensorboards for archived experiments? (Y/n) " RESP
if [ "$RESP" = "Y" ]; then
  run 'wave_net_tensorboard' \
    'tensorboard --logdir=experiments/wave_net --port=6006 --window_title=WaveNet'

  run 'tacotron_2_tensorboard' \
      'tensorboard --logdir=experiments/tacotron_2 --port=6007 --window_title=Tacotron-2'

  run 'wave_rnn_tensorboard' \
      'tensorboard --logdir=experiments/wave_rnn --port=6009 --window_title=WaveRNN'
fi

run 'signal_model_sync_tensorboard' \
    "tensorboard --logdir=sync/signal_model/ --port=6008 --window_title='Signal Model Sync'"

run 'spectrogram_model_sync_tensorboard' \
    "tensorboard --logdir=sync/spectrogram_model/ --port=6010 --window_title='Spectrogram Model Sync'"

echo 'Updating SSH files...'
gcloud compute config-ssh

# LEARN MORE: ``rsync`` slash on source
# http://qdosmsq.dunbar-it.co.uk/blog/2013/02/rsync-to-slash-or-not-to-slash/
run 'signal_model_sync_python' \
     "python3 -m src.bin.gcp.periodic_rsync \
        --destination ~/WellSaidLabs/sync/signal_model/  \
        --source ~/WellSaidLabs/experiments/signal_model/ \
        --all;"

run 'spectrogram_model_sync_python' \
     "python3 -m src.bin.gcp.periodic_rsync \
        --destination ~/WellSaidLabs/sync/spectrogram_model/  \
        --source ~/WellSaidLabs/experiments/spectrogram_model/ \
        --all;"

run 'backup' \
    "cd ..; watch -n 60 rsync -av WellSaidLabs/experiments/ backup/"

echo 'Done!'
