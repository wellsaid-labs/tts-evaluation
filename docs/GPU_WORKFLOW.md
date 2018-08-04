
# GPU Workflow

This section is intended to describe one workflow with remote GPU machines on OSX.

## Prerequisites

Install the latest version of `lsyncd` and `rsync` by running:
```
brew install rsync
brew install lsyncd
```

Create a folder locally `/path/to/synced_folder` to be synced remotely. Populate it with directories to be synced.

Create a file `/path/to/synced_folder/lrsyncssh.conf.lua` with the contents below:
```
settings {
  logfile = "/var/log/lsyncd.log", -- Sets the log file
  statusFile = "/var/log/lsyncd-status.log" -- Sets the status log file
}
sync {
  default.rsyncssh, -- Uses the rsyncssh defaults, learn more here: https://github.com/axkibe/lsyncd/blob/master/default-rsyncssh.lua
  source="/path/to/synced_folder/", -- Your source directory to watch
  host="username@host", -- The remote host (use hostname or IP)
  targetdir="/your/path/to/Tacotron-2/", -- The target dir on remote host, keep in mind this is absolute path
  delay = .2,
  delete = false,
  rsync = {
      binary = "/usr/local/bin/rsync", -- OSX does not have updated version of rsync, install via: `brew install rsync`
      rsh = "ssh -i /your/path/to/.ssh/id_rsa -o UserKnownHostsFile=/your/path/to/.ssh/known_hosts",
      verbose = true,
  },
  exclude = { -- Tacotron-2 specific rules
      "data/**",
      "experiments/**",
      "build/**",
      "__pycache__**",
      ".git**"
  }
}
```

`lrsyncssh.conf.lua` will by used by `lsyncd` to sync `/path/to/synced_folder` remotely. Remember
to edit fields like ``rsh``, ``host`` and ``targetdir``.

## Workflow

### Begin Sync

On your local machine, sync `/path/to/synced_folder/`.
```
cd /path/to/synced_folder/
sudo lsyncd lrsyncssh.conf.lua -nodaemon
```

### SSH

SSH from your local machine to the GPU machine.
```
ssh $USER@$GPU
```

### Clean up
Remember to kill your lsyncd daemon after your done.

## Appendix

### Sync GPU to CPU

After you are done training, you may need to transfer files from the GPU to your local machine.
We recommend `scp`.
```
scp $USER@$GPU:/path/to/foo /path/to/foo
```

Periodic sync:
```
while true; do scp -r $USER@$GPU:/path/to/foo /path/to/foo; sleep 960; done
```

### Multiple GPUs

You may find that you need to run more than one experiment at a time. We've provided a tool to sync
tensorboard events from multiple GCP servers to one server, like so:

    gcloud compute ssh tensorboard --zone=us-us-west1-b

    gcloud compute config-ssh # Find any new servers

    python3 src/bin/sync_tensorboard.py -s vocalx.us-west1-b flowvo.us-west1-b  -p ~/Tacotron-2/experiments/signal_model

Now, periodically, tensorboard files from ``vocalx.us-west1-b`` and ``flowvo.us-west1-b`` will be
sync'd to ``~/Tacotron-2/sync/`` on GCP server ``tensorboard``.
