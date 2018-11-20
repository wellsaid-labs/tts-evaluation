settings {
  logfile = "/var/log/lsyncd.log", -- Sets the log file
  statusFile = "/var/log/lsyncd-status.log", -- Sets the status log file,
  nodaemon = true
}
sync {
  default.rsyncssh, -- Uses the rsyncssh defaults, learn more here:
                    -- https://github.com/axkibe/lsyncd/blob/master/default-rsyncssh.lua
  source="{source}", -- Your source directory to watch
  host="{user}@{ip}", -- The remote host (use hostname or IP)
  targetdir="{destination}", -- The target dir on remote host, keep in mind this is absolute path
  delay = .2,
  delete = false,
  rsync = {
      binary = "/usr/local/bin/rsync", -- OSX does not have updated version of rsync,
                                       -- install via: `brew install rsync`
      rsh = "ssh -i {home}/.ssh/google_compute_engine -o UserKnownHostsFile={home}/.ssh/known_hosts",
      verbose = true,
  },
  exclude = {
      "notebooks/QA Speech Datasets/",
      "notebooks/Speech Dataset Script Generation/",
      "data/**",
      "experiments/**",
      "build/**",
      "__pycache__**",
      ".git**",
      "htmlcov/**",
      "coverage/**",
      "docs/**",
      "*.py*.py" -- Odd temporary files
  }
}