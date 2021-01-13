settings {
  logfile = "/var/log/lsyncd.log", -- Sets the log file
  statusFile = "/var/log/lsyncd-status.log", -- Sets the status log file,
  nodaemon = true
}
sync {
  default.rsyncssh, -- Uses the rsyncssh defaults, learn more here:
                    -- https://github.com/axkibe/lsyncd/blob/master/default-rsyncssh.lua
  -- Learn more about this path, as related to Mac OS:
  -- https://github.com/axkibe/lsyncd/issues/587
  source="/System/Volumes/Data/{source}", -- Your source directory to watch
  host="{user}@{public_dns}", -- The remote host (use hostname or IP)
  targetdir="{destination}", -- The target dir on remote host, keep in mind this is absolute path
  delay = .2,
  delete = true,
  rsync = {
      binary = "/usr/local/bin/rsync", -- OSX does not have updated version of rsync,
                                       -- install via: `brew install rsync`
      -- NOTE: We disable SSH host key checking because GCP frequently changes it's remote
      -- host identification.
      rsh = "ssh -i {identity_file} -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no",
      verbose = true,
      compress = true,
      -- Learn more:
      -- https://stackoverflow.com/questions/667992/rsync-error-failed-to-set-times-on-foo-bar-operation-not-permitted
      -- "Same on Linux (Debian Squeeze in my case)... If I'm not the owner of the target directory,
      -- rsync gives the "failed to set times" error message. (Having write permission on the
      -- directory is not enough.) "
      _extra = { "--omit-dir-times" }
  },
  exclude = {
      ".ipynb_checkpoints/**", -- Jupyter files
      ".mypy_cache/**", -- Mypy files
      ".DS_Store", -- Macbook files
      "__pycache__**",
      "htmlcov/**", -- PyTest Coverage files
      ".coverage*", -- PyTest Coverage files
      "coverage/**", -- PyTest Coverage files
      "git_diff.patch", -- Comet ML patch
      "*.py*.py", -- Odd temporary files
      -- Various compressed formats are typically slow to sync
      "*.zip",
      "*.tar",
      "*.tgz",
      "*.tar.gz",
      "*.gzip",
      "tmp/**",
      "venv/**",
      "node_modules/**",
      "disk/**",
  }
}
