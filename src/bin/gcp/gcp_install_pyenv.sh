# Install dependancies
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev

# Install pyenv installer
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# Add to startup script
cat <<EOT >> ~/.bashrc
export PATH="/home/michaelp/.pyenv/bin:$PATH"
eval "\$(pyenv init -)"
eval "\$(pyenv virtualenv-init -)"
EOT
source ~/.bashrc

# Allow PyEnv python installs to use ``.local`` without ``sudo``
sudo chown -R $(whoami) ~/.local/
