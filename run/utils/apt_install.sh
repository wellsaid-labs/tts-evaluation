sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update -y

# Install Python 3.8. Learn more about this approach:
# https://linuxconfig.org/how-to-change-from-default-to-alternative-python-version-on-debian-linux
sudo apt install python3.8 -y
sudo update-alternatives --install /usr/bin/python3 python3 $(readlink -f /usr/bin/python3) 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --set python3 /usr/bin/python3.8
sudo apt-get install python3.8-venv python3.8-dev -y

# Install other dependencies
sudo apt-get install sox ffmpeg espeak gcc g++ libsox-fmt-mp3 -y
