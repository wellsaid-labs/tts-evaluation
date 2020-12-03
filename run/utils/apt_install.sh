# Install Python 3.8
sudo apt install python3.8 -y
sudo rm /usr/bin/python3
sudo ln -s python3.8 /usr/bin/python3
python3 -V
sudo apt-get install python3.8-venv python3.8-dev -y

# Install other dependencies
sudo apt-get install sox ffmpeg espeak gcc libsox-fmt-mp3 -y
