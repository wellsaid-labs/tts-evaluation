sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update -y

# Install Python 3.8
sudo apt install python3.8 -y
sudo apt-get install python3.8-venv python3.8-dev -y

# Install other dependencies
sudo apt-get install sox ffmpeg espeak gcc g++ libsox-fmt-mp3 -y
