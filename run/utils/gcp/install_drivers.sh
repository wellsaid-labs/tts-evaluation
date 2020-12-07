sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update -y
sudo apt-get install ubuntu-drivers-common -y
sudo ubuntu-drivers autoinstall

# Reload drivers, in order to avoid a reboot
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# Double-check drivers are working
nvidia-smi
