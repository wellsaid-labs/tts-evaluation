# NOTE: This error can be resolved with...
# ```
# The following packages have unmet dependencies:
#  linux-modules-nvidia-460-gcp-edge : Depends: linux-modules-nvidia-460-5.4.0-1041-gcp (= 5.4.0-1041.44~18.04.1+1) but it is not going to be installed
#                                      Depends: nvidia-kernel-common-460 (<= 460.56-1) but 460.67-0ubuntu0~0.18.04.1 is to be installed
# E: Unable to correct problems, you have held broken packages.
# ```
# Run these commands, and one of the "solutions" worked for me.
# ```
# $ sudo apt-get install aptitude
# $ sudo aptitude install linux-modules-nvidia-460-gcp-edge
# ```
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update -y
sudo apt-get install ubuntu-drivers-common -y
sudo ubuntu-drivers autoinstall

# Reload drivers
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# Double-check drivers are working
nvidia-smi
