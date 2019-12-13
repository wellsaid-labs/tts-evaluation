# Learn more here:
# https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
# https://github.com/eddelbuettel/mkl4deb
# Add the intel APT Repository
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
apt-get update

# NOTE: This is the most recent update based on:
# https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
# Install Intel® Math Kernel Library (Intel® MKL)
apt-get install --no-install-recommends intel-mkl-2019.4-070 -y

# Configure dynamic linker run-time bindings
echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf
echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf
ldconfig

rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
