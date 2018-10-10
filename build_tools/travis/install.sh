#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# Exit immediately if a command exits with a non-zero status.
set -e

echo 'List files from cached directories'
if [ -d $HOME/download ]; then
    echo 'download:'
    ls $HOME/download
fi
if [ -d $HOME/.cache/pip ]; then
    echo 'pip:'
    ls $HOME/.cache/pip
fi

# LEARN MORE:
# https://stackoverflow.com/questions/14296531/what-does-error-option-single-version-externally-managed-not-recognized-ind
pip install pip --upgrade
pip install setuptools --upgrade
pip install wheel --upgrade

# Log for debugging
which python
which pip
python --version
pip --version

# Install requirements via pip
pip install -r requirements.txt --upgrade --progress-bar=off

# Install PyTorch Dependancies
if [[ $TRAVIS_PYTHON_VERSION == '3.6' ]]; then
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
fi
if [[ $TRAVIS_PYTHON_VERSION == '3.5' ]]; then
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
fi
