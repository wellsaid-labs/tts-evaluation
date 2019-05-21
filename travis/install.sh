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

python --version

# LEARN MORE:
# https://stackoverflow.com/questions/14296531/what-does-error-option-single-version-externally-managed-not-recognized-ind
python -m pip install pip --upgrade
python -m pip install setuptools --upgrade
python -m pip install wheel --upgrade

# Install requirements via pip
python -m pip install -r requirements.txt --upgrade --progress-bar=off

# Install PyTorch Dependancies
if [[ $TRAVIS_PYTHON_VERSION == '3.6' ]]; then
    python -m pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
fi

if [[ $TRAVIS_PYTHON_VERSION == '3.7' ]]; then
    python -m pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
fi
