
## FAQ

### No command 'nvcc' found

Depending on your CUDA installation, this is a reoccurring problem. The solution can be found here:
https://devtalk.nvidia.com/default/topic/457664/nvcc-quot-no-command-nvcc-found-quot-/

Typically, something like this works:

    export LD_LIBRARY_PATH=/usr/local/cuda/lib
    export PATH=$PATH:/usr/local/cuda/bin

### IOError: [Errno 24] Too many open files

The error can occur during processing of speech files. Typically, this can be resolved either with:

    ulimit -n 65536

or

    sudo sh -c "ulimit -n 65536 && exec su $LOGNAME"

### SystemError: unknown opcode

For me, this happened when I tried to load a model checkpoint on a different python version
than it was trained on. Specifically, I tried to load a model checkpoint created by Python3.5
in jupyter using Python 3.6.

To fix this, [this](https://stackoverflow.com/questions/9386048/ipython-reads-wrong-python-version)
Stack Overflow was helpful for fixing iPython python version.

To fix Jupyter, [this](https://github.com/jupyter/notebook/issues/2563) on GitHub was helpful.

### Could not install packages due to an EnvironmentError: [Errno 13] Permission denied

With pip ``--user`` installations, this error can occur because the root owns the ``.local``
directory that pip is attempting to install to. This can be solved by changing the folder's
ownership:

    sudo chown -R $(whoami) /home/michaelp/.local/
