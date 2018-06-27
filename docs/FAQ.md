
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

### /usr/bin/ld: cannot find -lz

This error can occur when installing NV-WaveNet and can be resolved with the instructions [here](https://stackoverflow.com/questions/3373995/usr-bin-ld-cannot-find-lz).

### nv_wavenet_util.cuh(89): error: more than one conversion function from "half"...

This error can occur when installing NV-WaveNet and can be resolved with the instructions [here](https://github.com/NVIDIA/nv-wavenet/issues/5).

### SystemError: unknown opcode

For me, this happened when I tried to load a model checkpoint on a different python version
than it was trained on. Specifically, I tried to load a model checkpoint created by Python3.5
in jupyter using Python 3.6.

To fix this, [this](https://stackoverflow.com/questions/9386048/ipython-reads-wrong-python-version)
Stack Overflow was helpful for fixing iPython python version.

To fix Jupyter, [this](https://github.com/jupyter/notebook/issues/2563) on GitHub was helpful.

