
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
