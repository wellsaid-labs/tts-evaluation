# NOTE: The total size of this image is expected to be around ~756mb. 715 megabytes consist of:
# - Python dependencies: 247 mb torch, 75 mb numpy
# - Model weights: 98 mb spectrogram model, 41 mb signal model
# - Other dependencies: 135 mb intel, 63 mb python 3.6, 56 mb standard packages

# NOTE: The GCP VM cold startup time is around 25 - 45 seconds. Afterwards, it takes time to
# download and start the docker image. Learn more:
# https://medium.com/google-cloud/understanding-and-profiling-gce-cold-boot-time-32c209fe86ab
# https://cloud.google.com/blog/products/gcp/three-steps-to-compute-engine-startup-time-bliss-google-cloud-performance-atlas?m=1

# NOTE: Alpine does not work with PyTorch wheel due to the reliance on various symbols like
# `__printf_chk`. Learn more: https://github.com/gliderlabs/docker-alpine/issues/149
FROM python:3.6-slim as base

# Learn more about multi-stage builds:
# https://blog.realkinetic.com/building-minimal-docker-containers-for-python-applications-37d0272c52f3
FROM base as builder

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# NOTE: Learn more about debian frontend, here: https://github.com/moby/moby/issues/4032
ENV DEBIAN_FRONTEND=noninteractive

# NOTE: This is PyTorch's MKL installation:
# https://github.com/pytorch/builder/blob/master/manywheel/Dockerfile_101#L16
RUN apt-get update --fix-missing \
  && apt-get install -y wget \
  # Required for `apt-key add`
  && apt-get install -y gnupg \
  # Required for `https` for various Docker installs.
  && apt-get install -y apt-transport-https \
  # Add the intel APT Repository, similar to: `src/bin/install_mkl.sh`
  && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
  && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
  && sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' \
  && apt-get update \
  && apt-get install -y --no-install-recommends intel-mkl-2019.4-070 \
  # Configure dynamic linker MKL run-time bindings
  && echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf \
  && echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf \
  && ldconfig

WORKDIR /app

RUN python3 -m venv venv

# NOTE: `COPY` experiment folder fist, so that this layer is reused the most.
COPY disk/experiments/spectrogram_model/add-your-experiment-here/checkpoints/add-your-checkpoint-here/ \
  disk/experiments/spectrogram_model/add-your-experiment-here/checkpoints/add-your-checkpoint-here/
COPY disk/experiments/signal_model/add-your-experiment-here/checkpoints/add-your-checkpoint-here/ \
  disk/experiments/signal_model/add-your-experiment-here/checkpoints/add-your-checkpoint-here/
COPY src/ src/
COPY third_party/ third_party/

# Install Python Dependencies
RUN apt-get update --fix-missing \
  && apt-get install -y --no-install-recommends git-core \
  && apt-get install -y ninja-build g++ \
  # TODO: Consider the ideas here to reduce the size of numpy installation:
  # https://towardsdatascience.com/how-to-shrink-numpy-scipy-pandas-and-matplotlib-for-your-data-product-4ec8d7e86ee4
  && venv/bin/pip install --no-cache-dir numpy \
  flask \
  gunicorn \
  # TODO: PyTorch is ~233mb due to MKL and MKLDNN. It may be possible to share our MKL installation
  # with PyTorch; however, we ran into issues while trying to do this:
  # https://github.com/pytorch/pytorch/issues/24844
  torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
  unidecode \
  tqdm \
  git+https://github.com/PetrochukM/HParams.git \
  git+https://github.com/PetrochukM/PyTorch-NLP.git \
  && mkdir disk/other \
  && venv/bin/python -m src.service.worker_setup \
  && venv/bin/pip uninstall -y pip \
  && rm -rf venv/lib/python3.6/site-packages/torch/test \
  && rm -rf venv/lib/python3.6/site-packages/caffe2 \
  # Inspired by: https://jcrist.github.io/conda-docker-tips.html
  && find venv/ -follow -type f -name '*.pyc' -delete \
  && find venv/ -follow -type f -name '*.a' -delete \
  && find venv/ -follow -type f -name '*.js.map' -delete

FROM base

WORKDIR /app

COPY --from=builder /app/ /app/

# We do not need 94.8% of `/opt/intel/` folder (2.6G) leaving ~135m.
COPY --from=builder /opt/intel/mkl/include /opt/intel/mkl/include
COPY --from=builder /opt/intel/mkl/lib/intel64/libmkl_rt.so /opt/intel/mkl/lib/intel64/libmkl_rt.so
COPY --from=builder /opt/intel/mkl/lib/intel64/libmkl_core.so /opt/intel/mkl/lib/intel64/libmkl_core.so
COPY --from=builder /opt/intel/mkl/lib/intel64/libmkl_intel_thread.so /opt/intel/mkl/lib/intel64/libmkl_intel_thread.so
COPY --from=builder /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so
# Configure dynamic linker MKL run-time bindings
RUN echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf \
  && echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf \
  && ldconfig

LABEL maintainer="michael@wellsaidlabs.com"

# Start web service
EXPOSE 8000

RUN echo 'Top N largest directories\n' && du -h / | sort -rh | head -100

# Concerning binding to 0.0.0.0, learn more here:
# https://stackoverflow.com/questions/35414479/docker-ports-are-not-exposed

# Increased timeout to 3600 which is an hour due to the nature of this API.
# TODO: `venv/bin/gunicorn` takes 3 seconds to initialize, consider looking into speeding up it's
# initialization time.
# NOTE: `--workers=2` enables the worker to handle multiple requests in parallel. All endpoints
# on the worker can handle parallel requests; however, the stream endpoint will suffer a large
# performance decrease.
# Learn more about various `gunicorn` optimizations:
# https://medium.com/building-the-system/gunicorn-3-means-of-concurrency-efbb547674b7
# NOTE: `--access-logfile='-'` forces the access logs to go to stdout, learn more:
# https://github.com/benoitc/gunicorn/issues/1184
# https://github.com/fofanov/gunicorn/commit/e8ecc20cc547b614eaa1bed5b0bf5f3b4b4d0274
# http://docs.gunicorn.org/en/stable/settings.html#logging
CMD ["venv/bin/gunicorn", "src.service.worker:app", "--bind=0.0.0.0:8000", "--timeout=3600", \
  "--keep-alive=300", "--workers=2", "--access-logfile='-'"]
