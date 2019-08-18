# NOTE: Alpine does not work with PyTorch wheel due to the reliance on various symbols like
# `__printf_chk`. Learn more: https://github.com/gliderlabs/docker-alpine/issues/149
FROM python:3.6-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# NOTE: Learn more about debian frontend, here: https://github.com/moby/moby/issues/4032
ENV DEBIAN_FRONTEND=noninteractive

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
  && ldconfig \
  # Removes > 90% of `/opt/intel/` folder leaving ~155m. There is 20m more potentially that can be
  # deleted.
  && rm -rf /opt/intel/mkl/lib/ia32_lin \
  && rm -rf /opt/intel/mkl/lib/intel64_lin/*.a \
  && rm -rf /opt/intel/conda_channel* \
  && rm -rf /opt/intel/mkl/benchmarks \
  && rm -rf /opt/intel/compilers_and_libraries_2019.4.243/linux/compiler/lib/ia32_lin \
  && rm -rf /opt/intel/tbb/lib/ia32_lin \
  && rm -rf /opt/intel/mkl/examples \
  && rm -rf /opt/intel/compilers_and_libraries_2019.4.243/linux/bin \
  && rm -rf /opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/interfaces \
  && find /opt/intel/mkl/lib/intel64_lin/ ! -name "libmkl_rt.so" ! -name "libmkl_core.so" \
  ! -name "libmkl_intel_thread.so" ! -name "libmkl_intel_lp64.so" -maxdepth 1 -type f -delete \
  && apt-get -y --purge autoremove wget gnupg \
  && apt-get -y clean \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv

# NOTE: COPY experiment folder fist, so that this layer is reused the most.
COPY experiments/ experiments/
COPY src/ src/
COPY third_party/ third_party/

# Install Python Dependencies
RUN apt-get update --fix-missing \
  && apt-get install -y --no-install-recommends git-core \
  && apt-get install -y ninja-build g++ \
  && /venv/bin/pip install --no-cache-dir numpy \
  git+https://github.com/PetrochukM/PyTorch-NLP.git \
  flask \
  gunicorn \
  torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
  python-dotenv \
  unidecode \
  && /venv/bin/python -m src.service.build \
  # Remove large unused directories.
  && /venv/bin/pip uninstall -y pip \
  && rm -rf /venv/lib/python3.6/site-packages/torch/test \
  && rm -rf /venv/lib/python3.6/site-packages/caffe2 \
  # Inspired by: https://jcrist.github.io/conda-docker-tips.html
  && find /venv/ -follow -type f -name '*.pyc' -delete \
  && find /venv/ -follow -type f -name '*.a' -delete \
  && find /venv/ -follow -type f -name '*.js.map' -delete \
  && apt-get -y --purge autoremove git-core g++ ninja-build \
  && apt-get -y clean \
  && rm -rf /var/lib/apt/lists/*

LABEL maintainer="michael@wellsaidlabs.com"

# Start web service
EXPOSE 8000

RUN echo 'Top N largest directories\n' && du -h  | sort -rh | head -100

# Concerning binding to 0.0.0.0, learn more here:
# https://stackoverflow.com/questions/35414479/docker-ports-are-not-exposed

# Increased timeout to 3600 which is an hour due to the nature of this API.
CMD ["/venv/bin/gunicorn", "src.service.worker:app", "--bind=0.0.0.0:8000", "--timeout=3600", "--keep-alive=300"]
