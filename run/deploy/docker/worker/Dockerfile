# NOTE: The total size of this image is expected to be around ~875 mb. 733 megabytes consist of:
# - Python packages: 304 mb torch, 60 mb spacy, 73 mb numpy, 14 mb en_core_web_sm (spacy),
#                    12 mb thinc (spacy), 12 mb blis (spacy)
# - Models: 102 mb spectrogram model, 76 mb signal model
# - Other: 80 mb python 3.7

# NOTE: The GCP VM cold startup time is around 25 - 45 seconds. Afterwards, it takes time to
# download and start the docker image. Learn more:
# https://medium.com/google-cloud/understanding-and-profiling-gce-cold-boot-time-32c209fe86ab
# https://cloud.google.com/blog/products/gcp/three-steps-to-compute-engine-startup-time-bliss-google-cloud-performance-atlas?m=1

# NOTE: Alpine does not work with PyTorch wheel due to the reliance on various symbols like
# `__printf_chk`. Learn more: https://github.com/gliderlabs/docker-alpine/issues/149
FROM python:3.8-slim as base

# Learn more about multi-stage builds:
# https://blog.realkinetic.com/building-minimal-docker-containers-for-python-applications-37d0272c52f3
FROM base as builder

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# NOTE: Learn more about debian frontend, here: https://github.com/moby/moby/issues/4032
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN python3 -m venv venv

ARG TTS_BUNDLE_PATH

# NOTE: `COPY` experiment folder fist, so that this layer is reused the most.
COPY $TTS_BUNDLE_PATH $TTS_BUNDLE_PATH
COPY run/ run/
COPY lib/ lib/
COPY third_party/ third_party/

# Install Python Dependencies
RUN apt-get update --fix-missing \
  && apt-get install -y --no-install-recommends git-core gcc g++ \
  # TODO: Consider the ideas here to reduce the size of numpy installation:
  # https://towardsdatascience.com/how-to-shrink-numpy-scipy-pandas-and-matplotlib-for-your-data-product-4ec8d7e86ee4
  && venv/bin/pip install --no-cache-dir numpy \
  flask \
  gunicorn \
  torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
  unidecode \
  ftfy \
  tqdm \
  spacy==2.2.4 \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm \
  git+https://github.com/PetrochukM/HParams.git \
  git+https://github.com/PetrochukM/PyTorch-NLP.git \
  && mkdir disk/other \
  && venv/bin/pip uninstall -y pip \
  && rm -rf venv/lib/python3.8/site-packages/torch/test \
  && rm -rf venv/lib/python3.8/site-packages/caffe2 \
  # Inspired by: https://jcrist.github.io/conda-docker-tips.html
  && find venv/ -follow -type f -name '*.pyc' -delete \
  && find venv/ -follow -type f -name '*.a' -delete \
  && find venv/ -follow -type f -name '*.js.map' -delete

FROM base

WORKDIR /app

COPY --from=builder /app/ /app/

# NOTE: To avoid manually copying `espeak`'s dependencies, we install `espeak` outside of builder.
# TODO: Attempt to reduce `ffmpeg` size, it's about 200mb without any optimization...
# https://github.com/alberthdev/alberthdev-misc/wiki/Build-your-own-tiny-FFMPEG
RUN apt-get update --fix-missing \
  && apt-get install -y espeak ffmpeg \
  && rm -rf /var/lib/apt/lists/*

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
# NOTE: The most basic and the default worker type is a synchronous worker class that handles a
# single request at a time. sync worker does not support persistent connections - each connection is
# closed after response has been sent (even if you manually add Keep-Alive or Connection: keep-alive
# header in your application).
# Learn more about various `gunicorn` optimizations:
# https://medium.com/building-the-system/gunicorn-3-means-of-concurrency-efbb547674b7
# NOTE: `--access-logfile='-'` forces the access logs to go to stdout, learn more:
# https://github.com/benoitc/gunicorn/issues/1184
# https://github.com/fofanov/gunicorn/commit/e8ecc20cc547b614eaa1bed5b0bf5f3b4b4d0274
# http://docs.gunicorn.org/en/stable/settings.html#logging
ENV GUNICORN=1
CMD ["venv/bin/gunicorn", "run.deploy.worker:app", "--bind=0.0.0.0:8000", "--timeout=3600", \
  "--graceful-timeout=600", "--workers=2", "--access-logfile='-'", "--preload"]
