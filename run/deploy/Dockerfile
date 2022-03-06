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

ARG TTS_PACKAGE_PATH

# NOTE: `COPY` experiment folder fist, so that this layer is reused the most.
COPY $TTS_PACKAGE_PATH $TTS_PACKAGE_PATH
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
  pythonic-config \
  git+https://github.com/PetrochukM/PyTorch-NLP.git \
  && mkdir disk/other \
  && venv/bin/pip uninstall -y pip \
  && rm -rf venv/lib/python3.8/site-packages/torch/test \
  && rm -rf venv/lib/python3.8/site-packages/caffe2 \
  # Inspired by: https://jcrist.github.io/conda-docker-tips.html
  && find venv/ -follow -type f -name '*.pyc' -delete \
  && find venv/ -follow -type f -name '*.a' -delete \
  && find venv/ -follow -type f -name '*.js.map' -delete

# NOTE: ffmpeg build stage. We are manually compiling the ffmpeg binary with a limited feature set
# based on our usage requirements (see the ffmpeg command in `worker.py`). This cuts the overall
# weight of our ffmpeg dependency down from ~268mb to ~26mb.
#   - Official compilation guide: https://trac.ffmpeg.org/wiki/CompilationGuide/Generic
#   - Dockerfile reference: https://github.com/alfg/docker-ffmpeg/blob/master/Dockerfile
#   - Informative guide: https://github.com/alberthdev/alberthdev-misc/wiki/Build-your-own-tiny-FFMPEG
FROM base as ffmpeg-build

ARG FFMPEG_VERSION=4.4

ARG PREFIX=/opt/ffmpeg
ARG LD_LIBRARY_PATH=/opt/ffmpeg/lib
ARG MAKEFLAGS="-j4"

# FFmpeg build dependencies.
RUN apt-get update --fix-missing && \
  apt-get install -y \
  build-essential \
  coreutils \
  gcc \
  libmp3lame-dev \
  openssl \
  pkgconf \
  wget \
  yasm

# Get ffmpeg source.
RUN cd /tmp/ && \
  wget http://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.gz && \
  tar zxf ffmpeg-${FFMPEG_VERSION}.tar.gz && rm ffmpeg-${FFMPEG_VERSION}.tar.gz

# Compile ffmpeg.
RUN cd /tmp/ffmpeg-${FFMPEG_VERSION} && \
  ./configure \
  --disable-shared \
  --enable-static \
  --enable-pic \
  --enable-libmp3lame \
  # general
  --disable-doc \
  --disable-debug \
  --disable-avdevice \
  --disable-swscale \
  --disable-programs \
  --enable-rdft \
  --enable-ffmpeg \
  --enable-ffprobe \
  --disable-network \
  --disable-zlib \
  --disable-bzlib \
  --disable-iconv \
  --disable-bsfs \
  --disable-indevs \
  --disable-outdevs \
  --disable-hwaccels \
  --disable-nvenc \
  --disable-xvmc \
  --disable-videotoolbox \
  --disable-audiotoolbox \
  # filters
  --disable-filters \
  --enable-filter=aformat \
  --enable-filter=aresample \
  --enable-filter=anull \
  --enable-filter=atrim \
  --enable-filter=format \
  --enable-filter=null \
  --enable-filter=setpts \
  --enable-filter=trim \
  # protocol
  --disable-protocols \
  --enable-protocol=file \
  --enable-protocol=pipe \
  # demuxer
  --disable-demuxers \
  --enable-demuxer=mp3 \
  --enable-demuxer=pcm_f32le \
  --enable-demuxer=wav \
  # muxer
  --disable-muxers \
  --enable-muxer=mp3 \
  --enable-muxer=wav \
  --enable-muxer=pcm_f32le \
  # decoder
  --disable-decoders \
  --enable-decoder=mp3 \
  --enable-decoder=mp3adu \
  --enable-decoder=mp3adufloat \
  --enable-decoder=mp3float \
  --enable-decoder=mp3on4 \
  --enable-decoder=mp3on4float \
  --enable-decoder=wavpack \
  --enable-decoder=pcm_f32le \
  # encoder
  --disable-encoders \
  --enable-encoder=libmp3lame \
  --enable-encoder=pcm_f32le \
  # parser
  --disable-parsers \
  --enable-parser=mpegaudio \
  # extra
  --extra-cflags="-I${PREFIX}/include" \
  --extra-ldflags="-L${PREFIX}/lib" \
  --extra-libs="-lpthread -lm" \
  --prefix="${PREFIX}" && \
  make && make install && make distclean && \
  rm -rf /tmp/*

# Final base image
FROM base

WORKDIR /app

COPY --from=builder /app/ /app/

# NOTE: ffmpeg linking
ENV PATH=/opt/ffmpeg/bin:$PATH
COPY --from=ffmpeg-build /opt/ffmpeg /opt/ffmpeg

RUN apt-get update --fix-missing \
  && apt-get install -y \
  # NOTE: ffmpeg dependency (mp3 encoding, see --enable-libmp3lame flag)
  lame \
  # NOTE: To avoid manually copying `espeak`'s dependencies, we install `espeak` outside of builder.
  espeak \
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
  "--graceful-timeout=600", "--workers=2", "--access-logfile=-", "--preload"]
