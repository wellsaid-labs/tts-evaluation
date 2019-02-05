# Inspired by:
# https://blog.realkinetic.com/building-minimal-docker-containers-for-python-applications-37d0272c52f3

# Builder stage
# NOTE: Alpine does not work with PyTorch wheel due to the reliance on various symbols like
# `__printf_chk`. Learn more: https://github.com/gliderlabs/docker-alpine/issues/149
FROM python:3.6-slim as base
FROM base as builder

RUN python3 -m venv /venv

# NOTE: Enable git installs via pip
RUN apt-get update \
  && apt-get install -y --no-install-recommends git-core \
  && /venv/bin/pip install numpy \
  git+https://github.com/PetrochukM/PyTorch-NLP.git \
  flask \
  gunicorn \
  https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl \
  python-dotenv

RUN ls /usr/local

FROM base

# NOTE: COPY experiment folder fist, so that this layer is reused the most.
COPY experiments/ experiments/
COPY --from=builder /venv /venv
COPY src/ src/
COPY third_party/ third_party/

LABEL maintainer="michael@wellsaidlabs.com"

# Start web service
EXPOSE 8000

# Concerning binding to 0.0.0.0, learn more here:
# https://stackoverflow.com/questions/35414479/docker-ports-are-not-exposed

# Increased timeout to 3600 which is an hour due to the nature of this API.
CMD ["/venv/bin/gunicorn", "src.service.worker:app", "--bind=0.0.0.0:8000", "--timeout=3600", "--keep-alive=300"]
