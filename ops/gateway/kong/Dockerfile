ARG KONG_BASE=kong:2.4.1

FROM kong:2.4.1

# Temporarily switch users while installing lua plugins
# https://github.com/Kong/docker-kong/issues/328
USER root

# Manually download and make the `google-logging` lua plugin.
# NOTE: This third party plugin requires manual installation. We were not able to load this plugin
#   via a simple ConfigMap due to external dependencies.
#   https://docs.konghq.com/hub/smartparkingtechnology/google-logging/#install
# NOTE: this should have been as simple as `RUN luarocks install kong-plugin-google-logging` but that was not the case :(
# RUN wget -O "kong-google-logging-plugin.zip" "https://github.com/SmartParkingTechnology/kong-google-logging-plugin/archive/master.zip"
RUN wget -O "kong-google-logging-plugin.zip" "https://github.com/wellsaid-labs/kong-google-logging-plugin/archive/master.zip"
RUN unzip kong-google-logging-plugin.zip \
  && cd ./kong-google-logging-plugin-master \
  && luarocks make *.rockspec
RUN rm kong-google-logging-plugin.zip

# Manually download and make `latest-version-
COPY kong-latest-version-transformer/ kong-latest-version-transformer/
RUN cd kong-latest-version-transformer \
  && luarocks make *.rockspec
# RUN wget -O "kong-latest-version-transformer.zip" "https://github.com/wellsaid-labs/kong-latest-version-transformer/archive/master.zip"
# RUN unzip kong-latest-version-transformer.zip \
#   && cd ./kong-latest-version-transformer-master \
#   && luarocks make *.rockspec
# RUN rm kong-latest-version-transformer.zip

RUN luarocks list

USER kong

