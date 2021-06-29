ARG KONG_BASE=kong:2.4.1

FROM kong:2.4.1

# Temporarily switch users while installing lua plugins
# https://github.com/Kong/docker-kong/issues/328
USER root

# Manual installation of third party `google-logging` plugin
COPY kong-google-logging-plugin/ kong-google-logging-plugin/
RUN cd kong-google-logging-plugin \
  && luarocks make *.rockspec

# Manually installation of custom `latest-version-transformer` plugin
COPY kong-latest-version-transformer/ kong-latest-version-transformer/
RUN cd kong-latest-version-transformer \
  && luarocks make *.rockspec

USER kong

