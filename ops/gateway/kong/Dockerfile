ARG KONG_BASE=kong:2.4.1

FROM kong:2.4.1

# Temporarily switch users while installing lua plugins
# https://github.com/Kong/docker-kong/issues/328
USER root

# Manually installation of custom `latest-version-transformer` plugin
COPY kong-latest-version-transformer/ kong-latest-version-transformer/
RUN cd kong-latest-version-transformer \
  && luarocks make *.rockspec

# Manually installation of custom `content-moderation` plugin
COPY kong-content-moderation/ kong-content-moderation/
RUN cd kong-content-moderation \
  && luarocks make *.rockspec

USER kong
