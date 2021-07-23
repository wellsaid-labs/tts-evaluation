/**
 * This file contains templating that produces kubernetes configuration for our
 * fallback route. This is a catch-all route that uses Kong's request-termination
 * plugin that ends the request at the proxy layer.
 *
 * Note that the `includeTls` argument is needed as part of our TLS Certificate
 * issuance process (see `~/ops/tls/README.md)
 *
 *    jsonnet fallback-route.jsonnet \
 *      -y \
 *      --tla-str env=staging
 *      --tla-str includeTls=true
 *
 *    jsonnet fallback-route.jsonnet ... | kubectl apply -f -
 *
 * Reference: https://docs.konghq.com/kubernetes-ingress-controller/1.3.x/guides/configuring-fallback-service/
 * Reference: https://docs.konghq.com/kubernetes-ingress-controller/1.3.x/guides/cert-manager/#request-tls-certificate-from-lets-encrypt
 * Reference: https://docs.konghq.com/hub/kong-inc/request-termination/
 */

function(env, includeTls='true')

  local hostname = if env == 'staging' then
    'staging.tts.wellsaidlabs.com'
    else if env == 'prod' then
    'tts.wellsaidlabs.com';

  local plugin = {
    apiVersion: 'configuration.konghq.com/v1',
    kind: 'KongPlugin',
    metadata: {
      name: 'tts-wellsaidlabs-com-request-termination',
      namespace: 'kong',
    },
    config: {
      status_code: 404,
      message: 'WellSaid Labs - Resource Not Found',
    },
    plugin: 'request-termination',
  };

  local ingress = {
    apiVersion: 'extensions/v1beta1',
    kind: 'Ingress',
    metadata: {
      name: 'tts-wellsaidlabs-com',
      namespace: 'kong',
      annotations: {
        'kubernetes.io/ingress.class': 'kong',
        'konghq.com/plugins': plugin.metadata.name,
      } + (if includeTls == 'true' then {
        // Value must match name of ClusterIssuer, see ../../tls/clsuterIssuer.yaml
        'cert-manager.io/cluster-issuer': 'letsencrypt-cluster-issuer',
        'kubernetes.io/tls-acme': 'true',
      } else {}),
    },
    spec: {
      rules: [
        {
          host: hostname,
          http: {
            paths: [
              {
                path: '/',
                pathType: 'Prefix',
                backend: {
                  serviceName: 'gateway-kong-proxy',
                  servicePort: 80,
                },
              },
            ],
          },
        },
      ],
    } + (if includeTls == 'true' then {
      tls: [
        {
          secretName: 'tts-wellsaidlabs-com',
          hosts: [hostname],
        },
      ],
    } else {}),
  };

  [ingress, plugin]
