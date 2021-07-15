/**
 * This file contains code for defining an individual TTS Cloud Run Service.
 * A service is an individually managed application that's dynamically
 * scaled by Cloud Run's control plane.
 */
{
  Service: function(spec) {
    apiVersion: 'serving.knative.dev/v1',
    kind: 'Service',
    metadata: {
      name: spec.name,
      namespace: spec.namespace,
      labels: {
        'serving.knative.dev/visibility': 'cluster-local'
      },
    },
    spec: {
      template: {
        metadata: {
          name: spec.name + '-' + spec.version,
          annotations: {
            'autoscaling.knative.dev/minScale': '' + spec.scale.min,  // cast to str
            'autoscaling.knative.dev/maxScale': '' + spec.scale.max,  // cast to str
            'autoscaling.knative.dev/scaleDownDelay': '90s',
            'run.googleapis.com/ingress': 'internal',
          },
        },
        spec: {
          containerConcurrency: spec.concurrency,
          timeoutSeconds: spec.timeout,
          containers: [
            {
              name: 'app',
              image: spec.image,
              args: [
                'venv/bin/gunicorn',
                spec.entrypoint,
                '--bind=0.0.0.0:8000',
                '--timeout=' + spec.timeout,
                '--graceful-timeout=' + spec.restartTimeout,
                // NOTE: min 2 workers to ensure readiness probes / health checks
                // work properly while under load
                '--workers=' + std.max(spec.concurrency, 2),
                "--access-logfile=-",
                '--preload',
              ],
              ports: [
                {
                  containerPort: 8000,
                },
              ],
              resources: spec.resources,
              readinessProbe: {
                successThreshold: 1,
                httpGet: {
                  path: '/healthy',
                },
              },
            } + (if "legacyContainerApiKey" in spec && spec.legacyContainerApiKey != null then {
              env: [
                {
                  // NOTE: _SPEECH_API_KEY suffix required per legacy container auth logic
                  name: 'LEGACY_SPEECH_API_KEY',
                  value: spec.legacyContainerApiKey,
                },
              ],
            } else {}),
          ],
        },
      },
      traffic: spec.traffic
    },
  },
  /**
   * A Route consists of the following resources that will tie the tts Service into our
   * Kong gateway.
   *    - KongPlugin: request-transformer plugin that will rewrite the host header for
   *      internal service routing (knative-local-gateway)
   *    - KongIngress: defines the kong service/route configurations and, most importantly
   *      specifies which Accept-Version header will route to the corresponding service.
   *    - Ingress: path/service routing (augmented by the KongIngress resource)
   */
  Route(spec):
    // NOTE: this transformer allows us to route `Accept-Version` headers to individual
    // Cloud Run revisions.
    // @see https://docs.konghq.com/hub/kong-inc/request-transformer/
    local hostNameTransformer = |||
      $((function()
        local value = headers['accept-version'] or ''
        local version, revision = value, nil
        local index = value:find('.', 1, true)
        if index then
          version = value:sub(1, index - 1)
          revision = value:sub(index + 1)
          return 'revision-'..revision..'-%(serviceName)s.%(namespace)s.svc.cluster.local'
        end
        return '%(serviceName)s.%(namespace)s.svc.cluster.local';
      end)())
    ||| % spec;

    local requestTransformer = {
      apiVersion: 'configuration.konghq.com/v1',
      kind: 'KongPlugin',
      metadata: {
        name: 'route-' + spec.serviceName + '-request-transformer',
        namespace: spec.namespace,
      },
      plugin: 'request-transformer',
      config: {
        replace: {
          headers: [
            'host:' + hostNameTransformer
          ],
        },
        add: {
          headers: [
            'host:' + hostNameTransformer
          ],
        } + if spec.legacyContainerApiKey != null then {
          body: [
            'api_key:' + spec.legacyContainerApiKey
          ],
        } else {}
      }
    };

    // https://docs.konghq.com/kubernetes-ingress-controller/1.2.x/references/custom-resources/#kongingress
    local kongIngress = {
      apiVersion: 'configuration.konghq.com/v1',
      kind: 'KongIngress',
      metadata: {
        name: 'route-' + spec.serviceName + '-configuration',
        namespace: spec.namespace,
      },
      proxy: {
        # defaults: https://docs.konghq.com/gateway-oss/2.4.x/admin-api/#service-object
        protocol: 'http',
        retries: 4,
        connect_timeout: 90000,
        read_timeout: 90000,
        write_timeout: 60000
      } + (if "proxy" in spec then spec.proxy else {}),
      route: {
        request_buffering: true,
        response_buffering: true,
        headers: {
          'accept-version': spec.acceptVersionHeaders,
        },
        methods: [
          'POST',
          'GET',
        ],
        protocols: [
          'https',
        ],
      },
    };

    local ingress = {
      apiVersion: 'extensions/v1beta1',
      kind: 'Ingress',
      metadata: {
        name: 'route-' + spec.serviceName,
        namespace: spec.namespace,
        annotations: {
          'kubernetes.io/ingress.class': 'kong',
          'konghq.com/override': kongIngress.metadata.name,
          'konghq.com/plugins': requestTransformer.metadata.name,
          'konghq.com/protocols':'https',
          'konghq.com/https-redirect-status-code':'301',
        },
      },
      spec: {
        rules: [
          {
            host: spec.hostname,
            http: {
              paths: [
                {
                  path: path,
                  pathType: "Exact",
                  backend: {
                    serviceName: spec.serviceName,
                    servicePort: if "servicePort" in spec then spec.servicePort else 80
                  },
                } for path in spec.servicePaths
              ],
            },
          },
        ],
      },
    };
    [requestTransformer, kongIngress, ingress]
}
