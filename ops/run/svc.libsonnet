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
    },
    spec: {
      template: {
        metadata: {
          name: spec.name + '-' + spec.version,
          annotations: {
            'autoscaling.knative.dev/minScale': '' + spec.scale.min,  // cast to str
            'autoscaling.knative.dev/maxScale': '' + spec.scale.max,  // cast to str
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
                '--workers=' + spec.concurrency,
                "--access-logfile='-'",
                '--preload',
              ],
              ports: [
                {
                  containerPort: 8000,
                },
              ],
              resources: {
                /**
                 * TODO: This might be something we need to tweak per service,
                 * as the input validation service might use less resources.
                 * If that's the case we could binpack better, which would
                 * improve cold-start latency and reduce costs.
                 */
                requests: {
                  cpu: '7',
                  memory: '5G',
                },
              },
              readinessProbe: {
                successThreshold: 1,
                httpGet: {
                  path: '/healthy',
                },
              },
            } + (if "apiKeySecretName" in spec && spec.apiKeySecretName != null then {
              envFrom: [
                {
                  secretRef: {
                    name: spec.apiKeySecretName,
                  },
                },
              ],
            } else {}),
          ],
        },
      },
      traffic: [
        {
          percent: 100,
          latestRevision: true,
        },
      ],
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
    // NOTE: in the event that we add the api_key to body transformation we store the plugin
    // configuration in a Secret.
    // https://docs.konghq.com/hub/kong-inc/request-transformer/
    local requestTransformerConfig = {
      replace: {
        headers: [
          'host:' + spec.serviceName + '.' + spec.namespace + '.svc.cluster.local',
        ],
      },
      add: {
        headers: [
          'host:' + spec.serviceName + '.' + spec.namespace + '.svc.cluster.local',
        ],
      } + if spec.apiKey != null then {
        body: [
          'api_key:' + spec.apiKey
        ],
      } else {},
    };

    local requestTransformerConfigSecret = if spec.apiKey != null then {
      apiVersion: 'v1',
      kind: 'Secret',
      metadata: {
        name: 'route-' + spec.serviceName + '-request-transformer-config',
        namespace: spec.namespace,
      },
      stringData: {
        // https://docs.konghq.com/hub/kong-inc/request-transformer/
        'request-transformer-config': std.manifestYamlDoc(requestTransformerConfig, true),
      },
      type: 'Opaque',
    };

    local requestTransformer = {
      apiVersion: 'configuration.konghq.com/v1',
      kind: 'KongPlugin',
      metadata: {
        name: 'route-' + spec.serviceName + '-request-transformer',
        namespace: spec.namespace,
      },
      plugin: 'request-transformer',
    } + (if spec.apiKey != null then {
      configFrom: {
        secretKeyRef: {
          namespace: spec.namespace,
          name: requestTransformerConfigSecret.metadata.name,
          key: std.objectFields(requestTransformerConfigSecret.stringData)[0]
        },
      },
    } else {
      config: requestTransformerConfig
    });

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
        retries: 5,
        connect_timeout: 60000,
        read_timeout: 60000,
        write_timeout: 60000
      } + (if "proxy" in spec then spec.proxy else {}),
      route: {
        request_buffering: true,
        response_buffering: true,
        headers: {
          'accept-version': [
            spec.namespace,
          ],
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
    // NOTE: prune simply removes requestTransformerConfigSecret if null
    std.prune([requestTransformerConfigSecret, requestTransformer, kongIngress, ingress])
}
