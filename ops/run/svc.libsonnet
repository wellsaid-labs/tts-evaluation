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
                'src.service.worker:app',
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
              envFrom: [
                {
                  secretRef: {
                    name: spec.apiKeysSecret,
                  },
                },
              ],
            },
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
    local requestTransformer = {
      apiVersion: 'configuration.konghq.com/v1',
      kind: 'KongPlugin',
      metadata: {
        name: 'route-' + spec.serviceName + '-request-transformer',
        namespace: spec.namespace,
      },
      config: {
        replace: {
          headers: [
            'host:' + spec.serviceName + '.' + spec.namespace + '.svc.cluster.local',
          ],
        },
        add: {
          headers: [
            'host:' + spec.serviceName + '.' + spec.namespace + '.svc.cluster.local',
          ],
        },
      },
      plugin: 'request-transformer',
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
        // TODO: restrict to https once TLS is live
        protocols: [
          'http',
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
          'konghq.com/override': kongIngress.metadata.name, //'route-configuration',
          'konghq.com/plugins': requestTransformer.metadata.name,
        },
      },
      spec: {
        rules: [
          {
            http: {
              paths: [
                {
                  path: path,
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
