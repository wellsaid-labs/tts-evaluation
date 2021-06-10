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
}
