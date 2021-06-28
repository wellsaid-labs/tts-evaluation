function(serviceName)
  [{
    apiVersion: 'configuration.konghq.com/v1',
    kind: 'KongClusterPlugin',
    metadata: {
      name: 'latest-pinned-service',
      namespace: 'kong',
      annotations: {
        'kubernetes.io/ingress.class': 'kong',
      },
      labels: {
        global: 'true',
      },
    },
    config: {
      rules: [
        {
          condition: {
            'accept-version': 'latest',
          },
          upstream_name: serviceName,
        },
      ],
    },
    plugin: 'route-by-header',
  }]
