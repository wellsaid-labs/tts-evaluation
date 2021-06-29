function(latestVersion)
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
      latest_version: latestVersion
    },
    plugin: 'latest-version-transformer',
  }]
