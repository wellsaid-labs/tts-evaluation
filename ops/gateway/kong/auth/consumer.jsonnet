
function(username, secretKey)
  local namespace = 'kong-consumers';

  local secret = {
    apiVersion: 'v1',
    kind: 'Secret',
    type: 'Opaque',
    metadata: {
      name: username + '-consumer-apikey',
      namespace: namespace,
    },
    stringData: {
      kongCredType: 'key-auth',
      key: secretKey
    },
  };

  local consumer = {
    apiVersion: 'configuration.konghq.com/v1',
    kind: 'KongConsumer',
    metadata: {
      name: username + '-consumer',
      namespace: namespace,
      annotations: {
        'kubernetes.io/ingress.class': 'kong',
      },
    },
    username: username,
    credentials: [secret.metadata.name]
  };

  [secret, consumer]
