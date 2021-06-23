/**
 * This file contains templating that produces kubernetes configuration for our
 * google-logging Kong plugin. The deployment of these configurations depend on:
 *   - Kong deployed along with the `google-logging` plugin
 *   - Service account created with relevant persmissions
 *
 *    jsonnet google-logging.jsonnet \
 *      -y \
 *      --tla-str location=us-central1 \
 *      --tla-str cluster=staging

 *    jsonnet google-logging.jsonnet ... | kubectl apply -f -
 *
 * Docs: https://docs.konghq.com/hub/smartparkingtechnology/google-logging/
 * Source: https://github.com/SmartParkingTechnology/kong-google-logging-plugin
 *
 */
local serviceAccount = import 'kong-google-logging-service-account.secrets.json';

function(namespace='kong', location, cluster)

  local pluginName = 'google-logging';

  local pluginConfiguration = {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: {
      name: 'kong-plugin-' + pluginName + '-secret',
      namespace: namespace,
    },
    stringData: {
      # See https://github.com/SmartParkingTechnology/kong-google-logging-plugin
      # google_key refers to the service account created for this plugin to write log entries
      'google-logging-config': |||
        google_key:
          private_key: "%(private_key)s"
          client_email: "%(client_email)s"
          project_id: "$(project_id)s"
          token_uri: "$(token_uri)s"
        resource:
            type: k8s_cluster
            labels:
              project_id: "$(project_id)s"
              location: us-central1
              cluster_name: staging
        retry_count: 2
        flush_timeout: 2
        batch_max_size: 200
      ||| % serviceAccount % { cluster: cluster, location: location },
    },
    type: 'Opaque',
  };

  local plugin = {
    apiVersion: 'configuration.konghq.com/v1',
    kind: 'KongClusterPlugin',
    metadata: {
      name: 'kong-plugin-' + pluginName,
      namespace: namespace,
      annotations: {
        'kubernetes.io/ingress.class': 'kong',
      },
      labels: {
        global: 'true',
      },
    },
    configFrom: {
      secretKeyRef: {
        namespace: pluginConfiguration.metadata.namespace,
        name: pluginConfiguration.metadata.name,
        key: 'google-logging-config',
      },
    },
    plugin: pluginName,
  };

  [pluginConfiguration, plugin]
