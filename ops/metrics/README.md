# Google Cloud Metrics

This folder contains several log-based metrics that we use to monitor our TTS
service.

In order to monitor scalability of our service we need to looks at requests and
latency, observable at the Kong gateway layer. The following filter will show us
request logs on the Kong proxy service. Note the `jsonPayload.request:*` filter
that distinguishes between request logs and the actual proxy service logs.

```txt
resource.type="k8s_container"
resource.labels.namespace_name="kong"
resource.labels.container_name="proxy"
jsonPayload.request:*
```

And a quick example of how to read those logs:

```bash
gcloud logging read \
  "resource.type=k8s_container AND\
   resource.labels.cluster_name=staging AND\
   resource.labels.namespace_name=kong AND\
   resource.labels.container_name=proxy AND\
   jsonPayload.request:*" --limit 1 --format json

```

## Creating log-based Metrics

```bash
gcloud logging metrics create kong/http_requests --config-from-file=./kong_http_requests.json

gcloud logging metrics create kong/upstream_latency --config-from-file=./kong_upstream_latency.json
```

## Viewing Metric Configuration

```bash
gcloud logging metrics describe $METRIC_NAME --format json
```

## Updating Metrics

One scenario where an update might make sense is adding an additional label. In
this scenario you would update the metric configuration and then run the
following:

```bash
gcloud logging metrics update kong/http_requests --config-from-file=../metrics/kong_http_requests.json
```

However, changes to metrics are not retroactive.
