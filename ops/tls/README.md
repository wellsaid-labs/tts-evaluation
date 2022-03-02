# TLS Certificate issuance for HTTPS

## Prerequisites

- Static IP reserved, see [ClusterSetup.md](../ClusterSetup.md)
- Kong gateway deployed, see [gateway/README.md](../gateway/README.md)
  - It is especially important that the
    [fallback route](../gateway/kong/plugins/fallback-route.jsonnet) is deployed
    (with the `includeTls=false` flag).
- DNS A record entry for hostname -> static ip mapping

## Deploying `cert-manager`

[Cert-manager](https://cert-manager.io/docs/) is a service that can be deployed
in a kubernetes environment for managing TLS certificate issuance and renewal.
For the most part, we are following the
[Using cert-manager with Kong](https://docs.konghq.com/kubernetes-ingress-controller/1.3.x/guides/cert-manager/)
guide to obtain a certificate and then configure our routes to use https.

1. Deploy the `fallback-route` _without_ the tls configurations (allows
   cert-manager to setup the ClusterIssuer prior to Certificate request)

```bash
 jsonnet ops/gateway/kong/plugins/fallback-route.jsonnet \
    -y \
    --tla-str env=$ENV \
    --tla-str includeTls=false \
    | kubectl apply -f -
```

1. Install `cert-manager`

   ```bash
   helm repo add jetstack https://charts.jetstack.io
   helm repo update
   helm install \
     cert-manager jetstack/cert-manager \
     --namespace cert-manager \
     --create-namespace \
     --version v1.4.0 \
     --set installCRDs=true
   ```

1. Setup the ClusterIssuer resource.

   ```bash
   kubectl apply -f ops/tls/clusterIssuer.yaml
   # Monitor the status of the Issuer via
   kubectl describe ClusterIssuer/letsencrypt-cluster-issuer
   # If you need to dig further, look into logs of the cert-manager service
   kubectl get pods -n cert-manager
   kubectl logs -n cert-manager $CERT_MANAGER_POD -f
   ```

   Once `Status.Conditions.Reason` reads `ACMEAccountRegistered` for the
   `ClusterIssuer`, the cluster should be ready to issue a certificate for us.
   The `cert-manager` service will listen for resources with specific
   annotations and respond accordingly

   - `kubernetes.io/tls-acme: "true"`: tells cert-manager to provision a cert
     for the host(s) defined in this Ingress resource. The host(s) are defined
     under the Ingress `spec.tls` field
   - `cert-manager.io/cluster-issuer: letsencrypt-cluster-issuer`: specify the
     issuer to be used, see our `./clusterIssuer.yaml`

   An Ingress resource with the above configurations is defined in the
   `fallback-route` resource in the Kong plugins directory

1. Re-deploy the Kong `fallback-route` with the tls configurations, triggering
   the Certificate Request

   ```bash
   jsonnet ops/gateway/kong/plugins/fallback-route.jsonnet \
      -y \
      --tla-str env=$ENV \
      --tla-str includeTls=true \
      | kubectl apply -f -
   ```

   Check the status of the Certificate:

   ```bash
   kubectl get certificates.cert-manager.io -n kong
   kubectl describe certificates.cert-manager.io/tts-wellsaidlabs-com -n kong
   ```

## Troubleshooting

### Certificate fails to auto-renew

_Assuming the environment is already setup and certificate has previously been
issued_

```bash
# Confirm the certificate is in a valid (although expired) state: Type=Ready, Status=True
kubectl describe certificates.cert-manager.io/tts-wellsaidlabs-com -n kong
# Restart the cert-manager
kubectl rollout restart deployment/cert-manager -n cert-manager
```
