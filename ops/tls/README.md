# TLS Certificate issuance for HTTPS

## Prerequisites

- Static IP reserved, see [ClusterSetup.md](../ClusterSetup.md)
- Kong gateway deployed, see [gateway/README.md](../gateway/README.md)
- DNS A record entry for hostname -> static ip mapping

## Deploying `cert-manager`

[Cert-manager](https://cert-manager.io/docs/) is a service that can be deployed
in a kubernetes environment for managing TLS certificate issuance and renewal.
For the most part, we are following the
[Using cert-manager with Kong](https://docs.konghq.com/kubernetes-ingress-controller/1.3.x/guides/cert-manager/)
guide to obtain a certificate and then configure our routes to use https.

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

Setup the ClusterIssuer resource.

```bash
kubectl apply -f clusterIssuer.yaml
# Monitor the status of the Issuer via
kubectl describe ClusterIssuer/letsencrypt-cluster-issuer
# If you need to dig further, look into logs of the cert-manager service
kubectl logs -n cert-manager $CERT_MANAGER_POD -f
```

Once the Issuer is in an `ACMEAccountRegistered` state, the cluster should be
ready to issue a certificate for us. The `cert-manager` service will listen for
resources with specific annotations and respond accordingly

- `kubernetes.io/tls-acme: "true"`: tells cert-manager to provision a cert for
  the host(s) defined in this Ingress resource. The host(s) are defined under
  the Ingress `spec.tls` field
- `cert-manager.io/cluster-issuer: letsencrypt-cluster-issuer`: specify the
  issuer to be used, see our `./clusterIssuer.yaml`

An Ingress resource with the above configurations is created as part of the Kong
deployment process.
