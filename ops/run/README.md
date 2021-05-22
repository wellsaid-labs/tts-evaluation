# Runtime Service Configuration

This directory contains code for deploying individual instances of the TTS service.

Right now this setup utilizes Kustomize, but we plan to deprecate Kustmoize in
the near future in favor of Pulumi or jsonnet.

## Deploying

There's currently single instance of the service that's running the smaller
image prepared by Neil. That service is accessible at [https://ttsdev.wellsaidlabs.com](https://ttsdev.wellsaidlabs.com).

The YAML files that define this instance live in the `dev/` directory.

Before updating anything you'll need to populate a file that's not
committed to the repository with the API keys. Eventually this file will
be stored in some sort of secret management solution. Follow these
steps to create and populate that file:

1. Create the secrets file:

    ```
    touch dev/.env.api-keys
    ```

2. Edit the file and add an API key per line, like so:

    ```
    SAMS_SPEECH_API_KEY=XXX
    NEIL_SPEECH_API_KEY=YYY
    ```

   It's important that every entry end with the `_SPEECH_API_KEY` suffix, as
   the Python application specifically looks for environment variables ending
   in this value.

Next make the desired changes to the YAML files in `base/` or `dev/`. The changes in `dev/`
are merged over the definitions in `base/`.

For instance, to update the mininum number of instances edit `base/service.yaml` like so:

```diff
      annotations:
-        autoscaling.knative.dev/minScale: '0'
+        autoscaling.knative.dev/minScale: '3'
        autoscaling.knative.dev/maxScale: '30'
```


Once you're done you can deploy your changes like so:

1. First, edit `dev/set-service-name.yaml` and increment the value of the revision name:

    ```diff
        - op: replace
          path: /spec/template/metadata/name
    -     value: tts-dev-002
    +     value: tts-dev-003
    ```

2. Then deploy that revision:

    ```
    gcloud config set project voice-service-2-313121
    gcloud container clusters get-credentials tts-dev --region us-central1
    kubectl apply -k dev/
    ```

You can see the new revision and it's status like so:

```
kubectl get revisions --namespace=dev
```

Once that command reports the new revision as ready, test it like so:

```
export API_KEY=XXX
curl \
    -H "Content-Type: application/json" \
    -X POST \
    -d "{ \"text\": \"hello world\", \"speaker_id\": 2, \"api_key\": \"$API_KEY\" }" \
    https://ttsdev.wellsaidlabs.com/api/speech_synthesis/v1/text_to_speech/stream \
    --output audio.mp3
```

Open the resulting `audio.mp3` file to make sure it's well formed and confirm it's working.

If you want to see the number of actual pods running at any point in time you can
do so like this:

```
kubectl get pods --namespace=dev
```

