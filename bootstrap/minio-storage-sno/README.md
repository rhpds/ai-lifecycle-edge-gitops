# GitOps repo for MinIO storage configuration

## Components

* MinIO: s3 storage used to accommodate pipeline artifacts, data and models. Therefore, a few buckets are created:

  * `s3-storage`: Connected to the Workbench.
    * `models`: used to store the new trained AI models.
    * `data`: used to store telemetry data.
  * `pipelines`: used to store pipeline artifacts.

## Installation

Create a new Argo application that points to `bootstrap/minio-storage-sno/groups/dev`.

````yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: minio-storage-sno
  namespace: openshift-gitops
spec:
  destination:
    name: ''
    namespace: ''
    server: https://kubernetes.default.svc
  source:
    path: bootstrap/minio-storage-sno/groups/dev
    repoURL: https://github.com/dialvare/ai-lifecycle-edge.git
    targetRevision: microshift
  sources: []
  project: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
````

Install the argo app:

````shellscript
oc apply -f minio-storage-sno.yaml -n openshift-gitops
````

