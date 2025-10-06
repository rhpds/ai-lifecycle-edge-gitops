# GitOps repo for node configuration

## Components

* MinIO: s3 storage used to accommodate data and models. Also to store artifacts generated when using pipelines. Therefore, two buckets are created:
  
  * `storage`: used to store data from sensors.
  * `models`: used to store the base and updated AI models.

## Installation

Create a new Argo application that points to `bootstrap/minio-storage-microshift/groups/dev`.

````yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: minio-storage-microshift
  namespace: openshift-gitops
spec:
  destination:
    name: ''
    namespace: ''
    server: https://kubernetes.default.svc
  source:
    path: bootstrap/minio-storage-microshift/groups/dev
    repoURL: https://github.com/dialvare/ai-lifecycle-edge.git
    targetRevision: main
  sources: []
  project: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
````

Install the argo app:

````shellscript
oc apply -f minio-storage-microshift.yaml -n openshift-gitops
````


