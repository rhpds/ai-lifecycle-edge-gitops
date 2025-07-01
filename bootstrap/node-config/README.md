# GitOps repo for stroage configuration

## Components

* MinIO: s3 storage used to accommodate data and models. Therefore, two buckets are created:

  * `s3-storage`: used to store telemetry data.
  * `models`: used to store the new trained AI models.

## Installation

Create a new Argo application that points to `bootstrap/node-config/groups/dev`.

````yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: node-config
  namespace: openshift-gitops
spec:
  destination:
    name: ''
    namespace: ''
    server: https://kubernetes.default.svc
  source:
    path: bootstrap/node-config/groups/dev
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
oc apply -f node-config.yaml -n openshift-gitops
````

