# GitOps repo for node configuration

## Components

- MinIO

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
    targetRevision: HEAD
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

