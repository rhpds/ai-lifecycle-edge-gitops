# GitOps repo for model serving configuration

## Components
- Server: component used for serving AI models for inference, including the ServingRuntime and InferenceService.  
  
## Installation

Create a new Argo application that points to `bootstrap/model-server/groups/dev`.

````yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: model-server
  namespace: openshift-gitops
spec:
  destination:
    name: ''
    namespace: ''
    server: https://kubernetes.default.svc
  source:
    path: bootstrap/model-server/groups/dev
    repoURL: https://github.com/dialvare/ai-lifecycle-edge.git
    targetRevision: test
  sources: []
  project: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
````

Install the argo app:

````shellscript
oc apply -f model-server.yaml -n openshift-gitops
````


