# GitOps repo for battery-simulation demo

## Components
- Mosquitto: MQTT broker that receives the telemetry data coming from the emulated sensors.
- Battery simulator: Quarkus component that simulates the battery of a driving electric vehicle. The BMS is sending telemetry data to the MQTT broker.
- InfluxDB: Time series database, configured with auto setup.
- influx-exporter: Scheduled task running every 10 minutes that collects the data stored in InfluxDB nd sends it to the a MinIO bucket.
- data-ingester: Quarkus component that reads data from MQTT and stores it InfluxDB.
- BMS Dashboard: Angular component that displays the battery telemetry data in realtime and serves as a frontend for a GenAI chatbot.
- mqtt2ws: Camel Quarkus component that reads data from MQTT and exposes it as Websocket for the BMS Dashboard.

## Installation

First, modify the `groups/dev/files/config.json` file and replace `.example.com`, in the `BATTERY_METRICS_WS_ENDPOINT` variable with your cluster domain. 

Then, create a new Argo application that points to `bootstrap/battery-simulation/groups/dev`. It will install all the components in the `battery-demo` namespace

````yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: battery-simulation
  namespace: openshift-gitops
spec:
  destination:
    name: ''
    namespace: ''
    server: https://kubernetes.default.svc
  source:
    path: bootstrap/battery-simulation/groups/dev
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
oc apply -f battery-simulation.yaml -n openshift-gitops
````

## Exploring the data in InfluxDB

Login to the InfluxDB Web UI with user `admin` and password `password`.
Navigate to `Explore` and create a query with `Script Editor`. 
Use this query to find the data from the last hour:

````shellscript
from(bucket: "bms")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "battery_data")
````

## Troubleshooting

It might be necessary to restart the `battery-simulation` pod when the AMQ broker is ready.
Check the logs to see if the component is sending the data.
