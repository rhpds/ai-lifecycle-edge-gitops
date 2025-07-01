# AI/ML Lifecycle at the Edge demo

## Enhancing Battery Management and Operations in Electric Vehicles

This demo showcases how the integration of AI and edge computing can effectively address specific use cases in the automotive industry. The demo will highlight the practical application of AI in resource-constrained environments, making it essential to use lightweight topologies and platforms such as Single Node OpenShift (SNO) and Red Hat Device Edge (RHDE). We will develop a robust end-to-end solution, highlighting the importance of automation, as dedicated teams are usually not feasible at the edge.

### Electric vehicle (RHDE)
Inside our electric vehicle, we’ve deployed a RHEL machine running MicroShift. This machine is also equipped with MinIO storage. One bucket is used to store sensor data, while another is dedicated to storing AI models used for battery fault detection. Additionally, the AI Model Serving component has been deployed to load and serve the models for inference. By default, two baseline models are preloaded at startup. These will later be replaced by the retrained models coming from our Single Node OpenShift instance. Additionally, our Battery Monitoring System application will also run on this node, making use of the trained models to provide predictions through the infotainment system alonw with the data generated.

### Re-training Node (SNO)
This is a single-node deployment of OpenShift that will serve as the primary platform for training and validating our AI models in an automated manner thanks to Red Hat OpenShift AI. This operator is already deployed and configured with a Workbench that will be used to review and run the Nodebooks used for re-training. This single node is located outside our vehicle and will only be used when the vehicle is plugged into a chargin station. In that moment, the data will flow from the MinIO database in the MicroShift cluster to MinIO instance that is also deployed in our SNO.

### Solution Workflow
Sometimes, a picture is worth a thousand words. Below, you will find a diagram illustrating the main components involved in our solution:

![Demo Diagram](https://raw.githubusercontent.com/dialvare/showroom-ai-lifecycle-edge/blob/microshift/content/modules/ROOT/assets/images/1-3_diagram.png)

## Lab instructions
You can find the lab instructions in the following repository:

```
https://github.com/dialvare/showroom-ai-lifecycle-edge 
```
