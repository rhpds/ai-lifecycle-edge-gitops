# AI/ML Lifecycle at the Edge demo

### Enhancing Battery Management and Operations in Electric Vehicles
This demo showcases how the integration of AI and edge computing can effectively address specific use cases in the automotive industry. The demo will highlight the practical application of AI in resource-constrained environments, making it essential to use lightweight topologies and platforms such as Single Node OpenShift (SNO) and Red Hat Device Edge (RHDE). We will develop a robust end-to-end solution, highlighting the importance of automation, as dedicated teams are usually not feasible at the edge.

![Demo Diagram](https://github.com/dialvare/showroom-ai-lifecycle-edge/blob/microshift/content/modules/ROOT/assets/images/1-3_diagram.png)

### Electric vehicle (RHDE)
Our autonomous vehicle, runs on a RHEL 9.6 operating system where we have deployed MicroShift. This MicroShift instance also comes with the AI model serving component and gitops service enabled. The model serving platform will be use to load and serve the AI models for inference. By default, two baseline models are preloaded at startup. These will later be replaced by the retrained models coming from our Single Node OpenShift instance. Also, in this machine we will deploy MinIO storage with two different buckets. One of them will be used to store sensor data, while another is dedicated to storing AI models used for battery stress and fault detection. Additionally, our Battery Monitoring System application will also run on this vehicle, making use of the trained models to provide predictions through the telemetry dashboard along with the data generated.

### Re-training Node (SNO)
This is a single-node deployment of OpenShift that will serve as the primary platform for re-training and validating our AI models in an automated manner thanks to Red Hat OpenShift AI. This single node is located outside our vehicle in the plant and will only be used when the vehicle is plugged into a charging station. In that moment, the data will flow from the MinIO data bucket in the MicroShift instance to MinIO storage that is already deployed in our SNO.

## Lab instructions
You can find the rendered lab instructions in the following link:

```
https://dialvare.github.io/showroom-ai-lifecycle-edge/
```
