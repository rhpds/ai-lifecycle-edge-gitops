# AI/ML Lifecycle at the Edge demo

## Enhancing Battery Management and Operations in Electric Vehicles

This demo showcases how the integration of AI and edge computing can effectively address specific use cases in the automotive industry. The demo will highlight the practical application of AI in resource-constrained environments, making it essential to use lightweight topologies and platforms such as Single Node OpenShift (SNO) and MicroShift. 

We will develop a robust end-to-end solution, highlighting the importance of automation, as dedicated teams are usually not feasible at the edge. To address this challenge, our solution incorporates key components within Red Hat OpenShift AI to support the entire AI/ML lifecycle at the edge, including model training, data science pipelines, model serving, and model monitoring. With all the infrastructure in place, we are going to address and resolve two key use cases:

### Battery Monitoring System
In our SNO, we will run an application responsible for monitoring the health of the electric car battery. To achieve this, we will train and use two AI models to detect battery stress and predict the time this component could potentially fail. This prediction will be made based on simulated data from various parameters such as voltage, temperature, driving distance or velocity.

### Charging Optimization
On an RHEL machine running MicroShift, we will deploy an application that interacts with the battery monitoring system to optimize charging patterns at an EV charging station. This application will use another AI model to enhance charging efficiency and extend battery lifespan.

## Lab instructions

You can find the lab instructions in the following repository:

```
https://github.com/dialvare/showroom-ai-lifecycle-edge 
```
