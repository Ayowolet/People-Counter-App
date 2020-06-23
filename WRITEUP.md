# Project Write-Up

Computer vision models and applications have seen an increase in use in recent times. However, there are different frameworks being used in the building of these applications and having running these models on different hardwares sometimes results in an inability to effectively optimise them. It also requires expertise which might not be as popular as required.

This problem has been captured and solved by the OpenVino toolkit in that different models built with different frameworks can be optimized to run on the Intel hardware and without a high level of expertise. 

In this project, a tensorflow object detection model will be used on the OpenVino toolkit on an Intel run CPU.

## Setup

### Extracting the model

First, download the [ssd_mobilenet_v2_coco model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)

From command line/terminal, navigate to the directory where the model was downloaded and extract it using

```
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```

Go to the just extracted directory (ssd_mobilenet_v2_coco_2018_03_29) and generate the intermediate represention (IR) files i.e.(.xml and .bin) using the model optimizer by running the following

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel
```

Clone this repo and create a "files" directory

Copy the generated .xml and .bin file into the "files" directory

## Explaining Custom Layers

The model optimizer in the OpenVino toolkit is used to convert identified layers in a model to a corresponding internl representaion (IR) thereby optimizing the model and giving as an output, the IR files. Some devices, however, do not support all layers in a model and these are referred to as unsupported layers. To take care of these unsupported layers, custom layers are required.

The unsupported layers have to first be identified using query_network on the IECore class. Upon identification of the layers, custom layers are then added with a method known as the add_extension method. It is also important to ensure that the right type of deployment device and the library are used.

## Comparing Model Performance

My methods to compare models before and after conversion to Intermediate Representations were using the time and inference size of the model before and after conversion.

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- conversion was 70mb and post-conversion was 68mb

The inference time of the model pre-conversion was on average 180.8ms and post-conversion on average was 80ms.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are detecting the number of people who go into a store during a period of time . It can also be used to ascertain the number of people who use an ATM over a period of time. Another use will be to determine the number of people in a particular place at a given period.

Each of these use cases would be useful because in these covid-19 times, knowing the number of people in a particular place at a given time will enable the store owners be notified when the number is reaching a high point so entry can be limited. Also, in the case of the ATMs, it will be useful to know the average number of people who use an ATM at a particular loaction in order to be able to predict when to refill the ATMs or have the most money in the ATM. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows: the lightning is very crucial as this model will not be able to detect persons accurately when the lightning is poor. Also, the camera has to be in a place where the person can be easily detected. It will most likely not detect people from a very far distance.
