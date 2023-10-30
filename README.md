<div align="center">
<h1 align="center">
<br>PNEUMONIA__DETECTION_INTELONEAPI</h1>
</div>
 



## DEDUCTION OF PNEUMONIA  USING CHEST X RAY IMAGES
This project focuses on developing an automated Pneumonia Detection system using deep learning and chest X-ray images. The system employs convolutional neural networks for accurate diagnosis, ensures compliance with medical regulations, provides an intuitive user interface for healthcare professionals, and offers insights into diagnostic decisions, contributing to early and effective treatment. The accuracy was 90% this was achived ony by intel oneAPI. oneAPI  has led to a remarkable enhancement in inference speed, with the average inference time being roughly half of what it was before . This points to a substantial improvement in the speed of inference. 🚀 Furthermore, the minimum and maximum inference times have seen a significant decrease post oneAPI implementation, contributing to a more consistent and expedited inference process.
___


## PROJECT FLOW
In this project, our initial model training and execution were conducted without harnessing the advantages of Intel's optimized libraries and the oneAPI toolkit. Although our accuracy was satisfactory, there was clear potential for enhancing the efficiency of our system. By seamlessly integrating the Intel-optimized TensorFlow model and deploying it using OpenVINO IR models, we not only maintained our accuracy levels but also witnessed a significant boost in efficiency. This optimization led to reduced runtime and effectively distributed the hardware load during the training process, resulting in a more streamlined and high-performance system. A pivotal component of this project was the utilization of the Intel oneAPI cloud, which provided a swift and finely-tuned environment, contributing to the successful completion of our objectives.
___



## Screenshots

Tensorflow runtime

<img src='openvinoIRruntime.PNG' alt='Tensorflowruntime.png'/>

OpenVino IR runtime

<img src='openvinoIRruntime.PNG' alt='Tensorflowruntime.png'/>

After completion of testing it clearly shows that openvino outperformed the tensorflow by runtime execution and performance.

___

## ONE API
OneAPI is a revolutionary, open, and standards-based programming model that liberates developers from the constraints of working with multiple hardware architectures, such as CPUs, GPUs, FPGAs, and diverse accelerators. With OneAPI, developers can utilize a single codebase across this spectrum, leading to accelerated computation while avoiding dependency on specific hardware vendors. This approach aligns with Intel's broader vision to establish a cohesive and adaptable software ecosystem tailored for the demands of high-performance computing and data analytics.
___

## OPEN VINO
OpenVINO, short for Open Visual Inference and Neural network Optimization, is a powerful open-source toolkit designed to enhance and streamline the deployment of AI inference. It offers a range of features and components aimed at optimizing deep learning performance in various applications, including computer vision, automatic speech recognition, natural language processing, and more.  The key highlights of OpenVINO:

Performance Enhancement: OpenVINO significantly boosts the performance of deep learning models, making them more efficient in tasks related to computer vision, speech recognition, and natural language processing. This acceleration is crucial for real-time and resource-constrained applications.

Framework Flexibility: It supports models trained with popular deep learning frameworks like TensorFlow, PyTorch, and others. This versatility allows developers to leverage their existing models and frameworks seamlessly.
___



## REPO STRUCTURE

```sh
└── PNEUMONIA__detection_IntelONEAPI/
    ├── model/
    │   ├── model.bin
    │   └── tf/
    │       ├── fingerprint.pb
    │       ├── keras_metadata.pb
    │       ├── saved_model.pb
    │       └── variables/
    ├── openvinoIRruntime.PNG
    ├── PNEUMONIA__detection_intelONEAPI.ipynb
    └── tensorflowruntime.PNG

```

____


## Installation

Clone the PNEUMONIA__detection_IntelONEAPI.ipynb repository:

```bash
  git clone
  https://github.com/aasima007/PNEUMONIA__detection_IntelONEAPI
```
Change to the project directory

```bash
cd PNEUMONIA__detection_IntelONEAPI
```
Install the dependencies:
```bash
!pip install opencv-python scikit-learn-intelex tensorflow==2.13 pandas openvino-dev numpy pandas matplotlib
```
 Running PNEUMONIA__detection_IntelONEAPI
 ```bash
jupyter nbconvert --execute notebook.ipynb
```

___


## 🚀 Getting Started

***Dependencies***

Dependencies for the project:  

`- ℹ️ OPENVINO`

`- ℹ️ OpenCV`

`- ℹ️ Tensorflow`

`- ℹ️ Sklearn(intel optimsed version of scikit-learn libraires)`

`- ℹ️ numpy`

`- ℹ️ pandas`

`- ℹ️ matplotlib`
