# Model Card for TAED2_SignAI

TAED2_SignAI is a computer vision model based on YOLOv8 for the detection and classification of traffic signs.

Given an input image, the model outputs bounding boxes together with confidence scores for one of the supported categories: prohibition, danger, mandatory or other.

<br>

## Model Details

### Model Description

- **Developed by:** Maria Poveda & Laia Villagrasa
- **Model type:** Object Detection Model (YOLOv8-based)
- **License:** *Educational use only (TAED2 project)*
- **Finetuned from model:** [Traffic Signs Detection using YOLOv8 (Kaggle notebook)](https://www.kaggle.com/code/diaakotb/traffic-signs-detection-using-yolov8)

<br>

## Uses

This model is designed for the detection of traffic signs. In particular, it identifies traffic signs in an image and classifies them into broader categories rather than specific sign types: prohibition, mandatory, danger or other.

Therefore, this task has a direct real-world impact: it supports progress toward autonomous driving, improves road safety and contributes to the reduction of accidents.

### Direct Use

The model can be directly used to detect and classify traffic signs in static images. By providing an input image, it outputs bounding boxes and confidence scores for one of the supported categories: prohibition, danger, mandatory or other.


### Out-of-Scope Use

* This model is not intended to identify the exact type of traffic sign, but only to classify it into general categories (prohibition, danger, mandatory, other).
* It is not suitable for deployment in production environments such as autonomous vehicles or traffic monitoring systems where safety-critical decisions depend on precise recognition.

<br>

## Bias, Risks, and Limitations

**BIAS**
* The model may show bias toward certain categories if they are overrepresented in the training dataset.
* It may also underperform on traffic signs that are less frequent or have fewer training examples.


**RISKS**
* Using the model in real-world traffic safety contexts could lead to false detections or misclassifications with serious consequences.
* Overreliance on the model without human validation may result in errors in decision-making.


**LIMITATIONS**
* The quality of the input camera or image directly affects the accuracy of the predictions.
* Environmental conditions such as lighting, weather or time of day may negatively impact detection.
* The presence of obstacles or occlusions can prevent correct classification of traffic signs.


### Recommendations

Due to the biases, risks, and limitations mentioned, it is recommended to validate the model’s predictions with human oversight in critical or safety-related environments, as well as to evaluate the model under different environmental conditions to better understand its limitations.

In addition, users are encouraged to expand and balance the training dataset to reduce potential class biases to improve robustness against lighting changes, occlusions and other real-world variations.

<br>

## How to Get Started with the Model

## Training Details

### Training Data  

The model was trained on the [Traffic Signs Dataset in YOLO format](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format), which contains traffic sign images annotated with bounding boxes and class labels suitable for object detection tasks. The dataset includes a variety of sign categories and follows the YOLO annotation format.  

For more information on the dataset design and collection, see the original [dataset paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608012000457).  


### Training Procedure

The model was fine-tuned from the existing YOLOv8m architecture (`yolov8m.yaml`).  



## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The evaluation is carried out using the validation/test split of the [Traffic Signs Dataset in YOLO format](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format).  
This dataset contains annotated images of traffic signs, with bounding boxes and class labels.  


#### Factors

Performance is evaluated across the main traffic sign categories defined in the dataset: prohibition, danger, mandatory and other.  However, the results may vary depending on:
* Sign frequency
* Sign visibility and occlusions
* Environmental conditions (lighting, weather).  


#### Metrics

The evaluation of the YOLOv8 model on the traffic sign detection task relies on the following metrics:

* **Precision:** Indicates the proportion of predicted traffic signs that are correct. A high precision score shows that when the model predicts a traffic sign, it is usually accurate.
* **Recall:** Represents the proportion of actual traffic signs that the model successfully detects. High recall means the model is able to find most of the signs present in the image, with few missed detections.
* **mAP@0.5:** The mean average precision computed at an IoU threshold of 0.5, reflecting how well predicted bounding boxes align with the ground truth. Higher values suggest accurate and well-localized detections.
* **mAP@0.5:0.95:** A more demanding metric that averages mAP across multiple IoU thresholds between 0.5 and 0.95. Strong performance here demonstrates the model’s robustness to stricter overlap requirements.




<br>

## Citation 

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). The German Traffic Sign Recognition Benchmark: A multi-class classification competition. *Neural Networks, 32,* 333–338. https://doi.org/10.1016/j.neunet.2012.02.016



## Model Card Authors 

* Villagrasa, Laia
* Poveda, Maria

