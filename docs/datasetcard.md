---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

# Dataset Card for {{ GTSRB | default("Dataset Name", true) }}

German Traffic Signal Recognition Benchmark (GTRSB) is a dataset commonly used in image classification tasks and contains images of Traffic signals in various real-life conditions.

## Dataset Details

### Dataset Description

The German Traffic Sign Recognition Benchmark (GTSRB) is a dataset containing images of traffic signs from Germany. It was developed in 2011 by the Institute of Neural Information Processing at Ruhr-University of Bochum in the context of the International Joint Conference on Neural Networks (IJCNN). It is considered a reference benchmark for the identification and classification of traffic signals.

In the context of this project, the dataset is used for detection and classification of traffic signals into a set of four specific categories.

- **Curated by:** Institute of Neural Information Processing, Ruhr-University Bochum
- **Funded by:** [More Information Needed]
- **Shared by:** Avalible on Kaggle through https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- **Language(s) (NLP):** Not applicable (dataset contains images and coordinates exclusively)
- **License:** [More Information Needed]

### Dataset Sources

The dataset is accessible through Kaggle 
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

- **Repository:** (https://benchmark.ini.rub.de/gtsrb_news.html)
- **Paper:** "The German Traffic Sign Recognition Benchmark: A multi-class classification competition" (IJCNN 2011) Availble at: https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf 
- **Demo:** [More Information Needed]

## Uses

The intended use of this dataset in this project is the detection, recognition and classification of traffic signs specifically in four categories. 

### Direct Use

The dataset is intended for machine learning applications focused on computer vision, specifically for the detection, recognition, and classification of traffic signs. It is widely used in academic and applied research as a benchmark for evaluating model performance in traffic sign detection systems.

### Out-of-Scope Use

The dataset should not be assumed to generalize to traffic signs outside of Germany. It is not suitable for tasks requiring global applicability without retraining or fine-tuning on additional region-specific datasets.

## Dataset Structure

The dataset contains more than 50,000 images across 43 traffic sign classes. Images are provided at varying resolutions and include conditions such as occlusion, variable lighting, and different viewpoints.

For this project, a subset of 1,000 images is selected and grouped into 4 broader categories of interest:

* Prohibitionary
* Danger
* Mandatory
* Others

Each image is labeled with an associated text file containing bounding box coordinates and class information in a YOLO-compatible format.

## Dataset Creation

### Curation Rationale

The dataset was created to provide a standardized benchmark for evaluating computer vision models on the task of traffic sign recognition. It addresses the need for diverse, real-world images of traffic signs to advance research in autonomous driving and driver assistance systems.

### Source Data

The data consists of images with an associated text file containing coordinates for a bounding box and a class. Compatible with a YOLO model

#### Data Collection and Processing

The dataset was collected from real-world driving scenarios in Germany. Images were preprocessed and labeled to ensure class consistency. For use in this project, annotations are adapted into a format compatible with YOLO models.

#### Who are the source data producers?

The images were originally captured and labeled by the Institute of Neural Information Processing at Ruhr-University Bochum.

### Annotations

Annotations are included in linked files and include bounding box coordinates and class per each image

#### Annotation process

Annotations include bounding boxes and class labels corresponding to each traffic sign. In the Kaggle distribution, annotations are provided as text files compatible with YOLO.

#### Who are the annotators?

The annotation process was conducted by the original dataset creators (Ruhr-University Bochum team).

#### Personal and Sensitive Information

The dataset does not contain personal or sensitive information. Images are strictly limited to traffic signs and their immediate surroundings.

## Bias, Risks, and Limitations

The dataset presents certain risks, biases and limitations:

* Regional bias: All images represent German traffic signs, making direct application in other countries problematic without retraining.
* Environmental bias: Images are subject to German weather and lighting conditions. Extreme scenarios such as heavy rain, snow, or nighttime driving are underrepresented.
* Class inbalance: Some classes are more represented than others, which may cause skewed model performance if not addressed through augmentation or balancing techniques.

### Recommendations

Users should be aware of these limitations when deploying models trained exclusively on GTSRB. For broader applicability, additional datasets from other regions and conditions should be included.

## Citation

**BibTeX:**

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011). The German Traffic Sign Recognition Benchmark: A multi-class classification competition. Proceedings of the IEEE International Joint Conference on Neural Networks.

**APA:**

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011). The German Traffic Sign Recognition Benchmark: A multi-class classification competition. Proceedings of the IEEE International Joint Conference on Neural Networks.

## Dataset Card Authors  

* Poveda, Maria
* Villagrasa, Laia

## Dataset Card Contact

[More Information Needed]
