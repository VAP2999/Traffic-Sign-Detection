# Traffic-Sign-Detection

This project leverages YOLOv5 to detect traffic signs in images and videos. The model is trained on a custom dataset consisting of traffic sign images annotated in YOLO format. The trained model can then be used for detection tasks on new images and videos.

## Overview

This repository demonstrates the following:
1. **Dataset Preparation**: The traffic sign dataset is organized into training and validation sets.
2. **Training**: The YOLOv5 model is trained on this dataset using custom annotations for traffic signs.
3. **Detection**: The trained model is used to perform detection on new images or videos.
4. **Video Detection**: The trained model can also be used to detect traffic signs in video files.

## Features

- Custom traffic sign detection model using YOLOv5.
- Dataset preprocessing for YOLOv5 compatibility.
- Model training on custom traffic sign dataset with labels for various traffic sign classes.
- Inference on both static images and dynamic videos.
- Easy integration with YOLOv5 for detection tasks.

## Technologies Used

- **Python 3.x**
- **YOLOv5** (Ultralytics)
- **PyTorch** for deep learning model training
- **OpenCV** for image and video processing
- **WandB** for experiment tracking (disabled in the notebook)
- **Google Drive Downloader** for dataset retrieval
- **NumPy, Pandas, Matplotlib** for data manipulation and visualization

## Setup

### 1. Clone the Repository
Clone the repository to get started with the project


### 2. Install Dependencies
Ensure that you have Python 3.x installed


### 3. Dataset Preparation
The dataset must be structured in YOLO format


The images are in `.jpg` or `.png` format, and the labels are in `.txt` format where each line represents a bounding box and the corresponding class.

The notebook automatically splits the dataset into training and validation sets based on a provided ratio (80% for training and 20% for validation).


Ensure the dataset YAML file (`dataset.yaml`) contains paths to the training and validation image and label directories.

### 4. Training the YOLOv5 Model
Once the dataset is ready, you can begin training the YOLOv5 model with the following command:
```bash
!python train.py --img 415 --batch 16 --epochs 30 --data /path/to/dataset.yaml --weights yolov5s.pt --cache --workers 2
```
- `--img 415`: Image size for training.
- `--batch 16`: Batch size.
- `--epochs 30`: Number of epochs to train the model.
- `--data /path/to/dataset.yaml`: Path to your dataset YAML configuration file.
- `--weights yolov5s.pt`: Pre-trained YOLOv5 weights to fine-tune on your custom dataset.
- `--cache --workers 2`: Caching and parallel data loading options for improved training speed.

### 5. Detection Inference on Images
After training, you can use the trained model to detect traffic signs in images. For example:
```bash
!python detect.py --source /path/to/image.jpg --weights runs/train/exp/weights/best.pt
```
This will perform detection on the image and save the resulting image with bounding boxes drawn around detected traffic signs.

### 8. Detection Inference on Videos
You can also apply the model to detect traffic signs in videos:
```bash
!python detect.py --source /path/to/video.mp4 --weights runs/train/exp/weights/best.pt
```
This will process the video frame by frame, detect traffic signs, and save the resulting video with annotations.



## Contributing

If you'd like to contribute to this project, please feel free to fork the repository and submit a pull request. If you find any bugs or have feature requests, please open an issue.
