# Medical-Prescription-Recognition-System
Word Detection and Recognition from Scanned Prescription Images
Introduction
This project aims to automate the process of interpreting handwritten prescriptions by detecting and recognizing words from scanned images. It comprises two main components:

Word Detection: This phase involves the segmentation of individual words from the scanned prescription image using ResNet18, a convolutional neural network.
Word Recognition: After detecting words, this phase deciphers the actual meaning of each isolated word using a combination of Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and a Connectionist Temporal Classification (CTC) layer.
Project Structure
data/: Contains the dataset used for training and evaluation.
models/: Holds the implemented models for word detection and recognition.
scripts/: Includes preprocessing scripts, training scripts, and inference scripts.
results/: Stores the output results such as detected bounding boxes and recognized text.
README.md: Project documentation.
requirements.txt: Python dependencies.
Data Collection and Preprocessing
We use the IAM dataset, a well-known resource for handwritten text samples, to train and evaluate our model.

Preprocessing Steps:
Gray-scale Conversion: Convert images to gray-scale to encode light intensity information.
Normalization: Normalize pixel values to a range of 0 to 1 to ensure consistency across the dataset, which is crucial for effective model training.
Word Detection
Model Architecture
ResNet18: Used as the feature extractor. The model classifies each pixel to determine whether it belongs to the inner part of a word, the surrounding area, or the background.
Output Maps:
Segmentation Maps: Encode the classification of pixels.
Geometry Maps: Encode distances between each pixel and the edges of the predicted bounding box (AABB).
Training Details
Loss Functions:
Segmentation Loss: Cross-entropy loss for pixelwise classification.
Geometry Loss: Intersection over Union (IoU) metric for bounding box prediction.
Post-Processing
DBSCAN Clustering: Used to group predicted bounding boxes and accurately localize words within the image.
Word Recognition
Model Architecture
CNN Layers: Extract features from the input image through convolution, RELU activation, and pooling.
RNN Layers: Process the feature sequence with LSTM cells to capture dependencies.
CTC Layer: Decodes the output into the final predicted text.
Training and Inference
Training: The CTC layer computes the loss between the predicted output and the ground truth text.
Inference: The CTC layer decodes the output matrix into text.
