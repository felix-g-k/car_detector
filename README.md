# Assignment 04: Car Image Viewpoint Prediction & Gradio App

## Training Summary

This assignment focuses on training a deep learning model to predict the viewpoint of car images (full frontal vs. not full frontal) and deploying it in an interactive Gradio app. The main steps were:

- **Dataset**: Images of cars with viewpoint labels (0° = full frontal, others = not full frontal). The dataset was balanced to ensure equal representation of both classes.
- **Model**: Transfer learning with a ResNet18 architecture. All layers except the last fully connected layer and layer4 were frozen. The final layer was adapted for binary classification.
- **Training**: The model was trained for up to 10 epochs (with early stopping) on 35,000 training images, validated on 10,000, and tested on 5,000. Data augmentations and ImageNet normalization were used.
- **Results**: The trained model achieved a test accuracy of **94.7%** on the binary viewpoint classification task.

## Gradio App Overview

The app, located in `PDL_ASS_IV/app.py`, provides an interactive interface for car image analysis. It performs the following steps:

1. **Car Detection & Segmentation**: Uses Detectron2's Mask R-CNN to detect and segment car instances in the uploaded image.
2. **Salient Car Selection**: Selects the largest detected car for further analysis.
3. **Viewpoint Prediction**: Uses the trained ResNet18 model to predict if the car is in a full frontal view. If not, the user is notified.
4. **Further Analysis (if frontal)**:
   - **Bodytype Prediction**: Classifies the car's bodytype (Hatchback, SUV, MPV, Convertible, Saloon).
   - **Modernity Score**: Predicts the car's year range (e.g., 2000-2003, 2015-2018).
   - **Typicality Score**: Computes how typical the car is for its predicted group using cosine similarity with group morphs.
5. **Visualization**: Shows the segmented car and the cropped car image.

## How to Use the App

### Requirements
Install dependencies (from `PDL_ASS_IV/requirements.txt`):

```
pip install torch torchvision numpy opencv-python gradio detectron2
```

### Running the App
1. Navigate to the `PDL_ASS_IV` directory:
   ```
   cd assignments/assignment_04/PDL_ASS_IV
   ```
2. Launch the Gradio app:
   ```
   python app.py
   ```
3. The app will open in your browser. Upload a car image or use one of the provided examples.

### Usage Notes
- The app works best with images containing at least one car. For full analysis, the car should be in a full frontal view.
- If no car is detected or the car is not frontal, the app will notify you.
- For frontal cars, the app predicts bodytype, modernity, and typicality.

### Example Images
Several example images are included in the `PDL_ASS_IV` directory and are available in the app interface.

---