## Plant Disease Classification using CNN
_This project implements a Convolutional Neural Network (CNN) model to classify plant leaf images into different disease categories. Early and accurate disease detection in plants is critical for crop management and agricultural productivity, making this model practically impactful._

#### Dataset Details:
- Source: Kaggle(http://kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Total Images: ~54,000
-Image Format: RGB images in .jpg format

#### ML Workflow: 
1. Import Libraries: 
    1. TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn
2. Loading Dataset 
    1. Used the PlantVillage Dataset containing thousands of labeled plant leaf images.
3. Image Preprocessing 
    1. Resized images, normalized pixel values, and applied data augmentation to improve model generalization.
4. Train–Test Split
    1. Divided data into training and validation sets. 
5. Model Architecture: 
    1. Conv2D Layer with 32 filters (3×3) and ReLU activation for feature extraction
    2. MaxPooling2D (2×2) to reduce spatial dimensions
    3. Conv2D Layer with 64 filters (3×3) and ReLU activation for deeper feature extraction
    4. MaxPooling2D (2×2) to downsample feature maps
    5. Flatten to convert 2D features into a 1D vector
    6. Dense Layer with 256 units and ReLU activation for high-level representation
    7. Output Layer (Dense) with num_classes units and Softmax activation for multi-class classification
7. Model Training
    1. Trained the CNN for 5 epochs using training and validation 
    2. Epochs: 5
    3. Batch Size: Defined by generator
    4. Optimizer: Adam (default params)
    5. Loss: Categorical crossentropy
    6. Validation: Performed using validation generatorgenerators.
8. Model Evaluation
    1. Assessed model performance on validation data.
9. Predictive System
    1. Developed a system to classify unseen leaf images into disease categories.

#### Frameworks:
- Tensorflow and Keras

#### Results: 
- Accuracy: 0.97
- Loss: 0.06
- Validation Accuracy: 0.87

#### Model Summary:

<img width="733" height="553" alt="Screenshot 2025-09-05 at 6 31 44 PM" src="https://github.com/user-attachments/assets/18b3a1ca-84c4-4f11-9336-898c0ecd1fc9" />

#### Visualizations:
Accuracy: 

<img width="614" height="458" alt="Screenshot 2025-09-05 at 6 33 10 PM" src="https://github.com/user-attachments/assets/d91b8157-805b-4ac0-990f-aba0141871ab" />

Loss:

<img width="606" height="466" alt="Screenshot 2025-09-05 at 6 33 19 PM" src="https://github.com/user-attachments/assets/18df5490-7d3b-4567-a5ad-585b7d8285d6" />


#### Assumptions
1. Assumed that all input images are resized to a fixed dimension

#### Improvements
1. Increasing epochs and augmentations could further improve performance.

#### Key Observation
1. Leaf texture and color are the strongest indicators of plant disease, and CNNs capture these features effectively for accurate classification.

