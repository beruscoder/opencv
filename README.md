Brain Tumor Segmentation using U-Net with Dice Coefficient of 89.6%
Overview
This repository contains a Jupyter notebook (brain-tumor-segmentation-unet-dice-coef-89-6.ipynb) that demonstrates the implementation of a U-Net model for brain tumor segmentation. The model achieves a Dice coefficient of 89.6%, indicating high accuracy in segmenting tumor regions from MRI images.

Key Features
U-Net Architecture: The notebook implements a U-Net model, a convolutional neural network designed for biomedical image segmentation.

Data Handling: Includes functions for loading and preprocessing brain tumor images and masks.

Data Augmentation: Uses ImageDataGenerator for augmenting training data to improve model generalization.

Metrics: Implements custom metrics such as Dice coefficient, Dice loss, and IoU (Intersection over Union) for model evaluation.

Visualization: Provides functions to display training history and sample images with segmentation masks.

Requirements
Python 3.x

TensorFlow 2.x

Keras

OpenCV

NumPy

Pandas

Matplotlib

scikit-image

scikit-learn

Usage
Clone the Repository:

bash
Copy
git clone https://github.com/your-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Run the Notebook:

Open the Jupyter notebook brain-tumor-segmentation-unet-dice-coef-89-6.ipynb.

Execute the cells sequentially to train the U-Net model and evaluate its performance.

Dataset
The notebook assumes the dataset is structured in a specific directory format. Ensure your dataset includes:

Images: Stored in a subdirectory with the naming convention *_image.png.

Masks: Stored in a corresponding subdirectory with the naming convention *_mask.png.

Example structure:

Copy
data/
  ├── train/
  │   ├── image1_image.png
  │   ├── image1_mask.png
  │   ├── image2_image.png
  │   └── image2_mask.png
  ├── valid/
  └── test/
Model Training
The notebook includes:

Data loading and preprocessing.

U-Net model definition and compilation.

Training with early stopping and model checkpointing.

Evaluation using Dice coefficient and IoU.

Results
Dice Coefficient: 89.6%
![Screenshot from 2025-04-07 01-50-06](https://github.com/user-attachments/assets/ba074209-aefb-434b-ac36-4a74c5dd58c8)
![Screenshot from 2025-04-07 01-49-51](https://github.com/user-attachments/assets/d1b4f17c-6ec7-4a32-b95c-a9dac5823c9b)
![Screenshot from 2025-04-07 01-50-59](https://github.com/user-attachments/assets/cb565eb0-3a5e-4739-8a56-4b99eef00892)

IoU (Intersection over Union): Included in the evaluation metrics.

Training history plots (accuracy, loss, Dice coefficient, and IoU) are displayed for analysis.

Customization
Adjust hyperparameters such as batch size, learning rate, and number of epochs in the notebook.

Modify the U-Net architecture (e.g., number of filters, depth) to experiment with performance.

License
This project is open-source. Feel free to use and modify the code for your purposes.

Acknowledgments
The U-Net architecture was originally proposed by Olaf Ronneberger et al. for biomedical image segmentation.

The dataset used in this project is not included; users must provide their own or use a public dataset like BraTS.
