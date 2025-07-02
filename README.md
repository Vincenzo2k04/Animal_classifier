
# 🐾 Animal Image Classification with TensorFlow

This project implements a deep learning model using Convolutional Neural Networks (CNNs) to classify images of animals into their respective categories. The model is built with TensorFlow and trained on a custom dataset using data augmentation and best practices for image classification.

---

## 📁 Dataset

- **Source**: The dataset is organized in a directory structure where each subfolder represents a class (animal type).
- **Structure**:
  ```
  dataset/
  ├── cat/
  ├── dog/
  ├── lion/
  └── ...
  ```
- The images are automatically split into training and validation sets using an 80-20 split.

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🧠 Model Architecture

```python
Sequential([
    Rescaling(1./255),
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

---

## 🏋️‍♂️ Training Strategy

- **Image Augmentation**: Includes flipping, zooming, contrast, and rotation to improve generalization.
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy

---

## 📈 Results

Training and validation accuracy and loss are plotted to monitor overfitting and model performance.

---

## 📊 How to Use

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/animal-image-classification.git
   cd animal-image-classification
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**
   Place your dataset inside the `dataset/` directory in the specified folder structure.

4. **Run the Notebook**
   Open `ANIMALS.ipynb` in Jupyter Notebook or VS Code and run all cells.

---

## 🔮 Future Improvements

- Add transfer learning (e.g., ResNet50, MobileNet)
- Hyperparameter tuning (with Keras Tuner or Optuna)
- Add a web app using Streamlit or Flask for real-time predictions

---
