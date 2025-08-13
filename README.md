**Author:** Sujal Gupta  

---

## 📌 Project Overview  
This project was developed as part of the **Vision AI Bootcamp**.  
Over the span of 5 days, I built and trained multiple image classification models using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**, starting from basic datasets like MNIST and CIFAR-10 to real-world binary classification (Cats vs Dogs).  

The final deliverable is a **portfolio-ready AI toolkit** capable of performing image recognition tasks with high accuracy.

---

## 🎯 Objectives  
- Learn image preprocessing and augmentation  
- Build custom CNN models from scratch  
- Apply transfer learning using MobileNetV2  
- Evaluate models using various metrics  
- Visualize results and performance  
- Create a deployable prediction tool for new images  

---

## 📂 Datasets Used  
1. **MNIST** – Handwritten digits dataset (Keras built-in)  
2. **CIFAR-10** – 10-class object dataset (Keras built-in)  
3. **Cats vs Dogs** – Binary classification dataset ([Kaggle Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog))  

---

## 🛠️ Tech Stack  
- **Language:** Python  
- **Frameworks & Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn  
- **Platform:** Google Colab (GPU enabled)  
- **Data Augmentation:** ImageDataGenerator  
- **Transfer Learning Model:** MobileNetV2 (ImageNet weights)  

---

## 📅 Daily Progress  

### **Day 01 – Image Recognition Basics**  
- Loaded MNIST dataset  
- Preprocessed and visualized sample images  

### **Day 02 – Basic CNN on MNIST**  
- Built a CNN with Conv2D, MaxPooling, Dense layers  
- Achieved high accuracy on digit classification  

### **Day 03 – CIFAR-10 + Augmentation**  
- Applied image augmentation (rotation, flipping, shifting)  
- Trained CNN to handle more complex images  
- Evaluated with a confusion matrix  

### **Day 04 – Transfer Learning (Cats vs Dogs)**  
- Downloaded dataset from Kaggle  
- Applied MobileNetV2 with frozen base layers  
- Fine-tuned classifier head for binary classification  
- Plotted ROC curve & calculated AUC score  

### **Day 05 – Final Prediction + Portfolio Visualization**  
- Uploaded a custom image and predicted the class (Dog 🐶 / Cat 🐱)  
- Plotted bar chart comparing accuracies across datasets  

---

## 📊 Results  

| Dataset        | Model Type             | Accuracy |
|----------------|------------------------|----------|
| MNIST          | Custom CNN              | ~99%     |
| CIFAR-10       | CNN + Augmentation      | ~80%     |
| Cats vs Dogs   | MobileNetV2 Transfer Learning | ~95%     |

---

## 📸 Sample Visualizations  
- Training Accuracy vs Validation Accuracy plots  
- Confusion Matrix for CIFAR-10  
- ROC curve for Cats vs Dogs model  
- Portfolio accuracy comparison bar chart  

---

## 🚀 How to Run the Project  

1. **Clone this repo**  
```bash    
git clone https://github.com/yourusername/vision-ai-5days.git
cd vision-ai-5days
```

2️⃣ Open in Google Colab
Upload the vision_ai_final.ipynb file to Google Colab

Enable GPU from Runtime → Change runtime type → GPU

3️⃣ Install dependencies
``` bash
!pip install kaggle tensorflow matplotlib seaborn scikit-learn
```

4️⃣ Setup Kaggle API (for Cats vs Dogs dataset)

Get your API key from Kaggle Account Settings → Create API Token

Upload kaggle.json to Colab:
```python
from google.colab import files files.upload()  # upload kaggle.json here
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
5️⃣ Run the notebook
Execute all cells sequentially from Day 01 to Day 05

On Day 05, upload a custom image to test predictions

📌 Notes
The Cats vs Dogs dataset download will fail without a valid Kaggle API key.

CIFAR-10 and MNIST are built into Keras and will download automatically.

Training times vary depending on GPU availability.

📜 License
This project is licensed under the MIT License. You are free to use, modify, and distribute it with attribution.

👏 Acknowledgements
Kaggle for providing datasets

TensorFlow/Keras for deep learning tools

Vision AI Bootcamp instructors for guidance


