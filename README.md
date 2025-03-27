### 📝 **README: Breast Cancer Classification (Malignant vs. Benign) Using Logistic Regression (From Scratch)**

---

## 📌 **Project Title:**  
🎯 **Breast Cancer Classification: Malignant vs. Benign Tumor Prediction (From Scratch)**  

---

## 🔥 **Project Description**

This project involves building a **Breast Cancer Classification Model** from scratch using **Logistic Regression** to predict whether a tumor is **malignant** or **benign** based on real-world data.  

The model:  
- Uses the **Breast Cancer Wisconsin dataset** containing **30 numerical features** and **569 samples**.  
- Implements **Logistic Regression** using only **NumPy**, without relying on machine learning libraries like Scikit-Learn or TensorFlow.  
- Optimizes weights using **Gradient Descent**.  
- Uses **Cross-Entropy Loss** as the cost function.  
- Evaluates the model with **accuracy, precision, recall, and F1-score**.  
- Visualizes the **loss reduction over epochs**.  

---

## 🚀 **Features and Functionality**

✅ **Real-World Dataset:**  
- Utilizes the **Breast Cancer Wisconsin dataset** for realistic classification.  
- Contains 30 numerical features and a binary target label:  
    - `1` → **Malignant**  
    - `0` → **Benign**  

✅ **Model Implementation:**  
- **Logistic Regression from Scratch:** Built without pre-built ML libraries.  
- **Gradient Descent Optimization:** Implements manual weight updates.  
- **Cross-Entropy Loss:** Used as the cost function.  

✅ **Evaluation Metrics:**  
- **Accuracy:** Measures the percentage of correctly classified tumors.  
- **Precision:** Measures the proportion of correctly identified malignant tumors.  
- **Recall:** Measures how well the model identifies malignant cases.  
- **F1-Score:** Harmonic mean of precision and recall.  

✅ **Visualization:**  
- **Loss Curve:** Shows the reduction in cross-entropy loss over epochs.  
- **Decision Boundary:** Visualizes the model's decision-making ability.  

---

## 📊 **Technologies Used**

- **Python:** Core language used for the project.  
- **NumPy:** For matrix operations, weight updates, and cost calculations.  
- **Matplotlib:** For data visualization (loss reduction and predictions).  
- **Scikit-Learn:** Only used for dataset loading and data splitting.  

---

## 💡 **Model Architecture**

### ✅ **Data Preparation:**
1. **Load the Dataset:**  
   - Import the **Breast Cancer Wisconsin dataset** from Scikit-Learn.  
   - Extract features and target labels.  
2. **Normalize the Features:**  
   - Applied **StandardScaler** for feature normalization.  
   - Added a **bias term** (intercept) for logistic regression.  
3. **Split the Dataset:**  
   - **80-20 split**: 80% training, 20% testing.  

---

### ✅ **Model Implementation:**
1. **Logistic Regression from Scratch:**  
   - Implemented with **NumPy** without using pre-built ML models.  
2. **Sigmoid Function:**  
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]  
3. **Cross-Entropy Loss:**  
\[
\text{Loss} = - \frac{1}{m} \sum \left[ y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right]
\]  
4. **Gradient Descent:**  
\[
W = W - \alpha \cdot \nabla
\]  

---

## 📈 **Sample Output**

✅ **Training Loss Over Epochs**
```
Epoch 0, Loss: 0.6927  
Epoch 100, Loss: 0.3021  
Epoch 900, Loss: 0.2357  
```

✅ **Model Performance**
```
Final Accuracy: 97.37%  
Precision: 0.9870  
Recall: 0.9743  
F1-Score: 0.9806  
```
---

## 👨‍🏫 **Credits**
This project is inspired by the course:  
📚 **"Supervised Learning: Regression and Classification" by Andrew Ng** on **Coursera**.  
Link: [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning)  
