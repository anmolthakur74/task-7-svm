# Task 7: Support Vector Machines (SVM)

This repository contains my solution for **Task 7** of the AI & ML Internship. The task focuses on using **Support Vector Machines (SVM)** with both linear and RBF kernels on the **Heart Disease dataset** to classify whether a person is likely to have heart disease. It also includes visualization of decision boundaries using PCA.

---

## Objective

Use SVMs for linear and non-linear classification.

---

## Files Included

| File Name        | Description                                                          |
|------------------|----------------------------------------------------------------------|
| `heart.csv`      | Dataset used for binary classification                               |
| `svm_heart.ipynb`| Jupyter Notebook containing full implementation                      |
| `pca.png`        | Screenshot of SVM decision boundary after PCA dimensionality reduction |
| `README.md`      | Project documentation                                                |

---

## What I Did

### 1. Dataset Loading and Preparation  
I started by loading the `heart.csv` dataset which contains patient health data with features like age, cholesterol, blood pressure, and more. The target column indicates whether or not the person has heart disease (0 or 1).  
After verifying there were no missing values, I separated the features and labels. Since SVMs are sensitive to the scale of input features, I standardized all the features using `StandardScaler`. I then split the data into training and testing sets in an 80/20 ratio.

---

### 2. Training SVM Models  
To explore how different kernels perform, I trained two models using `SVC` from scikit-learn:  
- One with a **linear kernel**, which tries to separate the classes using a straight hyperplane  
- Another with an **RBF (Radial Basis Function) kernel**, which allows for non-linear decision boundaries  
I trained both on the scaled training data and evaluated them on the test data using accuracy scores.

---

### 3. Visualizing Decision Boundaries  
Since the dataset had 13 features, I used **PCA** to reduce the data to 2 dimensions for visualization. I retrained the RBF SVM on the 2D PCA data and plotted the decision boundary.  
The plot helped me visualize how the model separates the two classes, especially in regions where the data points are tightly packed and overlapping. It was interesting to see how the RBF kernel curved the boundary to adapt to the complex structure of the data.

---

### 4. Hyperparameter Tuning  
To get the best results, I used **GridSearchCV** to tune the `C` and `gamma` parameters for the RBF kernel. I tested different values across a grid and picked the combination that gave the highest cross-validation score.  
This showed me how important proper tuning is — some combinations worked far better than others.

---

### 5. Cross-Validation  
Finally, I used **5-fold cross-validation** to evaluate the overall performance of the best-tuned model. This helped me understand how the model performs on different subsets of the data, and whether it generalizes well beyond the train/test split.

---

## What I Learned

This task helped me get a hands-on understanding of how **SVMs** work for classification tasks. I learned:

- The **importance of feature scaling** before training an SVM, since it's a distance-based model.
- The difference between **linear and non-linear kernels**, and how the RBF kernel can handle complex, non-linear patterns better.
- How to **visualize decision boundaries** by reducing high-dimensional data using PCA — this made the inner workings of the model more interpretable.
- That **hyperparameter tuning** (especially `C` and `gamma` in RBF) can drastically impact model performance.
- How **cross-validation** gives a more robust estimate of model accuracy than just a single train-test split.

---

## Libraries Used

- `pandas`, `numpy` — data loading and preprocessing
- `matplotlib`, `seaborn` — visualization
- `sklearn` — SVM, PCA, scaling, model selection, evaluation

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/task-7-svm-heart.git
   cd task-7-svm-heart

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

4. Run the notebook:
   ```bash
   jupyter notebook svm_heart.ipynb

## Author

**Anmol Thakur**

GitHub: [anmolthakur74](https://github.com/anmolthakur74/)
