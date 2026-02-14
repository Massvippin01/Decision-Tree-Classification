# 🌳 DECISION TREE CLASSIFICATION PROJECT
## CODTECH Internship - Task 1

---

## 📋 PROJECT OVERVIEW

This project implements a **Decision Tree Classifier** using scikit-learn to classify iris flower species based on their physical measurements. The project includes comprehensive visualizations, model evaluation, and analysis.

**Dataset:** Iris Dataset (150 samples, 4 features, 3 classes)

**Objective:** Build and visualize a Decision Tree model with detailed analysis

---

## 📁 PROJECT STRUCTURE

```
Decision_Tree_Project/
│
├── Decision_Tree_Classification.ipynb    # Main Jupyter notebook
├── decision_tree_script.py               # Standalone Python script
├── LEARNING_GUIDE.md                     # Comprehensive learning guide
├── README.md                             # This file
│
└── Generated Output Files:
    ├── feature_distributions.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── decision_tree_visualization.png
    ├── feature_importance.png
    ├── cross_validation.png
    └── decision_tree_model.pkl
```

---

## 🚀 QUICK START

### **Option 1: Run Jupyter Notebook** (Recommended)

1. **Install required libraries:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

2. **Launch Jupyter:**
```bash
jupyter notebook
```

3. **Open the notebook:**
   - Navigate to `Decision_Tree_Classification.ipynb`
   - Run all cells: `Cell > Run All`

### **Option 2: Run Python Script**

1. **Install required libraries:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. **Run the script:**
```bash
python decision_tree_script.py
```

3. **Output:**
   - Console shows all metrics and analysis
   - 7 visualization files saved in current directory
   - Model saved as `decision_tree_model.pkl`

---

## 📦 DEPENDENCIES

### **Required Libraries:**

```python
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0  # For notebook only
```

### **Installation:**

```bash
# Install all at once
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Or use requirements.txt
pip install -r requirements.txt
```

---

## 🎯 WHAT THE PROJECT DOES

### **1. Data Loading & Exploration**
- Loads the Iris dataset
- Displays dataset statistics
- Shows class distribution

### **2. Exploratory Data Analysis**
- Feature distribution plots
- Correlation heatmap
- Pairwise relationships
- Statistical summaries

### **3. Data Preparation**
- Train-test split (80-20)
- Stratified sampling
- Feature-target separation

### **4. Model Training**
- Decision Tree classifier
- Optimized hyperparameters
- Gini impurity criterion

### **5. Model Evaluation**
- Accuracy scores
- Classification report
- Confusion matrix
- Precision, Recall, F1-score

### **6. Visualizations**
- Complete decision tree structure
- Feature importance chart
- Cross-validation performance
- Multiple comparison plots

### **7. Model Persistence**
- Saves trained model
- Enables future predictions
- Model reusability

---

## 📊 EXPECTED OUTPUT

### **Console Output:**

```
====================================
LOADING DATASET
====================================
Dataset Shape: (150, 5)

Class Distribution:
setosa        50
versicolor    50
virginica     50

====================================
MODEL TRAINING
====================================
✓ Decision Tree Model trained successfully!
Tree Depth: 4
Number of Leaves: 9

====================================
MODEL EVALUATION
====================================
Training Accuracy: 98.33%
Testing Accuracy: 96.67%

CLASSIFICATION REPORT
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      1.00      0.95        10
   virginica       1.00      0.90      0.95        10

    accuracy                           0.97        30
```

### **Visualizations Generated:**

1. **feature_distributions.png**
   - Histogram for each feature by species
   - Shows class separability

2. **correlation_heatmap.png**
   - Feature correlation matrix
   - Identifies redundant features

3. **confusion_matrix.png**
   - Prediction accuracy by class
   - Shows misclassification patterns

4. **decision_tree_visualization.png**
   - Complete tree structure
   - Decision rules at each node

5. **feature_importance.png**
   - Bar chart of feature importance
   - Identifies key predictors

6. **cross_validation.png**
   - 5-fold CV performance
   - Model stability analysis

---

## 🔧 CUSTOMIZATION OPTIONS

### **Adjust Tree Complexity:**

```python
# Simpler tree (less overfitting)
DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)

# More complex tree (might overfit)
DecisionTreeClassifier(max_depth=10, min_samples_leaf=1)
```

### **Change Split Criterion:**

```python
# Use entropy instead of Gini
criterion='entropy'
```

### **Modify Train-Test Split:**

```python
# 70-30 split instead of 80-20
test_size=0.3
```

### **Use Different Dataset:**

```python
from sklearn.datasets import load_wine, load_breast_cancer

# Wine dataset
data = load_wine()

# Breast cancer dataset
data = load_breast_cancer()
```

---

## 📈 MODEL PERFORMANCE METRICS

### **What Each Metric Means:**

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Accuracy** | Overall correctness | > 90% |
| **Precision** | Positive prediction accuracy | > 90% |
| **Recall** | Actual positive detection rate | > 90% |
| **F1-Score** | Balance of precision & recall | > 90% |
| **CV Score** | Cross-validation average | > 90% |

### **Interpreting Results:**

✅ **Good Model:**
- Train & test accuracy within 5%
- All metrics > 90%
- Consistent CV scores

⚠️ **Warning Signs:**
- Train accuracy >> Test accuracy (overfitting)
- Low precision or recall
- High CV variance

---

## 🐛 TROUBLESHOOTING

### **Issue: ModuleNotFoundError**

```bash
# Solution: Install missing library
pip install [missing_library_name]
```

### **Issue: Plots not displaying**

```python
# Add at the start of notebook/script
%matplotlib inline  # For Jupyter
plt.show()          # For scripts
```

### **Issue: Low accuracy**

```python
# Try adjusting hyperparameters
max_depth=5  # Increase complexity
min_samples_split=5  # Decrease minimum split samples
```

### **Issue: File save errors**

```bash
# Ensure write permissions
chmod +w .

# Check available disk space
df -h
```

---

## 📝 DELIVERABLES FOR SUBMISSION

### **Required Files:**

1. ✅ **Decision_Tree_Classification.ipynb** - Main notebook
2. ✅ **All visualizations** (6 PNG files)
3. ✅ **decision_tree_model.pkl** - Saved model
4. ✅ **This README.md** - Documentation

### **Submission Checklist:**

- [ ] All code cells executed successfully
- [ ] All visualizations saved
- [ ] Model performance documented
- [ ] Comments added to code
- [ ] Results interpreted
- [ ] README included

---

## 🎓 LEARNING RESOURCES

### **Included in This Project:**

1. **LEARNING_GUIDE.md** - Comprehensive explanations
2. **Code comments** - Inline explanations
3. **Visualizations** - Visual understanding

### **External Resources:**

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Decision Trees Explained](https://www.datacamp.com/tutorial/decision-tree-classification-python)
- [Understanding Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 💡 NEXT STEPS

After completing this project, try:

1. **Different Algorithms:**
   - Random Forest
   - Gradient Boosting
   - Support Vector Machines

2. **Hyperparameter Tuning:**
   - GridSearchCV
   - RandomizedSearchCV

3. **Other Datasets:**
   - Wine quality
   - Diabetes
   - Breast cancer

4. **Advanced Techniques:**
   - Feature engineering
   - Ensemble methods
   - Model stacking

---

## 🤝 SUPPORT

If you encounter issues:

1. Check the **LEARNING_GUIDE.md** for explanations
2. Review the **Troubleshooting** section above
3. Verify all dependencies are installed
4. Ensure you're using Python 3.7 or higher

---

## 📄 LICENSE

This project is created for educational purposes as part of the CODTECH Internship program.

---

## ✨ ACKNOWLEDGMENTS

- **Dataset:** Iris dataset from UCI Machine Learning Repository
- **Framework:** Scikit-learn
- **Visualization:** Matplotlib & Seaborn

---

## 📞 PROJECT INFO

**Project:** Decision Tree Classification
**Task:** CODTECH Internship Task 1
**Deliverable:** Notebook with model visualization and analysis
**Status:** ✅ Complete

---

**Happy Learning! 🚀**

*Remember: The best way to learn is by doing. Experiment, break things, and learn from errors!*
