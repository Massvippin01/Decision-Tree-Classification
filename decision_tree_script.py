"""
DECISION TREE CLASSIFICATION - CODTECH INTERNSHIP TASK 1
=========================================================
A complete implementation of Decision Tree classifier with visualization and analysis
Dataset: Iris Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load and prepare the Iris dataset"""
    print("="*60)
    print("LOADING DATASET")
    print("="*60)
    
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    
    print(f"Dataset Shape: {df.shape}")
    print(f"\nClass Distribution:\n{df['species'].value_counts()}")
    
    return df, iris

def explore_data(df, iris):
    """Perform exploratory data analysis"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Distributions by Species', fontsize=16, fontweight='bold')
    
    features = iris.feature_names
    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        for species in iris.target_names:
            data = df[df['species'] == species][feature]
            ax.hist(data, alpha=0.6, label=species, bins=15)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature distribution plot saved as 'feature_distributions.png'")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[iris.feature_names].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation heatmap saved as 'correlation_heatmap.png'")

def prepare_data(df, iris):
    """Split data into training and testing sets"""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    X = df[iris.feature_names]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the Decision Tree classifier"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    dt_classifier = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    dt_classifier.fit(X_train, y_train)
    
    print("✓ Decision Tree Model trained successfully!")
    print(f"Tree Depth: {dt_classifier.get_depth()}")
    print(f"Number of Leaves: {dt_classifier.get_n_leaves()}")
    
    return dt_classifier

def evaluate_model(model, X_train, X_test, y_train, y_test, iris):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT")
    print("-"*60)
    print(classification_report(y_test, y_test_pred, target_names=iris.target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names,
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='white')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    return y_test_pred

def visualize_tree(model, iris):
    """Visualize the Decision Tree"""
    print("\n" + "="*60)
    print("TREE VISUALIZATION")
    print("="*60)
    
    plt.figure(figsize=(20, 12))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True
    )
    plt.title('Decision Tree Structure', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Decision tree visualization saved as 'decision_tree_visualization.png'")

def analyze_features(model, iris):
    """Analyze feature importance"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'Feature': iris.feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance Scores:")
    print(feature_importance)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
             color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance in Decision Tree', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance plot saved as 'feature_importance.png'")

def cross_validation_analysis(model, X, y):
    """Perform cross-validation"""
    print("\n" + "="*60)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print("\nCross-Validation Scores:")
    for fold, score in enumerate(cv_scores, 1):
        print(f"Fold {fold}: {score * 100:.2f}%")
    
    print(f"\nMean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
    print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 6), cv_scores * 100, marker='o', linewidth=2, markersize=10, color='#4ECDC4')
    plt.axhline(y=cv_scores.mean() * 100, color='red', linestyle='--', 
                label=f'Mean: {cv_scores.mean() * 100:.2f}%')
    plt.xlabel('Fold Number', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Cross-Validation Performance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 6))
    plt.tight_layout()
    plt.savefig('cross_validation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Cross-validation plot saved as 'cross_validation.png'")

def save_model(model):
    """Save the trained model"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    with open('decision_tree_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    print("✓ Model saved as 'decision_tree_model.pkl'")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("DECISION TREE CLASSIFICATION PROJECT")
    print("CODTECH INTERNSHIP - TASK 1")
    print("="*60)
    
    # Load data
    df, iris = load_data()
    
    # Explore data
    explore_data(df, iris)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, iris)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test, iris)
    
    # Visualize tree
    visualize_tree(model, iris)
    
    # Analyze features
    analyze_features(model, iris)
    
    # Cross-validation
    X = df[iris.feature_names]
    y = df['target']
    cross_validation_analysis(model, X, y)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nAll visualizations and model saved successfully!")
    print("\nGenerated Files:")
    print("1. feature_distributions.png")
    print("2. correlation_heatmap.png")
    print("3. confusion_matrix.png")
    print("4. decision_tree_visualization.png")
    print("5. feature_importance.png")
    print("6. cross_validation.png")
    print("7. decision_tree_model.pkl")

if __name__ == "__main__":
    main()
