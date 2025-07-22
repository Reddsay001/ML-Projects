
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import numpy as np

pipelines = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='l2',
            C=0.7,
            solver='liblinear',
            max_iter=300,
            class_weight='balanced'
        ))
    ]),
    'DecisionTree': Pipeline([
        ('clf', DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=None,
            class_weight='balanced'
        ))
    ]),
    'RandomForest': Pipeline([
        ('clf', RandomForestClassifier(
            n_estimators=150,
            criterion='gini',
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        ))
    ]),
    'GradientBoosting': Pipeline([
        ('clf', GradientBoostingClassifier(
            loss='log_loss',
            learning_rate=0.08,
            n_estimators=120,
            subsample=0.9,
            criterion='friedman_mse',
            max_depth=4,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt'
        ))
    ]),
    'AdaBoost': Pipeline([
        ('clf', AdaBoostClassifier(
            n_estimators=90,
            learning_rate=0.6,
            algorithm='SAMME.R',
            random_state=42
        ))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            C=0.9,
            kernel='rbf',
            gamma='auto',
            probability=True,
            class_weight='balanced'
        ))
    ]),
    'NaiveBayes': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB(
            var_smoothing=1e-8
        ))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski'
        ))
    ]),
    'XGBoost': Pipeline([
        ('clf', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            learning_rate=0.07,
            max_depth=6,
            n_estimators=110,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.5,
            gamma=0.1,
            scale_pos_weight=1
        ))
    ])
}

def evaluate_model(name, model, X_test, y_test, plot_roc=True, plot_pr=True):
    y_pred = model.predict(X_test)

    # Predict probabilities if supported
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None

    print(f"
===== {name} =====")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:   ", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score: ", f1_score(y_test, y_pred, zero_division=0))
    if y_proba is not None:
        print("ROC AUC:  ", roc_auc_score(y_test, y_proba))
        print("Avg Precision (PR AUC):", average_precision_score(y_test, y_proba))

    print("
Classification Report:
", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:
", confusion_matrix(y_test, y_pred))

    if y_proba is not None and plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'{name} (ROC AUC = {roc_auc_score(y_test, y_proba):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    if y_proba is not None and plot_pr:
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f'{name} (PR AUC = {average_precision_score(y_test, y_proba):.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {name}')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.show()

# Training and evaluating the models
for name, pipeline in pipelines.items():
    print(f"
{'='*30}
Training model: {name}
{'='*30}")
    pipeline.fit(X_train, y_train)
    evaluate_model(name, pipeline, X_test, y_test, plot_roc=True, plot_pr=True)
