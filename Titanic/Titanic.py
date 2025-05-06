"""
Titanic Survival Prediction with Encrypted Machine Learning
==========================================================
This script demonstrates machine learning on encrypted data using the TenSEAL library.
It implements both Logistic Regression and KNN classifiers on:
1. Plain data
2. Encrypted data with 4096 polynomial degree
3. Encrypted data with 8192 polynomial degree

The script uses the Titanic dataset to predict passenger survival.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import tenseal as ts
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# PART 1: DATA LOADING AND PREPROCESSING
# ======================================

def load_data(train_path, test_path):
    """Load the Titanic dataset."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def preprocess_data(df_train, df_test):
    """
    Preprocess and engineer features for the Titanic dataset.
    Includes handling missing values, creating new features, and encoding.
    """
    # Handle Cabin feature
    df_train['Cabin'].fillna('N0', inplace=True)
    df_train['Cabin'] = df_train['Cabin'].str.extract(r'([A-Za-z]+)')
    
    df_test['Cabin'].fillna('N0', inplace=True)
    df_test['Cabin'] = df_train['Cabin'].str.extract(r'([A-Za-z]+)')
    
    # Create Family_Size feature
    df_train["Family_Size"] = df_train["SibSp"] + df_train["Parch"]
    df_test["Family_Size"] = df_test["SibSp"] + df_test["Parch"]
    
    df_train["Family_Size"] = df_train["Family_Size"].apply(lambda x: 6 if x > 6 else x)
    df_test["Family_Size"] = df_test["Family_Size"].apply(lambda x: 6 if x > 6 else x)
    
    # Create Age bins
    age_bins = [0, 7, 18, 30, 50, 120]
    age_labels = ['Young Child', 'Child', 'Young Adult', 'Adult', 'Senior']
    
    df_train['Age_Bin'] = pd.cut(df_train['Age'], bins=age_bins, labels=age_labels, right=False)
    df_train['Age_Bin'] = df_train['Age_Bin'].cat.add_categories('Unknown').fillna('Unknown')
    df_train.drop('Age', axis=1, inplace=True)
    
    df_test['Age_Bin'] = pd.cut(df_test['Age'], bins=age_bins, labels=age_labels, right=False)
    df_test['Age_Bin'] = df_test['Age_Bin'].cat.add_categories('Unknown').fillna('Unknown')
    df_test.drop('Age', axis=1, inplace=True)
    
    # Extract titles from names
    title_mapping = {
        'Sir': 'Mr', 'Jonkheer': 'Mr', 'Countess': 'Mrs', 'Lady': 'Mrs', 'Mlle': 'Miss', 'Ms': 'Mrs',
        'Capt': 'Other', 'Mme': 'Mrs', 'Don': 'Other', 'Major': 'Other', 'Col': 'Other', 'Dona': 'Mrs',
        'Rev': 'Other', 'Dr': 'Other', 'Master': 'Other'
    }
    
    df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\.')
    df_train['Title'].replace(title_mapping, inplace=True)
    df_train.drop('Name', axis=1, inplace=True)
    
    df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\.')
    df_test['Title'].replace(title_mapping, inplace=True)
    df_test.drop('Name', axis=1, inplace=True)
    
    # Create Fare bins
    fare_bins = [-1, 25, 50, 100, np.inf]
    labels_fare = ['0-25', '25-50', '50-100', '100+']
    
    df_train['Fare_bin'] = pd.cut(df_train['Fare'], bins=fare_bins, labels=labels_fare)
    df_train['Fare_bin'] = df_train['Fare_bin'].cat.add_categories('Unknown').fillna('Unknown')
    df_train.drop('Fare', axis=1, inplace=True)
    
    df_test['Fare_bin'] = pd.cut(df_test['Fare'], bins=fare_bins, labels=labels_fare)
    df_test['Fare_bin'] = df_test['Fare_bin'].cat.add_categories('Unknown').fillna('Unknown')
    df_test.drop('Fare', axis=1, inplace=True)
    
    # Create isAlone feature
    df_train['isAlone'] = (df_train['Family_Size'] == 0).astype(int)
    df_test['isAlone'] = (df_test['Family_Size'] == 0).astype(int)
    
    # Drop unnecessary columns
    dropped_column = ['PassengerId', 'Ticket']
    df_train.drop(dropped_column, axis=1, inplace=True)
    df_test_new = df_test.drop(dropped_column, axis=1, inplace=False)
    
    # Handle missing Embarked values
    df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
    df_test_new['Embarked'].fillna(df_test_new['Embarked'].mode()[0], inplace=True)
    
    # Encode Sex feature
    df_train['Sex'] = (df_train['Sex'] == 'male')
    df_test_new['Sex'] = (df_test_new['Sex'] == 'male')
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    df_train['Pclass'] = label_encoder.fit_transform(df_train['Pclass'])
    df_train['Age_Bin'] = label_encoder.fit_transform(df_train['Age_Bin'])
    df_train['Fare_bin'] = label_encoder.fit_transform(df_train['Fare_bin'])
    
    df_test_new['Pclass'] = label_encoder.fit_transform(df_test_new['Pclass'])
    df_test_new['Age_Bin'] = label_encoder.fit_transform(df_test_new['Age_Bin'])
    df_test_new['Fare_bin'] = label_encoder.fit_transform(df_test_new['Fare_bin'])
    
    # One-hot encode remaining categorical features
    df_train = pd.get_dummies(df_train, columns=['Cabin', 'Embarked', 'Title'], prefix=['Cabin', 'Embarked', 'Title'])
    df_test_new = pd.get_dummies(df_test_new, columns=['Cabin', 'Embarked', 'Title'], prefix=['Cabin', 'Embarked', 'Title'])
    
    # Create interaction features
    df_train['Age_Pclass'] = df_train['Age_Bin'] * df_train['Pclass']
    df_train['isAlone_Age'] = df_train['isAlone'] * df_train['Age_Bin']
    df_train['isAlone_Sex'] = df_train['isAlone'] * df_train['Sex']
    
    df_test_new['Age_Pclass'] = df_test_new['Age_Bin'] * df_test_new['Pclass']
    df_test_new['isAlone_Age'] = df_test_new['isAlone'] * df_test_new['Age_Bin']
    df_test_new['isAlone_Sex'] = df_test_new['isAlone'] * df_test_new['Sex']
    
    return df_train, df_test_new

def prepare_data_for_modeling(df_train):
    """
    Split the data into features and target, and perform preprocessing steps for modeling.
    """
    # Split data into features and target
    X = df_train.drop('Survived', axis=1)
    y = df_train['Survived']
    
    # Apply SMOTE for class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X, y = smote.fit_resample(X, y)
    
    # Encode target and standardize features
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


# PART 2: LOGISTIC REGRESSION MODELS
# =================================

class LR(nn.Module):
    """Logistic Regression model using PyTorch."""
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.lr(x))

def train_lr_model(model, optimizer, criterion, x_train, y_train, epochs=200):
    """Train a logistic regression model."""
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
    return model

def evaluate_plain_lr(model, x_test, y_test):
    """Evaluate a plain (non-encrypted) logistic regression model."""
    model.eval()
    with torch.no_grad():
        t_start = time.time()
        out = model(x_test)
        preds = (out > 0.5).float()
        t_end = time.time()
        correct = (preds == y_test).float().sum()
        accuracy = correct / len(y_test)
        print(f"Plain Accuracy (LR): {accuracy:.4f}")
        print(f"Plain Evaluation Time: {t_end - t_start:.2f} seconds")
        return accuracy.item(), t_end - t_start

class EncryptedLR:
    """Logistic Regression model for encrypted data."""
    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        
    def forward(self, enc_x):
        return enc_x.dot(self.weight) + self.bias
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)

def create_tenseal_context(poly_mod_degree, coeff_mod_bit_sizes, scale):
    """Create a TenSEAL context for encrypted computation."""
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = scale
    ctx.generate_galois_keys()
    return ctx

def encrypt_test_data(ctx, X_test):
    """Encrypt test data using the TenSEAL context."""
    t_start = time.time()
    enc_x_test = [ts.ckks_vector(ctx, x.tolist()) for x in X_test]
    t_end = time.time()
    print(f"Encryption of test set took {int(t_end - t_start)} seconds")
    return enc_x_test

def evaluate_encrypted_lr(model, enc_x_test, y_test):
    """Evaluate an encrypted logistic regression model."""
    correct = 0
    t_start = time.time()
    for enc_x, y in zip(enc_x_test, y_test):
        enc_out = model(enc_x)
        out = torch.tensor(enc_out.decrypt())
        out = torch.sigmoid(out)
        if torch.abs(out - y) < 0.5:
            correct += 1
    t_end = time.time()
    accuracy = correct / len(y_test)
    print(f"Encrypted Accuracy (LR): {accuracy:.4f}")
    print(f"Encrypted Evaluation Time: {t_end - t_start:.2f} seconds")
    return accuracy, t_end - t_start


# PART 3: K-NEAREST NEIGHBORS MODELS
# =================================

class KNN(nn.Module):
    """K-Nearest Neighbors model using scikit-learn."""
    def __init__(self, n_neighbors):
        super(KNN, self).__init__()
        self.n_neighbors = n_neighbors

    def forward(self, x_train, y_train, x_test):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        return torch.tensor(y_pred)

def evaluate_knn(model, X_train, y_train, X_test, y_test):
    """Evaluate a plain (non-encrypted) KNN model."""
    with torch.no_grad():
        y_pred = model(X_train.numpy(), y_train.numpy(), X_test.numpy())
        accuracy = accuracy_score(y_test.numpy(), y_pred)
        print('Accuracy_plain_KNN:', accuracy)

        # Confusion Matrix
        cm = confusion_matrix(y_test.numpy(), y_pred)
        print(f"Confusion Matrix:\n{cm}")

        # Visualizing Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Precision, Recall, F1 Score
        precision, recall, f1, _ = precision_recall_fscore_support(y_test.numpy(), y_pred, average='binary')
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test.numpy(), y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
        return accuracy, precision, recall, f1

class EncKNNClassifier:
    """K-Nearest Neighbors classifier for encrypted data."""
    def __init__(self, n_neighbors=3, context=None):
        self.n_neighbors = n_neighbors
        self.context = context
        self.encrypted_data = None

    def encrypt_data(self, X):
        return [ts.ckks_vector(self.context, x.tolist()) for x in X]

    def train(self, X_train, y_train):
        # Encrypt the training data
        enc_X_train = self.encrypt_data(X_train)

        # Encrypt the labels
        enc_y_train = [ts.ckks_vector(self.context, [float(y)]) for y in y_train]

        # Store the encrypted training data and labels
        self.encrypted_data = list(zip(enc_X_train, enc_y_train))

    def predict(self, X_test):
        # Encrypt the test data
        enc_X_test = self.encrypt_data(X_test)

        # Make predictions
        predictions = []
        for x_test in enc_X_test:
            distances = [np.linalg.norm(np.array(x_test.decrypt()) - np.array(x_train.decrypt())) ** 2
                        for x_train, _ in self.encrypted_data]

            # Sort distances and get the indices of the k-nearest neighbors
            indices = np.argsort(distances)[:self.n_neighbors]

            # Get the labels of the k-nearest neighbors
            neighbor_labels = [self.encrypted_data[i][1].decrypt()[0] for i in indices]

            # Perform a simple majority voting to predict the label
            result = sum(neighbor_labels) / len(neighbor_labels)

            predictions.append(result)

        return predictions

def evaluate_encrypted_knn(y_test, enc_pred):
    """Evaluate an encrypted KNN model."""
    # Convert predictions to binary
    dec_y_pred = [pred for pred in enc_pred]
    y_pred_binary = np.round(dec_y_pred).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_binary, average='binary')
    
    print(f"Encrypted KNN Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_binary)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return accuracy, precision, recall, f1


# PART 4: MAIN EXECUTION FUNCTION
# ==============================

def main():
    """Main execution function to run the entire pipeline."""
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df_train, df_test = load_data('E:/Encryped_Titanic/highAcc/data/train.csv', 'E:/Encryped_Titanic/highAcc/data/test.csv')
    df_train, df_test = preprocess_data(df_train, df_test)
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(df_train)
    
    # 2. Plain Logistic Regression
    print("\n--- Plain Logistic Regression ---")
    n_features = X_train.shape[1]
    model = LR(n_features)
    optimizer = optim.SGD(model.parameters(), lr=1)
    criterion = nn.BCELoss()
    
    model = train_lr_model(model, optimizer, criterion, X_train, y_train)
    plain_accuracy, plain_eval_time = evaluate_plain_lr(model, X_test, y_test)
    
    # 3. Encrypted Logistic Regression with 4096 polynomial degree
    print("\n--- Encrypted Logistic Regression (4096) ---")
    eelr_4096 = EncryptedLR(model)
    ctx_4096 = create_tenseal_context(4096, [40, 20, 40], 2**20)
    enc_x_test_4096 = encrypt_test_data(ctx_4096, X_test)
    encrypted_accuracy_4096, encrypted_eval_time_4096 = evaluate_encrypted_lr(eelr_4096, enc_x_test_4096, y_test)
    
    # 4. Encrypted Logistic Regression with 8192 polynomial degree
    print("\n--- Encrypted Logistic Regression (8192) ---")
    eelr_8192 = EncryptedLR(model)
    ctx_8192 = create_tenseal_context(8192, [60, 40, 40, 60], 2**40)
    enc_x_test_8192 = encrypt_test_data(ctx_8192, X_test)
    encrypted_accuracy_8192, encrypted_eval_time_8192 = evaluate_encrypted_lr(eelr_8192, enc_x_test_8192, y_test)
    
    # 5. Plain KNN
    print("\n--- Plain KNN ---")
    knn_model = KNN(n_neighbors=5)
    accuracy_plain_knn, precision_plain_knn, recall_plain_knn, f1_plain_knn = evaluate_knn(knn_model, X_train, y_train, X_test, y_test)
    
    # 6. Encrypted KNN with 4096 polynomial degree
    print("\n--- Encrypted KNN (4096) ---")
    context_4096 = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 20, 40])
    context_4096.global_scale = 2 ** 40
    context_4096.generate_galois_keys()
    
    enc_knn_4096 = EncKNNClassifier(n_neighbors=3, context=context_4096)
    enc_knn_4096.train(X_train, y_train)
    enc_y_pred_4096 = enc_knn_4096.predict(X_test)
    accuracy_enc_knn_4096, precision_enc_knn_4096, recall_enc_knn_4096, f1_enc_knn_4096 = evaluate_encrypted_knn(y_test, enc_y_pred_4096)
    
    # 7. Encrypted KNN with 8192 polynomial degree
    print("\n--- Encrypted KNN (8192) ---")
    context_8192 = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context_8192.global_scale = 2 ** 40
    context_8192.generate_galois_keys()
    
    enc_knn_8192 = EncKNNClassifier(n_neighbors=3, context=context_8192)
    enc_knn_8192.train(X_train, y_train)
    enc_y_pred_8192 = enc_knn_8192.predict(X_test)
    accuracy_enc_knn_8192, precision_enc_knn_8192, recall_enc_knn_8192, f1_enc_knn_8192 = evaluate_encrypted_knn(y_test, enc_y_pred_8192)
    
    # 8. Display summary of results
    print("\n=== SUMMARY OF RESULTS ===")
    print("\nLogistic Regression:")
    print(f"Plain LR: Accuracy={plain_accuracy:.4f}, Time={plain_eval_time:.2f}s")
    print(f"Encrypted LR (4096): Accuracy={encrypted_accuracy_4096:.4f}, Time={encrypted_eval_time_4096:.2f}s")
    print(f"Encrypted LR (8192): Accuracy={encrypted_accuracy_8192:.4f}, Time={encrypted_eval_time_8192:.2f}s")
    
    print("\nK-Nearest Neighbors:")
    print(f"Plain KNN: Accuracy={accuracy_plain_knn:.4f}, Precision={precision_plain_knn:.4f}, Recall={recall_plain_knn:.4f}, F1={f1_plain_knn:.4f}")
    print(f"Encrypted KNN (4096): Accuracy={accuracy_enc_knn_4096:.4f}, Precision={precision_enc_knn_4096:.4f}, Recall={recall_enc_knn_4096:.4f}, F1={f1_enc_knn_4096:.4f}")
    print(f"Encrypted KNN (8192): Accuracy={accuracy_enc_knn_8192:.4f}, Precision={precision_enc_knn_8192:.4f}, Recall={recall_enc_knn_8192:.4f}, F1={f1_enc_knn_8192:.4f}")


if __name__ == "__main__":
    main()