
import time 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tenseal as ts
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset 

import warnings
warnings.filterwarnings('ignore')

# Load data
df_train = pd.read_csv('E://Encryped_Titanic/highAcc/data/train.csv')
df_test = pd.read_csv('E://Encryped_Titanic/highAcc/data/test.csv')

# Feature engineering, preprocessing, and encoding
df_train['Cabin'].fillna('N0', inplace=True)
df_train['Cabin'] = df_train['Cabin'].str.extract(r'([A-Za-z]+)')

df_test['Cabin'].fillna('N0', inplace=True)
df_test['Cabin'] = df_train['Cabin'].str.extract(r'([A-Za-z]+)')

df_train["Family_Size"] = df_train["SibSp"] + df_train["Parch"]
df_test["Family_Size"] = df_test["SibSp"] + df_test["Parch"]

df_train["Family_Size"] = df_train["Family_Size"].apply(lambda x: 6 if x > 6 else x)
df_test["Family_Size"] = df_test["Family_Size"].apply(lambda x: 6 if x > 6 else x)

age_bins = [0, 7, 18, 30, 50, 120]
age_labels = ['Young Child', 'Child', 'Young Adult', 'Adult', 'Senior']

df_train['Age_Bin'] = pd.cut(df_train['Age'], bins=age_bins, labels=age_labels, right=False)
df_train['Age_Bin'] = df_train['Age_Bin'].cat.add_categories('Unknown').fillna('Unknown')
df_train.drop('Age', axis=1, inplace=True)

df_test['Age_Bin'] = pd.cut(df_test['Age'], bins=age_bins, labels=age_labels, right=False)
df_test['Age_Bin'] = df_test['Age_Bin'].cat.add_categories('Unknown').fillna('Unknown')
df_test.drop('Age', axis=1, inplace=True)

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

fare_bins = [-1, 25, 50, 100, np.inf]
labels_fare = ['0-25', '25-50', '50-100', '100+']

df_train['Fare_bin'] = pd.cut(df_train['Fare'], bins=fare_bins, labels=labels_fare)
df_train['Fare_bin'] = df_train['Fare_bin'].cat.add_categories('Unknown').fillna('Unknown')
df_train.drop('Fare', axis=1, inplace=True)

df_test['Fare_bin'] = pd.cut(df_test['Fare'], bins=fare_bins, labels=labels_fare)
df_test['Fare_bin'] = df_test['Fare_bin'].cat.add_categories('Unknown').fillna('Unknown')
df_test.drop('Fare', axis=1, inplace=True)

df_train['isAlone'] = (df_train['Family_Size'] == 0).astype(int)
df_test['isAlone'] = (df_test['Family_Size'] == 0).astype(int)

dropped_column = ['PassengerId', 'Ticket']
df_train.drop(dropped_column, axis=1, inplace=True)
df_test_new = df_test.drop(dropped_column, axis=1, inplace=False)

df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test_new['Embarked'].fillna(df_test_new['Embarked'].mode()[0], inplace=True)

df_train['Sex'] = (df_train['Sex'] == 'male')

label_encoder = LabelEncoder()
df_train['Pclass'] = label_encoder.fit_transform(df_train['Pclass'])
df_train['Age_Bin'] = label_encoder.fit_transform(df_train['Age_Bin'])
df_train['Fare_bin'] = label_encoder.fit_transform(df_train['Fare_bin'])

df_train = pd.get_dummies(df_train, columns=['Cabin', 'Embarked', 'Title'], prefix=['Cabin', 'Embarked', 'Title'])

df_train['Age_Pclass'] = df_train['Age_Bin'] * df_train['Pclass']
df_train['isAlone_Age'] = df_train['isAlone'] * df_train['Age_Bin']
df_train['isAlone_Sex'] = df_train['isAlone'] * df_train['Sex']

df_test_new['Sex'] = (df_test_new['Sex'] == 'male')

label_encoder = LabelEncoder()
df_test_new['Pclass'] = label_encoder.fit_transform(df_test_new['Pclass'])
df_test_new['Age_Bin'] = label_encoder.fit_transform(df_test_new['Age_Bin'])
df_test_new['Fare_bin'] = label_encoder.fit_transform(df_test_new['Fare_bin'])

df_test_new = pd.get_dummies(df_test_new, columns=['Cabin', 'Embarked', 'Title'], prefix=['Cabin', 'Embarked', 'Title'])

df_test_new['Age_Pclass'] = df_test_new['Age_Bin'] * df_test_new['Pclass']
df_test_new['isAlone_Age'] = df_test_new['isAlone'] * df_test_new['Age_Bin']
df_test_new['isAlone_Sex'] = df_test_new['isAlone'] * df_test_new['Sex']


# Split the data into features (X) and target (y)
X = df_train.drop('Survived', axis=1)  # Corrected from df to df_train
y = df_train['Survived']  # Corrected from df to df_train


# Oversampling (if data is imbalanced)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)


# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)             

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic Regression on plain data
class LR(nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.lr(x))

n_features = X_train.shape[1]
model = LR(n_features)
optimizer = optim.SGD(model.parameters(), lr=1)
criterion = nn.BCELoss()
EPOCHS = 200

def train(model, optimizer, criterion, x_train, y_train, epochs=EPOCHS):
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
    return model

def evaluate_plain(model, x_test, y_test):
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

# Train and evaluate on plain data
model = train(model, optimizer, criterion, X_train, y_train)
plain_accuracy, plain_eval_time = evaluate_plain(model, X_test, y_test)

# Encrypted LR
class EncryptedLR:
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

eelr = EncryptedLR(model)

# CKKS context
poly_mod_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 20
ctx_eval.generate_galois_keys()

# Encrypt test data
t_start = time.time()
enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in X_test]
t_end = time.time()
print(f"Encryption of test set took {int(t_end - t_start)} seconds")

# Evaluate on encrypted data
def evaluate_encrypted(model, enc_x_test, y_test):
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

encrypted_accuracy, encrypted_eval_time = evaluate_encrypted(eelr, enc_x_test, y_test)

# Encrypted LR
class EncryptedLR:
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

eelr = EncryptedLR(model)

# CKKS context
poly_mod_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 40
ctx_eval.generate_galois_keys()

# Encrypt test data
t_start = time.time()
enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in X_test]
t_end = time.time()
print(f"Encryption of test set took {int(t_end - t_start)} seconds")

# Evaluate on encrypted data
def evaluate_encrypted(model, enc_x_test, y_test):
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

encrypted_accuracy, encrypted_eval_time = evaluate_encrypted(eelr, enc_x_test, y_test)

#KNN models

class KNN(nn.Module):
    def __init__(self, n_neighbors):
        super(KNN, self).__init__()
        self.n_neighbors = n_neighbors

    def forward(self, x_train, y_train, x_test):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        return torch.tensor(y_pred)

def evaluate_knn(model, X_train, y_train, X_test, y_test):
    with torch.no_grad():
        y_pred = model(X_train.numpy(), y_train.numpy(), X_test.numpy())
        accuracy = accuracy_score(y_test.numpy(), y_pred)
        print('Accuracy_plain_KNN:', accuracy)

        # Confusion Matrix
        cm = confusion_matrix(y_test.numpy(), y_pred)
        print(f"Confusion Matrix:\n{cm}")

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
        precision_plain_knn, recall_plain_knn, f1_plain_knn, _ = precision_recall_fscore_support(y_test.numpy(), y_pred, average='binary')
        print(f"Precision: {precision_plain_knn}, Recall: {recall_plain_knn}, F1 Score: {f1_plain_knn}")

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

        # Loss curve can't be directly computed for KNN as it's a non-parametric method.

# Instantiate KNN model
knn_model = KNN(n_neighbors=5)

# Evaluate KNN model
evaluate_knn(knn_model, X_train, y_train, X_test, y_test)

accuracy_plain_knn = ...  # Extract this from the evaluate_knn function
precision_plain_knn = ...  # Extract this from the evaluate_knn function
recall_plain_knn = ...  # Extract this from the evaluate_knn function

#Loss Curve: Since KNN doesn't have a training process like neural networks where loss is minimized during training, there's no loss curve in the traditional sense.

#4096 polynomial degree

class EncKNNClassifier:
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

# Feature engineering, preprocessing, and encoding
# (The provided feature engineering and preprocessing code goes here)

# Split the data into features (X) and target (y)
X = df_train.drop('Survived', axis=1)
y = df_train['Survived']

# Oversampling (if data is imbalanced)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a context for encryption and decryption
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 20, 40])
context.global_scale = 2 ** 40
context.generate_galois_keys()

# Create and train the encrypted KNN classifier
enc_knn_classifier = EncKNNClassifier(n_neighbors=3, context=context)
enc_knn_classifier.train(X_train, y_train)

# Make predictions on the test set
enc_y_pred = enc_knn_classifier.predict(X_test)

# Decrypt predictions for evaluation
dec_y_pred = [pred for pred in enc_y_pred]


# Calculate metrics
accuracy_enc_knn = accuracy_score(y_test, np.round(dec_y_pred).astype(int))
conf_matrix = confusion_matrix(y_test, np.round(dec_y_pred).astype(int))
precision_enc_knn, recall_enc_knn, f1_enc_knn, _ = precision_recall_fscore_support(y_test, np.round(dec_y_pred).astype(int), average='binary')

import seaborn as sns

# Calculate metrics
accuracy = accuracy_score(y_test, np.round(dec_y_pred).astype(int))
conf_matrix = confusion_matrix(y_test, np.round(dec_y_pred).astype(int))
precision_enc_knn, recall_enc_knn, f1_enc_knn, _ = precision_recall_fscore_support(y_test, np.round(dec_y_pred).astype(int), average='binary')

print(f"accuracy_enc_KNN: {accuracy * 100:.2f}%")
print(f"Precision: {precision_enc_knn}, Recall: {recall_enc_knn}, F1 Score: {f1_enc_knn}")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, np.round(dec_y_pred).astype(int))
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


#8192 polynomial degree

class EncKNNClassifier:
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

# Feature engineering, preprocessing, and encoding
# (The provided feature engineering and preprocessing code goes here)

# Split the data into features (X) and target (y)
X = df_train.drop('Survived', axis=1)
y = df_train['Survived']

# Oversampling (if data is imbalanced)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a context for encryption and decryption
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2 ** 40
context.generate_galois_keys()

# Create and train the encrypted KNN classifier
enc_knn_classifier = EncKNNClassifier(n_neighbors=3, context=context)
enc_knn_classifier.train(X_train, y_train)

# Make predictions on the test set
enc_y_pred = enc_knn_classifier.predict(X_test)

# Decrypt predictions for evaluation
dec_y_pred = [pred for pred in enc_y_pred]


# Calculate metrics
accuracy_enc_knn = accuracy_score(y_test, np.round(dec_y_pred).astype(int))
conf_matrix = confusion_matrix(y_test, np.round(dec_y_pred).astype(int))
precision_enc_knn, recall_enc_knn, f1_enc_knn, _ = precision_recall_fscore_support(y_test, np.round(dec_y_pred).astype(int), average='binary')

import seaborn as sns

# Calculate metrics
accuracy = accuracy_score(y_test, np.round(dec_y_pred).astype(int))
conf_matrix = confusion_matrix(y_test, np.round(dec_y_pred).astype(int))
precision_enc_knn, recall_enc_knn, f1_enc_knn, _ = precision_recall_fscore_support(y_test, np.round(dec_y_pred).astype(int), average='binary')

print(f"accuracy_enc_KNN: {accuracy * 100:.2f}%")
print(f"Precision: {precision_enc_knn}, Recall: {recall_enc_knn}, F1 Score: {f1_enc_knn}")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, np.round(dec_y_pred).astype(int))
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
