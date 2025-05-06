import time
import torch
import numpy as np
import random 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch import nn, optim
import tenseal as ts
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc

def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    # delimiter between test and train data
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

def parse_data():
    data = pd.read_csv("E://Encryped_Titanic/PPML/pima_india_diabetes.csv")
    # drop rows with missing values
    data = data.dropna()
    # drop some features
    # data = data.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"])
    # balance data
    grouped = data.groupby('Outcome')
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
    # extract labels
    y = torch.tensor(data["Outcome"].values).float().unsqueeze(1)
    data = data.drop("Outcome", axis=1)  # Corrected line
    # standardize data
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    return split_train_test(x, y)



def random_data(m=1024, n=2):
    # data separable by the line `y = x`
    x_train = torch.randn(m, n)
    x_test = torch.randn(m // 2, n)
    y_train = (x_train[:, 0] >= x_train[:, 1]).float().unsqueeze(0).t()
    y_test = (x_test[:, 0] >= x_test[:, 1]).float().unsqueeze(0).t()
    return x_train, y_train, x_test, y_test


# You can use whatever data you want without modification to the tutorial
# x_train, y_train, x_test, y_test = random_data()
x_train, y_train, x_test, y_test = parse_data()

# LR on plain data
class LR(nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = nn.Linear(n_features, 1)
        
    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out

n_features = x_train.shape[1]
model = LR(n_features)

# Use gradient descent with a learning_rate=1
optimizer = optim.SGD(model.parameters(), lr=1e-1)
# Use Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Define the number of epochs for both plain and encrypted training
EPOCHS = 200

def train(model, optimizer, criterion, x_train, y_train, x_val=None, y_val=None, epochs=EPOCHS):
    train_losses = []
    val_accuracies = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if x_val is not None and y_val is not None:
            val_acc = accuracy(model, x_val, y_val)
            val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model, train_losses, val_accuracies

def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        out = model(x)
        preds = (out > 0.5).float()
        correct = (preds == y).float().sum()
        return correct / len(y)

# Train the model
model, train_losses, val_accuracies = train(model, optimizer, criterion, x_train, y_train, x_test, y_test)

# Evaluate on test data
plain_accuracy = accuracy(model, x_test, y_test).item()
print(f"Accuracy on plain test set LR: {plain_accuracy}")

# Calculate metrics and confusion matrix
model.eval()
with torch.no_grad():
    out = model(x_test)
    preds = (out > 0.5).float()
    y_true = y_test.cpu().numpy()
    y_pred = preds.cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    precision_plain_lr, recall_plain_lr, f1_plain_lr, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    #precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision_plain_lr}, Recall: {recall_plain_lr}, F1 Score: {f1_plain_lr}")

#4096 LR
class EncryptedLR:
    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        
    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self, context):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

eelr = EncryptedLR(model)

poly_mod_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 20
ctx_eval.generate_galois_keys()

t_start = time.time()
enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in x_test]
t_end = time.time()
time_taken = t_end - t_start
print(f"Encryption of the test-set took {int(time_taken)} seconds") 

def encrypted_evaluation(model, enc_x_test, y_test):
    t_start = time.time()
    
    correct = 0
    y_true = []
    y_pred = []
    losses = []

    for enc_x, y in zip(enc_x_test, y_test):
        enc_out = model(enc_x)
        out = enc_out.decrypt()
        out = torch.tensor(out)
        out = torch.sigmoid(out)

        loss = nn.BCELoss()(out, y)
        losses.append(loss.item())

        if torch.abs(out - y) < 0.5:
            correct += 1
        y_true.append(y.item())
        y_pred.append(out.item())

    t_end = time.time()
    accuracy = correct / len(x_test)

    print(f"Evaluated test_set of {len(x_test)} entries in {t_end - t_start:.2f} seconds")
    print(f"Accuracy_enc_LR: {correct}/{len(x_test)} = {accuracy:.4f}")

    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    cm = confusion_matrix(y_true_tensor, (y_pred_tensor > 0.5).int())
    precision_enc_lr, recall_enc_lr, f1_enc_lr, _ = precision_recall_fscore_support(
        y_true_tensor, (y_pred_tensor > 0.5).int(), average='binary'
    )

    print("Confusion Matrix:", cm.tolist())
    print(f"Precision: {precision_enc_lr:.4f}")
    print(f"Recall: {recall_enc_lr:.4f}")
    print(f"F1 Score: {f1_enc_lr:.4f}")
    print(f"Average Loss: {sum(losses)/len(losses):.6f}")

    fpr, tpr, _ = roc_curve(y_true_tensor, y_pred_tensor)
    roc_auc = auc(fpr, tpr)
    print(f"AUC Score: {roc_auc:.4f}")

    return accuracy, t_end - t_start

encrypted_accuracy, evaluation_time = encrypted_evaluation(eelr, enc_x_test, y_test)
print(f"Encrypted Evaluation Time: {evaluation_time:.2f} seconds")
print(f"Encrypted Accuracy: {encrypted_accuracy:.4f}")


#8192 LR

class EncryptedLR:
    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        
    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self, context):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

eelr = EncryptedLR(model)

poly_mod_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 40
ctx_eval.generate_galois_keys()

t_start = time.time()
enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in x_test]
t_end = time.time()
time_taken = t_end - t_start
print(f"Encryption of the test-set took {int(time_taken)} seconds") 

def encrypted_evaluation(model, enc_x_test, y_test):
    t_start = time.time()
    
    correct = 0
    y_true = []
    y_pred = []
    losses = []  # List to store losses
    
    for enc_x, y in zip(enc_x_test, y_test):
        enc_out = model(enc_x)
        out = enc_out.decrypt()
        out = torch.tensor(out)
        out = torch.sigmoid(out)
        
        loss = nn.BCELoss()(out, y)  # Compute the loss
        losses.append(loss.item())  # Append the loss
        
        if torch.abs(out - y) < 0.5:
            correct += 1
        y_true.append(y)
        y_pred.append(out)
    
    t_end = time.time()
    eval_time = t_end - t_start
    accuracy = correct / len(x_test)

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    
    cm = confusion_matrix(y_true, (y_pred > 0.5).int())
    precision_enc_lr, recall_enc_lr, f1_enc_lr, _ = precision_recall_fscore_support(
        y_true, (y_pred > 0.5).int(), average='binary'
    )
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    print(f"Evaluated test_set of {len(x_test)} entries in {int(eval_time)} seconds")
    print(f"Accuracy_enc_LR: {correct}/{len(x_test)} = {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision_enc_lr:.4f}")
    print(f"Recall: {recall_enc_lr:.4f}")
    print(f"F1 Score: {f1_enc_lr:.4f}")
    print(f"AUC Score: {roc_auc:.4f}")
    print(f"Average Loss: {sum(losses)/len(losses):.6f}")
    
    return accuracy, eval_time

encrypted_accuracy, evaluation_time = encrypted_evaluation(eelr, enc_x_test, y_test)

model.eval()
with torch.no_grad():
    out = model(x_test)
    preds = (out > 0.5).float()
    y_true = y_test.cpu().numpy()
    y_pred = preds.cpu().numpy()
    precision_plain, recall_plain, f1_plain, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

diff_accuracy = plain_accuracy - encrypted_accuracy
print(f"Encrypted Evaluation Time: {evaluation_time:.2f} seconds")
print(f"Encrypted Accuracy: {encrypted_accuracy:.4f}")
print(f"Difference between plain and encrypted accuracies: {diff_accuracy:.4f}")

if diff_accuracy < 0:
    print("Oh! We got a better accuracy on the encrypted test-set! The noise was on our side...")



#KNN

import time
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import numpy as np
import tenseal as ts

# -----------------------------
# Plain KNN Class Definition
# -----------------------------
class KNN(nn.Module):
    def __init__(self, n_neighbors):
        super(KNN, self).__init__()
        self.n_neighbors = n_neighbors

    def forward(self, x_train, y_train, x_test):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        return torch.tensor(y_pred)

# -----------------------------
# Evaluate Plain KNN
# -----------------------------
def evaluate_knn(model, X_train, y_train, X_test, y_test):
    t_start = time.time()

    with torch.no_grad():
        y_pred = model(X_train.numpy(), y_train.numpy(), X_test.numpy())
        
        accuracy = accuracy_score(y_test.numpy(), y_pred)
        cm = confusion_matrix(y_test.numpy(), y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test.numpy(), y_pred, average='binary')
        fpr, tpr, _ = roc_curve(y_test.numpy(), y_pred)
        roc_auc = auc(fpr, tpr)

    t_end = time.time()
    eval_time = t_end - t_start

    print(f"\n--- Plain KNN Evaluation ---")
    print(f"Evaluated on {len(X_test)} samples in {eval_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")
    
    return accuracy, precision, recall, f1, roc_auc, eval_time

# -----------------------------
# Encrypted KNN Class
# -----------------------------
class EncKNNClassifier:
    def __init__(self, n_neighbors=3, context=None):
        self.n_neighbors = n_neighbors
        self.context = context
        self.encrypted_data = None

    def encrypt_data(self, X):
        return [ts.ckks_vector(self.context, x.tolist()) for x in X]

    def train(self, X_train, y_train):
        enc_X_train = self.encrypt_data(X_train)
        enc_y_train = [ts.ckks_vector(self.context, [float(y)]) for y in y_train]
        self.encrypted_data = list(zip(enc_X_train, enc_y_train))

    def predict(self, X_test):
        enc_X_test = self.encrypt_data(X_test)

        predictions = []
        for x_test in enc_X_test:
            distances = [np.linalg.norm(np.array(x_test.decrypt()) - np.array(x_train.decrypt())) ** 2
                         for x_train, _ in self.encrypted_data]

            indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = [self.encrypted_data[i][1].decrypt()[0] for i in indices]
            result = sum(neighbor_labels) / len(neighbor_labels)
            predictions.append(result)

        return predictions

# -----------------------------
# Evaluate Encrypted KNN
# -----------------------------
def evaluate_encrypted_knn(enc_model, X_test, y_test):
    t_start = time.time()

    enc_y_pred = enc_model.predict(X_test)
    dec_y_pred = np.round(enc_y_pred).astype(int)

    accuracy = accuracy_score(y_test, dec_y_pred)
    cm = confusion_matrix(y_test, dec_y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, dec_y_pred, average='binary')
    fpr, tpr, _ = roc_curve(y_test, dec_y_pred)
    roc_auc = auc(fpr, tpr)

    t_end = time.time()
    eval_time = t_end - t_start

    print(f"\n--- Encrypted KNN Evaluation ---")
    print(f"Evaluated on {len(X_test)} samples in {eval_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc, eval_time


# Instantiate and evaluate plain KNN
knn_model = KNN(n_neighbors=5)
accuracy_plain, precision_plain, recall_plain, f1_plain, auc_plain, time_plain = evaluate_knn(knn_model, x_train, y_train, x_test, y_test)

# Setup encryption context
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 20, 40])
context.global_scale = 2 ** 20
context.generate_galois_keys()

# Instantiate and evaluate encrypted KNN
enc_knn_classifier = EncKNNClassifier(n_neighbors=3, context=context)
enc_knn_classifier.train(x_train.numpy(), y_train.numpy())
accuracy_enc, precision_enc, recall_enc, f1_enc, auc_enc, time_enc = evaluate_encrypted_knn(enc_knn_classifier, x_test.numpy(), y_test.numpy())


print(f"\n--- Comparison ---")
print(f"Accuracy Drop: {accuracy_plain - accuracy_enc:.4f}")
print(f"Time Increase: {time_enc - time_plain:.2f} seconds")

# -----------------------------
# Encrypted KNN Class
# -----------------------------
class EncKNNClassifier:
    def __init__(self, n_neighbors=3, context=None):
        self.n_neighbors = n_neighbors
        self.context = context
        self.encrypted_data = None

    def encrypt_data(self, X):
        return [ts.ckks_vector(self.context, x.tolist()) for x in X]

    def train(self, X_train, y_train):
        enc_X_train = self.encrypt_data(X_train)
        enc_y_train = [ts.ckks_vector(self.context, [float(y)]) for y in y_train]
        self.encrypted_data = list(zip(enc_X_train, enc_y_train))

    def predict(self, X_test):
        enc_X_test = self.encrypt_data(X_test)

        predictions = []
        for x_test in enc_X_test:
            distances = [np.linalg.norm(np.array(x_test.decrypt()) - np.array(x_train.decrypt())) ** 2
                         for x_train, _ in self.encrypted_data]

            indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = [self.encrypted_data[i][1].decrypt()[0] for i in indices]
            result = sum(neighbor_labels) / len(neighbor_labels)
            predictions.append(result)

        return predictions

# -----------------------------
# Evaluate Encrypted KNN
# -----------------------------
def evaluate_encrypted_knn(enc_model, X_test, y_test):
    t_start = time.time()

    enc_y_pred = enc_model.predict(X_test)
    dec_y_pred = np.round(enc_y_pred).astype(int)

    accuracy = accuracy_score(y_test, dec_y_pred)
    cm = confusion_matrix(y_test, dec_y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, dec_y_pred, average='binary')
    fpr, tpr, _ = roc_curve(y_test, dec_y_pred)
    roc_auc = auc(fpr, tpr)

    t_end = time.time()
    eval_time = t_end - t_start

    print(f"\n--- Encrypted KNN Evaluation ---")
    print(f"Evaluated on {len(X_test)} samples in {eval_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc, eval_time


# Instantiate and evaluate plain KNN
knn_model = KNN(n_neighbors=5)
accuracy_plain, precision_plain, recall_plain, f1_plain, auc_plain, time_plain = evaluate_knn(knn_model, x_train, y_train, x_test, y_test)

# Setup encryption context
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2 ** 20
context.generate_galois_keys()

# Instantiate and evaluate encrypted KNN
enc_knn_classifier = EncKNNClassifier(n_neighbors=3, context=context)
enc_knn_classifier.train(x_train.numpy(), y_train.numpy())
accuracy_enc, precision_enc, recall_enc, f1_enc, auc_enc, time_enc = evaluate_encrypted_knn(enc_knn_classifier, x_test.numpy(), y_test.numpy())


print(f"\n--- Comparison ---")
print(f"Accuracy Drop: {accuracy_plain - accuracy_enc:.4f}")
print(f"Time Increase: {time_enc - time_plain:.2f} seconds")

