# ===============================
# BUILD FLAGS
# ===============================
BUILD95 = True
BUILD96 = True

# ===============================
# IMPORT LIBRARIES
# ===============================
import torch
import tenseal as ts
import pandas as pd
import random
from time import time  # Using time module instead of %%time magic
import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import dump
import seaborn as sns
import gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# ===============================
# COLUMN DEFINITIONS
# ===============================
# Categorical String Columns
str_type = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 
    'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 
    'id_38', 'DeviceType', 'DeviceInfo',
    'id-12', 'id-15', 'id-16', 'id-23', 'id-27', 'id-28', 'id-29', 
    'id-30', 'id-31', 'id-33', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38'
]

# First 53 Columns
cols = [
    'TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 
    'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 
    'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
    'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 
    'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 
    'M6', 'M7', 'M8', 'M9'
]

# Selected V Columns
v =  [1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47, 48,
      54, 56, 59, 62, 65, 67, 68, 70, 76, 78, 80, 82, 86, 88, 89, 91, 107, 108, 111, 
      115, 117, 120, 121, 123, 124, 127, 129, 130, 136, 138, 139, 142, 147, 156, 162, 
      165, 160, 166, 178, 176, 173, 182, 187, 203, 205, 207, 215, 169, 171, 175, 180, 
      185, 188, 198, 210, 209, 218, 223, 224, 226, 228, 229, 235, 240, 258, 257, 253, 
      252, 260, 261, 264, 266, 267, 274, 277, 220, 221, 234, 238, 250, 271, 294, 284, 
      285, 286, 291, 297, 303, 305, 307, 309, 310, 320, 281, 283, 289, 296, 301, 314]

cols += ['V'+str(x) for x in v]

# Data Types Dictionary
dtypes = {}
for c in cols + ['id_0'+str(x) for x in range(1,10)] + ['id_'+str(x) for x in range(10,34)] + \
         ['id-0'+str(x) for x in range(1,10)] + ['id-'+str(x) for x in range(10,34)]:
    dtypes[c] = 'float32'
for c in str_type:
    dtypes[c] = 'category'

# ===============================
# LOAD DATA
# ===============================
start_time = time()  # Start the timer
X_train = pd.read_csv(
    'E://Encryped_Titanic/Fraud_detection/train_transaction.csv',
    index_col='TransactionID', dtype=dtypes, usecols=cols+['isFraud']
)
train_id = pd.read_csv(
    'E://Encryped_Titanic/Fraud_detection/train_identity.csv',
    index_col='TransactionID', dtype=dtypes
)
X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)

X_test = pd.read_csv(
    'E://Encryped_Titanic/Fraud_detection/test_transaction.csv',
    index_col='TransactionID', dtype=dtypes, usecols=cols
)
test_id = pd.read_csv(
    'E://Encryped_Titanic/Fraud_detection/test_identity.csv',
    index_col='TransactionID', dtype=dtypes
)
fix = {o:n for o, n in zip(test_id.columns, train_id.columns)}
test_id.rename(columns=fix, inplace=True)
X_test = X_test.merge(test_id, how='left', left_index=True, right_index=True)

Y_train = X_train['isFraud'].copy()
del train_id, test_id, X_train['isFraud']
x = gc.collect()
print('Train shape', X_train.shape, 'test shape', X_test.shape)
end_time = time()  # End the timer
print(f"Time taken to load data: {end_time - start_time} seconds")

# ===============================
# LABEL ENCODING & PREPROCESSING
# ===============================
for i, f in enumerate(X_train.columns):
    if (str(X_train[f].dtype) == 'category') | (X_train[f].dtype == 'object'): 
        df_comb = pd.concat([X_train[f], X_test[f]], axis=0)
        df_comb, _ = df_comb.factorize(sort=True)
        if df_comb.max() > 32000:
            print(f, 'needs int32')
        X_train[f] = df_comb[:len(X_train)].astype('int16')
        X_test[f] = df_comb[len(X_train):].astype('int16')
    elif f not in ['TransactionAmt', 'TransactionDT']:
        mn = np.min((X_train[f].min(), X_test[f].min()))
        X_train[f] -= np.float32(mn)
        X_test[f] -= np.float32(mn)
        X_train[f].fillna(-1, inplace=True)
        X_test[f].fillna(-1, inplace=True)

# ===============================
# FEATURE ENGINEERING FUNCTIONS
# ===============================
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc).astype('float32')
        df2[nm] = df2[col].map(vc).astype('float32')
        print(nm,', ',end='')

def encode_LE(col, train=X_train, test=X_test, verbose=True):
    df_comb = pd.concat([train[col], test[col]], axis=0)
    df_comb, _ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max() > 32000: 
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')
    else:
        train[nm] = df_comb[:len(train)].astype('int16')
        test[nm] = df_comb[len(train):].astype('int16')
    del df_comb
    x = gc.collect()
    if verbose: print(nm, ', ', end='')

def encode_AG(main_columns, uids, aggregations=['mean'], train_df=X_train, test_df=X_test, fillna=True, usena=False):
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = f'{main_column}_{col}_{agg_type}'
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                if usena:
                    temp_df.loc[temp_df[main_column]==-1, main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})
                temp_df.index = temp_df[col]
                temp_df = temp_df[new_col_name].to_dict()
                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name] = test_df[col].map(temp_df).astype('float32')
                if fillna:
                    train_df[new_col_name].fillna(-1, inplace=True)
                    test_df[new_col_name].fillna(-1, inplace=True)
                print(f"'{new_col_name}', ", end='')

def encode_CB(col1, col2, df1=X_train, df2=X_test):
    nm = col1 + '_' + col2
    df1[nm] = df1[col1].astype(str) + '_' + df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str) + '_' + df2[col2].astype(str)
    encode_LE(nm, verbose=False)
    print(nm, ', ', end='')

def encode_AG2(main_columns, uids, train_df=X_train, test_df=X_test):
    for main_column in main_columns:
        for col in uids:
            comb = pd.concat([train_df[[col]+[main_column]], test_df[[col]+[main_column]]], axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col + '_' + main_column + '_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col + '_' + main_column + '_ct'] = test_df[col].map(mp).astype('float32')
            print(col + '_' + main_column + '_ct, ', end='')

# ===============================
# FEATURE ENGINEERING
# ===============================
start_time = time()  # Start the timer
X_train['cents'] = (X_train['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')
X_test['cents'] = (X_test['TransactionAmt'] - np.floor(X_test['TransactionAmt'])).astype('float32')
print('cents, ', end='')

encode_FE(X_train, X_test, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
encode_CB('card1', 'addr1')
encode_CB('card1_addr1', 'P_emaildomain')
encode_FE(X_train, X_test, ['card1_addr1', 'card1_addr1_P_emaildomain'])
encode_AG(['TransactionAmt', 'D9', 'D11'], ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'], usena=True)

# ===============================
# FINAL COLUMN SETUP
# ===============================
cols = list(X_train.columns)
cols.remove('TransactionDT')
print('NOW USING THE FOLLOWING', len(cols), 'FEATURES.')
np.array(cols)

# ===============================
# TRAIN-TEST SPLIT
# ===============================
idxTrain = X_train.index
X_test = X_test[cols].copy()
X_train = X_train[cols].copy()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
end_time = time()  # End the timer
print(f"Feature engineering took {end_time - start_time} seconds")


# ===============================
# TRAIN-TEST SPLIT & FEATURE SELECTION
# ===============================
from sklearn.model_selection import train_test_split

# Train-test split with stratification for imbalanced dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
x_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# ===============================
# LOGISTIC REGRESSION MODEL CLASS
# ===============================
class LR(torch.nn.Module):
    def __init__(self, num_features):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

# ===============================
# INITIALIZE MODEL, LOSS, AND OPTIMIZER
# ===============================
model = LR(num_features=X_train.shape[1])  # Number of features in training data
criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# ===============================
# TRAINING LOOP
# ===============================
EPOCHS = 10
start_time = time()  # Start time for training

for epoch in range(EPOCHS):
    model.train()
    y_pred = model(x_train_tensor)  # Forward pass
    loss = criterion(y_pred, y_train_tensor)  # Calculate loss

    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagate
    optimizer.step()  # Update weights
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# End time for training
end_time = time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# ===============================
# MODEL EVALUATION
# ===============================
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred_test = model(x_test_tensor)  # Predictions on test data
    test_loss = criterion(y_pred_test, y_test_tensor).item()  # Test loss
    y_pred_labels = (y_pred_test.numpy() > 0.5).astype(int)  # Convert probabilities to class labels
    accuracy = (y_pred_labels == y_test_tensor.numpy()).mean()  # Accuracy calculation

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


import time

# Encrypted Inference using TenSEAL with polynomial modulus degree 4096
import tenseal as ts

context_4096 = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=4096,
    coeff_mod_bit_sizes=[40, 20, 40]
)
context_4096.global_scale = 2**20
context_4096.generate_galois_keys()

# Inference on test data with encryption (4096)
encrypted_preds_4096 = []
start_time = time.time()

for i in range(len(x_test_tensor)):
    enc_x = ts.ckks_vector(context_4096, x_test_tensor[i].tolist())
    enc_out = enc_x.dot(model.linear.weight[0].tolist()) + model.linear.bias[0].item()
    encrypted_preds_4096.append(enc_out.decrypt())

inference_time_4096 = time.time() - start_time

# Apply sigmoid and threshold
y_pred_enc_4096 = torch.tensor(encrypted_preds_4096)
y_pred_enc_4096 = torch.sigmoid(y_pred_enc_4096).numpy()
y_pred_labels_4096 = (y_pred_enc_4096 > 0.5).astype(int)
accuracy_enc_4096 = (y_pred_labels_4096 == y_test_tensor.numpy()).mean()

print(f"[4096] Encrypted Inference Accuracy: {accuracy_enc_4096:.4f}")
print(f"[4096] Inference Time: {inference_time_4096:.2f} seconds")


# Encrypted Inference using TenSEAL with polynomial modulus degree 8192
context_8192 = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[40, 20, 40]
)
context_8192.global_scale = 2**20
context_8192.generate_galois_keys()

# Inference on test data with encryption (8192)
encrypted_preds_8192 = []
start_time = time.time()

for i in range(len(x_test_tensor)):
    enc_x = ts.ckks_vector(context_8192, x_test_tensor[i].tolist())
    enc_out = enc_x.dot(model.linear.weight[0].tolist()) + model.linear.bias[0].item()
    encrypted_preds_8192.append(enc_out.decrypt())

inference_time_8192 = time.time() - start_time

# Apply sigmoid and threshold
y_pred_enc_8192 = torch.tensor(encrypted_preds_8192)
y_pred_enc_8192 = torch.sigmoid(y_pred_enc_8192).numpy()
y_pred_labels_8192 = (y_pred_enc_8192 > 0.5).astype(int)
accuracy_enc_8192 = (y_pred_labels_8192 == y_test_tensor.numpy()).mean()

print(f"[8192] Encrypted Inference Accuracy: {accuracy_enc_8192:.4f}")
print(f"[8192] Inference Time: {inference_time_8192:.2f} seconds")
