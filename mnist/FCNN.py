import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the FCNN model
class SimpleFCNN(nn.Module):
    def __init__(self):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28 images) -> Hidden layer (128 units)
        self.fc2 = nn.Linear(128, 64)      # Hidden layer (128 units) -> Hidden layer (64 units)
        self.fc3 = nn.Linear(64, 10)       # Hidden layer (64 units) -> Output layer (10 units)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, criterion, and optimizer
model = SimpleFCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Train the model
train_model(model, train_loader, criterion, optimizer)

##The polynomial degree of 4096

import tenseal as ts
import time

# Function to encrypt a tensor
def encrypt_tensor(tensor, context):
    encrypted_tensor = []
    for i in range(tensor.shape[0]):
        encrypted_tensor.append(ts.ckks_tensor(context, tensor[i].view(-1).tolist()))
    return encrypted_tensor

# Function to decrypt a tensor
def decrypt_tensor(encrypted_tensor):
    decrypted_tensor = []
    for i in range(len(encrypted_tensor)):
        decrypted_tensor.append(torch.tensor(encrypted_tensor[i].decrypt().tolist()).view(1, 28 * 28))
    return torch.cat(decrypted_tensor, dim=0)

# Create CKKS context
poly_mod_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 20
ctx_eval.generate_galois_keys()

# Evaluate the model with encrypted inputs
def evaluate_encrypted(model, test_loader, context):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            encrypted_data = encrypt_tensor(data, context)
            encrypted_output = []
            for encrypted_sample in encrypted_data:
                decrypted_sample = decrypt_tensor([encrypted_sample])
                output = model(decrypted_sample)
                encrypted_output.append(output)
            output = torch.cat(encrypted_output, dim=0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    end_time = time.time()
    return correct / total, end_time - start_time

# Load test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the model with encrypted inputs
accuracy, time_taken = evaluate_encrypted(model, test_loader, ctx_eval)
print(f'Encrypted Evaluation Accuracy: {accuracy:.4f}')
print(f'Encrypted Evaluation Time: {time_taken:.4f} seconds')

def evaluate_normal(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    end_time = time.time()
    return correct / total, end_time - start_time

# Evaluate the model with normal inputs
accuracy, time_taken = evaluate_normal(model, test_loader)
print(f'Normal Evaluation Accuracy: {accuracy:.4f}')
print(f'Normal Evaluation Time: {time_taken:.4f} seconds')

##Polynomial degree of 8192
# Function to encrypt a tensor
def encrypt_tensor(tensor, context):
    encrypted_tensor = []
    for i in range(tensor.shape[0]):
        encrypted_tensor.append(ts.ckks_tensor(context, tensor[i].view(-1).tolist()))
    return encrypted_tensor

# Function to decrypt a tensor
def decrypt_tensor(encrypted_tensor):
    decrypted_tensor = []
    for i in range(len(encrypted_tensor)):
        decrypted_tensor.append(torch.tensor(encrypted_tensor[i].decrypt().tolist()).view(1, 28 * 28))
    return torch.cat(decrypted_tensor, dim=0)

# Create CKKS context with polynomial degree 8192
poly_mod_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]  # Adjust these values based on your requirements
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 40  # Adjust the global scale as needed
ctx_eval.generate_galois_keys()

# Evaluate the model with encrypted inputs
def evaluate_encrypted(model, test_loader, context):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            encrypted_data = encrypt_tensor(data, context)
            encrypted_output = []
            for encrypted_sample in encrypted_data:
                decrypted_sample = decrypt_tensor([encrypted_sample])
                output = model(decrypted_sample)
                encrypted_output.append(output)
            output = torch.cat(encrypted_output, dim=0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    end_time = time.time()
    return correct / total, end_time - start_time

# Load test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the model with encrypted inputs
accuracy, time_taken = evaluate_encrypted(model, test_loader, ctx_eval)
print(f'Encrypted Evaluation Accuracy: {accuracy:.4f}')
print(f'Encrypted Evaluation Time: {time_taken:.4f} seconds')
