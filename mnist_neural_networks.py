import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time

torch.manual_seed(73)

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 64

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, 1024)
        self.fc2 = torch.nn.Linear(1024, hidden)
        self.fc3 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        x = x * x
        x = self.fc3(x)
        return x


def train(model, train_loader, criterion, optimizer, n_epochs=10):
    # model in training mode
    model.train()
    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    # model in evaluation mode
    model.eval()
    return model


model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


start = time.time()
model = train(model, train_loader, criterion, optimizer, 10)
end = time.time()

print("model training takes", end - start, "s")

# PATH = "./mnist_cnn.pth"
# torch.save(model, PATH)

def test(model, test_loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss/len(test_loader)

    class_precision = []
    class_recall = []
    class_f1_score = []

    for label in range(10):
        precision = class_correct[label] / class_total[label]
        recall = class_correct[label] / np.sum(class_correct[label])
        f1_score = 2 * (precision * recall) / (precision + recall)
        class_precision.append(precision)
        class_recall.append(recall)
        class_f1_score.append(f1_score)

    overall_precision = np.sum(class_correct) / np.sum(class_total)
    overall_recall = np.sum(class_correct) / np.sum(class_correct)
    overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

    overall_accuracy = np.sum(class_correct) / np.sum(class_total)

    return test_loss, class_precision, class_recall, class_f1_score, overall_precision, overall_recall, overall_f1_score, overall_accuracy

start = time.time()
test_loss, class_precision, class_recall, class_f1_score, overall_precision, overall_recall, overall_f1_score, overall_accuracy = test(model, test_loader, criterion)
end = time.time()

print("model testing takes", end - start, "s")

average_precision = np.mean(class_precision)
average_recall = np.mean(class_recall)
average_f1_score = np.mean(class_f1_score)

print(f'Average Precision: {average_precision:.4f}')
print(f'Average Recall: {average_recall:.4f}')
print(f'Average F1 Score: {average_f1_score:.4f}')
print(f'Overall Accuracy: {overall_accuracy:.4f}')


import tenseal as ts
    
class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        
    def forward(self, enc_x, windows_nb):
        # conv layer
        start = time.time()
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        end = time.time()
        print("cov2d takes", end - start)
        
        
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        end = time.time()
        print("fc1 takes", end - start)
        # square activation
        enc_x.square_()
        # fc2 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        end = time.time()
        print("fc2 takes", end - start)
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    
def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    cnt = 0
    for data, target in test_loader:
        
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
        # Encrypted evaluation
        enc_output = enc_model(x_enc, windows_nb)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()
        
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
        cnt += 1
        if cnt == 100:
            break

    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')
    
    # Calculate accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    accuracy = total_correct / total_samples
    accuracy_percentage = accuracy * 100
    print(f'Accuracy: {accuracy_percentage:.2f}%')

    print(class_correct)
    print(class_total)

# Load one element at a time
model = torch.load("mnist_cnn.pth")
model.eval()

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
# required for encoding
kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]

## Encryption Parameters
bits_scale = 20

# Create TenSEAL context
poly_mod_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 20
ctx_eval.generate_galois_keys()

# enc_model = EncConvNet(model)
enc_model = EncConvNet(model)
start = time.time()
enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
end = time.time()
print("model testing takes", end - start, "s")


import tenseal as ts
    
class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        
    def forward(self, enc_x, windows_nb):
        # conv layer
        start = time.time()
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        end = time.time()
        print("cov2d takes", end - start)
        
        
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        end = time.time()
        print("fc1 takes", end - start)
        # square activation
        enc_x.square_()
        # fc2 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        end = time.time()
        print("fc2 takes", end - start)
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    
def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    cnt = 0
    for data, target in test_loader:
        
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
        # Encrypted evaluation
        enc_output = enc_model(x_enc, windows_nb)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()
        
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
        cnt += 1
        if cnt == 100:
            break

    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')
    
    # Calculate accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    accuracy = total_correct / total_samples
    accuracy_percentage = accuracy * 100
    print(f'Accuracy: {accuracy_percentage:.2f}%')

    print(class_correct)
    print(class_total)

# Load one element at a time
model = torch.load("mnist_cnn.pth")
model.eval()

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
# required for encoding
kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]

## Encryption Parameters
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

# enc_model = EncConvNet(model)
enc_model = EncConvNet(model)
start = time.time()
enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
end = time.time()
print("model testing takes", end - start, "s")



#########################
#FCNN model
#########################

import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
import tenseal as ts

# Set random seed for reproducibility
torch.manual_seed(73)

def load_data(batch_size=64, single_batch=False):
    """Load MNIST dataset with specified batch size"""
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    
    if single_batch:
        # For encryption testing, we only need batch size of 1
        test_loader_single = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
        return train_data, test_data, None, test_loader_single
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_data, test_data, train_loader, test_loader


class FullyConnectedNet(torch.nn.Module):
    """Fully connected neural network for MNIST classification"""
    def __init__(self, hidden=64, output=10):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = torch.nn.Linear(784, 1024)
        self.fc2 = torch.nn.Linear(1024, hidden)
        self.fc3 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten while keeping the batch axis
        x = self.fc1(x)
        x = x * x  # Square activation
        x = self.fc2(x)
        x = x * x  # Square activation
        x = self.fc3(x)
        return x


def train(model, train_loader, criterion, optimizer, n_epochs=10):
    """Train the model"""
    print("\n" + "="*50)
    print("TRAINING STANDARD MODEL")
    print("="*50)
    
    start_time = time.time()
    model.train()
    
    for epoch in range(1, n_epochs+1):
        epoch_start = time.time()
        train_loss = 0.0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        epoch_end = time.time()
        
        print(f'Epoch: {epoch}/{n_epochs} | Training Loss: {train_loss:.6f} | Time: {epoch_end-epoch_start:.2f}s')
    
    # Model in evaluation mode
    model.eval()
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")
    
    return model


def test(model, test_loader, criterion):
    """Test the standard model"""
    print("\n" + "="*50)
    print("TESTING STANDARD MODEL")
    print("="*50)
    
    start_time = time.time()
    
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    # Model in evaluation mode
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            
            # Compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            
            # Calculate test accuracy for each object class
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # Calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    # Print per-class accuracy
    for label in range(10):
        accuracy = 100 * class_correct[label] / class_total[label]
        print(f'Test Accuracy of {label}: {accuracy:.2f}% ({int(class_correct[label])}/{int(class_total[label])})')

    # Print overall accuracy
    total_correct = int(np.sum(class_correct))
    total_samples = int(np.sum(class_total))
    overall_accuracy = 100 * total_correct / total_samples
    print(f'\nTest Accuracy (Overall): {overall_accuracy:.2f}% ({total_correct}/{total_samples})')
    
    total_time = time.time() - start_time
    print(f"\nTotal testing time: {total_time:.2f}s")
    
    return overall_accuracy


class EncFullyConnectedNet:
    """Encrypted version of the fully connected neural network"""
    def __init__(self, torch_nn):
        # Extract weights and biases from PyTorch model
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        self.fc3_weight = torch_nn.fc3.weight.T.data.tolist()
        self.fc3_bias = torch_nn.fc3.bias.data.tolist()
    
    def forward(self, enc_x):
        layer_times = {}
        
        # fc1 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        layer_times["fc1"] = time.time() - start
        
        # Square activation
        start = time.time()
        enc_x.square_()
        layer_times["square1"] = time.time() - start
        
        # fc2 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        layer_times["fc2"] = time.time() - start
        
        # Square activation
        start = time.time()
        enc_x.square_()
        layer_times["square2"] = time.time() - start
        
        # fc3 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc3_weight) + self.fc3_bias
        layer_times["fc3"] = time.time() - start
        
        return enc_x, layer_times
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def setup_encryption_context(bits_scale=26):
    """Set up the encryption context for TenSEAL"""
    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )

    # Set the scale
    context.global_scale = pow(2, bits_scale)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    
    return context


def enc_test(context, enc_model, test_loader, criterion, max_samples=50):
    """Test the encrypted model"""
    print("\n" + "="*50)
    print("TESTING ENCRYPTED MODEL")
    print("="*50)
    
    total_start_time = time.time()
    
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    
    # Collect layer timing information
    layer_timing = {
        "fc1": 0.0,
        "square1": 0.0,
        "fc2": 0.0,
        "square2": 0.0,
        "fc3": 0.0,
        "encryption": 0.0,
        "decryption": 0.0,
        "inference": 0.0
    }
    
    # Counter for processed samples
    cnt = 0
    
    for data, target in test_loader:
        # Encoding and encryption
        encryption_start = time.time()
        x_enc = ts.ckks_vector(context, data.view(-1, 784)[0].tolist())
        layer_timing["encryption"] += time.time() - encryption_start
        
        # Encrypted evaluation
        inference_start = time.time()
        enc_output, batch_layer_timing = enc_model(x_enc)
        layer_timing["inference"] += time.time() - inference_start
        
        # Update layer timing
        for key, value in batch_layer_timing.items():
            layer_timing[key] += value
        
        # Decryption of result
        decryption_start = time.time()
        output = enc_output.decrypt()
        layer_timing["decryption"] += time.time() - decryption_start
        
        output = torch.tensor(output).view(1, -1)

        # Compute loss
        loss = criterion(output, target)
        test_loss += loss.item()
        
        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        
        # Compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        
        # Calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
        
        cnt += 1
        if cnt % 10 == 0:
            print(f"Processed {cnt}/{max_samples} samples...")
        
        if cnt >= max_samples:
            break

    # Calculate and print avg test loss
    processed_samples = sum(class_total)
    test_loss = test_loss / processed_samples
    print(f'\nTest Loss: {test_loss:.6f}')
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for label in range(10):
        if class_total[label] > 0:
            accuracy = 100 * class_correct[label] / class_total[label]
            print(f'Class {label}: {accuracy:.2f}% ({int(class_correct[label])}/{int(class_total[label])})')
        else:
            print(f'Class {label}: No samples')
    
    # Print overall accuracy
    total_correct = int(np.sum(class_correct))
    overall_accuracy = 100 * total_correct / processed_samples
    print(f'\nEncrypted Test Accuracy (Overall): {overall_accuracy:.2f}% ({total_correct}/{processed_samples})')
    
    # Print timing information
    total_time = time.time() - total_start_time
    print(f"\nTotal encrypted testing time: {total_time:.2f}s")
    print("\nLayer-wise timing breakdown:")
    for layer, timing in layer_timing.items():
        percentage = (timing / total_time) * 100
        print(f"  {layer}: {timing:.4f}s ({percentage:.2f}%)")
    
    avg_inference_time = total_time / processed_samples
    print(f"\nAverage time per sample: {avg_inference_time:.4f}s")
    
    return overall_accuracy


def main():
    # Load data
    batch_size = 64
    _, _, train_loader, test_loader = load_data(batch_size=batch_size)
    _, _, _, test_loader_single = load_data(batch_size=batch_size, single_batch=True)
    
    # Create and train model
    model = FullyConnectedNet(hidden=64)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train(model, train_loader, criterion, optimizer, n_epochs=10)
    
    # Test the standard model
    standard_accuracy = test(model, test_loader, criterion)
    
    # Save the model (optional)
    # torch.save(model, "./mnist_fc_net.pth")
    
    # Setup encryption context
    context = setup_encryption_context(bits_scale=26)
    
    # Create encrypted model
    enc_model = EncFullyConnectedNet(model)
    
    # Test the encrypted model
    encrypted_accuracy = enc_test(context, enc_model, test_loader_single, criterion, max_samples=50)
    
    # Print comparison
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Standard model accuracy: {standard_accuracy:.2f}%")
    print(f"Encrypted model accuracy: {encrypted_accuracy:.2f}%")
    print(f"Accuracy difference: {abs(standard_accuracy - encrypted_accuracy):.2f}%")


if __name__ == "__main__":
    main()