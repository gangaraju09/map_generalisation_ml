import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import pandas as pd
import numpy as np
import csv
from helper import CustomDataset, Net, CustomDatasetParquet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
torch.manual_seed(40)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Standardization, Normalization, No scaling
# 0 - No scaling, 1 - Standardization, 2 - Normalization
SCALING_FLAG=2

# TODO: Extend to all states by the end
all_states = ['idaho.parq', 'Maine.parq']

# all_states = ['California.parq', 'Florida.parq', 'idaho.parq',
#                     'Louisiana.parq', 'Maine.parq', 'NorthCarolina.parq',
#                     'Texas.parq']

# Change the train/val and test set to account for all combinations (total 7 combinations)
train_val_states = all_states[:-1]
test_state = all_states[-1]

train_val_df = pd.DataFrame()
test_df = pd.read_parquet('Vertices_Labels/' + test_state)
# Read entire training data

for file in train_val_states:
    print(f"Reading {file}---")
    df = pd.read_parquet('Vertices_Labels/' + file)
    train_val_df = train_val_df.append(df, ignore_index=True)

# Moving 'CASE' as last column 
columns = train_val_df.columns.tolist()
columns.append(columns.pop(2))
train_val_df = train_val_df[columns]
test_df = test_df[columns]

# TODO: Dropping NaNs rows
train_val_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# print(train_val_df.columns)
# print(train_val_df.head())
# print("-"*30)

def scale_features(SCALING_FLAG, train_val_df):
    # Different types of scaling features
    # Decision trees don't need scaling, NN needs, so have both
    if SCALING_FLAG == 0:
        rescaled_train_val_df = train_val_df
        rescaled_test_df = test_df
    if SCALING_FLAG == 1:
        columns_to_standardize = train_val_df.columns.tolist()[:-1]
        rescaled_train_val_df, rescaled_test_df = train_val_df.copy(), test_df.copy()
        std_scaler = StandardScaler()
        rescaled_train_val_df[columns_to_standardize] = std_scaler.fit_transform(rescaled_train_val_df[columns_to_standardize])
        rescaled_test_df[columns_to_standardize] = std_scaler.transform(rescaled_test_df[columns_to_standardize])
    if SCALING_FLAG == 2:
        columns_to_standardize = train_val_df.columns.tolist()[:-1]
        rescaled_train_val_df, rescaled_test_df = train_val_df.copy(), test_df.copy()
        std_scaler = MinMaxScaler()
        rescaled_train_val_df[columns_to_standardize] = std_scaler.fit_transform(rescaled_train_val_df[columns_to_standardize])
        rescaled_test_df[columns_to_standardize] = std_scaler.transform(rescaled_test_df[columns_to_standardize])
    return rescaled_train_val_df, rescaled_test_df

rescaled_train_val_df, rescaled_test_df = scale_features(SCALING_FLAG, train_val_df)

# Dealing with Class Imbalance ratio by sampling the minority class more!
labels_unique, count = np.unique(rescaled_train_val_df['case'], return_counts=True)
print(f"Number of samples with their counts are: {labels_unique},{count}")
class_weights = [sum(count)/c for c in count]
print(f"Class weights needed for resampling: {class_weights}")

combined_dataset = CustomDatasetParquet(rescaled_train_val_df)
# Train and Validation splits
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - (train_size)

test_dataset = CustomDatasetParquet(rescaled_test_df)
test_loader = DataLoader(test_dataset, batch_size=8192)

print(f"Train and val size are {train_size}, {val_size}")
train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
print("Random split done")

example_weights = [class_weights[e['output']] for e in train_dataset]
print(f"Example weights unique items are: {set(example_weights)}")

# indices = [i for i, x in enumerate(example_weights) if x == 329.3644859813084]
# print(len(indices))

print(f"Length of training and validation dataset is {train_size}, {val_size}")
sampler = WeightedRandomSampler(example_weights, train_size, replacement=True)
# print(f"First 10 elements in sampler are: {list(sampler)[:10]}")

# Set up the data loaders
train_loader = DataLoader(train_dataset, batch_size=8196, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=8196)

### Code to verify that sampler is working and we are getting samples in a balanced way!!
# a = 0
# for data in train_loader:
#   print(np.unique(data['output'], return_counts=True))
#   a += 1
#   if a == 10:
#     break

# NN
net = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
net.to(device)

def train_validate_loop(num_epochs, model_name):
    min_loss = float('inf')
# Train the neural network
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            data['lat'] = data['lat'].float().to(device)
            data['long'] = data['long'].float().to(device)
            data['len_forward'] = data['len_forward'].float().to(device)
            data['len_backward'] = data['len_backward'].float().to(device)
            data['angle'] = data['angle'].float().to(device)
            data['offset'] = data['offset'].float().to(device)
            data['area'] = data['area'].float().to(device)
            data['arc'] = data['arc'].float().to(device)
            data['ratio1'] = data['ratio1'].float().to(device)
            data['ratio2'] = data['ratio2'].float().to(device)
            data['output'] = data['output'].float().to(device)
            inputs = torch.stack([data['lat'], data['long'], data['len_forward'], data['len_backward'],
                                data['angle'], data['offset'], data['area'], data['arc'],
                                data['ratio1'], data['ratio2']], dim=1)
            labels = torch.tensor(data['output']).unsqueeze(1)
            # print(inputs)
            # print(labels[:10])
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs[:10])
            # print("-------------------------------------")
            # break
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d], Training Loss: %.4f' % (epoch+1, running_loss/len(train_loader)))
        
        # Evaluate the neural network on the test set
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data['lat'] = data['lat'].float().to(device)
                data['long'] = data['long'].float().to(device)
                data['len_forward'] = data['len_forward'].float().to(device)
                data['len_backward'] = data['len_backward'].float().to(device)
                data['angle'] = data['angle'].float().to(device)
                data['offset'] = data['offset'].float().to(device)
                data['area'] = data['area'].float().to(device)
                data['arc'] = data['arc'].float().to(device)
                data['ratio1'] = data['ratio1'].float().to(device)
                data['ratio2'] = data['ratio2'].float().to(device)
                data['output'] = data['output'].float().to(device)
                inputs = torch.stack([data['lat'], data['long'], data['len_forward'], data['len_backward'],
                                    data['angle'], data['offset'], data['area'], data['arc'],
                                    data['ratio1'], data['ratio2']], dim=1)
                labels = torch.tensor(data['output']).unsqueeze(1)
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if running_loss < min_loss:
                min_loss = running_loss
                torch.save(net, model_name)

            print('Epoch [%d], Validation Loss: %.4f' % (epoch+1, running_loss/len(val_loader)))
            print(f"Total number of Validation points are: {total} and correct points are {correct}")
            print('Accuracy of the network on the validation data: %d %%' % (100 * correct / total))


def test_loop(model_name, output_file, col_name):
    net = torch.load(model_name)
    total, correct = 0, 0
    predicted_values = []
    with torch.no_grad():
        for data in test_loader:
            data['lat'] = data['lat'].float().to(device)
            data['long'] = data['long'].float().to(device)
            data['len_forward'] = data['len_forward'].float().to(device)
            data['len_backward'] = data['len_backward'].float().to(device)
            data['angle'] = data['angle'].float().to(device)
            data['offset'] = data['offset'].float().to(device)
            data['area'] = data['area'].float().to(device)
            data['arc'] = data['arc'].float().to(device)
            data['ratio1'] = data['ratio1'].float().to(device)
            data['ratio2'] = data['ratio2'].float().to(device)
            data['output'] = data['output'].float().to(device)
            inputs = torch.stack([data['lat'], data['long'], data['len_forward'], data['len_backward'],
                                data['angle'], data['offset'], data['area'], data['arc'],
                                data['ratio1'], data['ratio2']], dim=1)
            labels = torch.tensor(data['output']).unsqueeze(1)
            outputs = net(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_values.extend(predicted.cpu().numpy().astype(int).flatten().tolist())

    print(f"Total number of test points are: {total} and correct points are {correct}")
    print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))

    test_df[col_name] = predicted_values
    test_df.to_parquet(output_file, index=False)

train_validate_loop(num_epochs=20, model_name="idaho_maine.pt")
test_loop(model_name="idaho_maine.pt", output_file='maine_pred_normalization.parquet', col_name='normalized_preds')