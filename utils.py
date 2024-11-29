import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import pickle
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

def split_mnist_by_classes(dataset, num_tasks=5):
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    task_datasets = []
    classes_per_task = 10 // num_tasks
    for i in range(num_tasks):
        classes = range(i * classes_per_task, (i + 1) * classes_per_task)
        indices = [idx for c in classes for idx in class_indices[c]]
        task_datasets.append(Subset(dataset, indices))
    return task_datasets


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index].item() # (tensor, int)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.LazyConv2d(16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LazyLinear(2)
        #self.bottleneck = nn.Linear(128, 2)  # Bottleneck layer
        self.fc2 = nn.Linear(2, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  

        #x = self.relu(self.fc1(x))
        features = self.fc1(x)
        out = self.fc2(features)
        return out, features

class ReplayBufferReservoir:
    def __init__(self, capacity=500):
        self.buffer = []
        self.capacity = capacity
        self.seen_samples = 0  # Tracks the total number of samples seen so far

    def update(self, loader):
        for inputs, labels in loader:
            batch_size = inputs.size(0)
            for i in range(batch_size):
                self.seen_samples += 1
                sample = (inputs[i], labels[i])

                if len(self.buffer) < self.capacity:
                    self.buffer.append(sample)
                else:
                    replace_idx = np.random.randint(0, self.seen_samples - 1)
                    if replace_idx < self.capacity:
                        self.buffer[replace_idx] = sample

    def get_replay_loader(self, current_loader, replay_ratio=0.3):
        replay_size = int(replay_ratio * len(current_loader.dataset))
        if len(self.buffer) == 0:
            return current_loader

        replay_samples = np.random.choice(len(self.buffer), replay_size, replace=True)
        replay_data = [self.buffer[i] for i in replay_samples]
        replay_inputs, replay_labels = zip(*replay_data)
        replay_inputs = torch.stack(replay_inputs)
        replay_labels = torch.tensor(replay_labels, dtype=torch.long)

        replay_dataset = CustomDataset(replay_inputs, replay_labels)
        combined_dataset = torch.utils.data.ConcatDataset([current_loader.dataset, replay_dataset])
        return DataLoader(combined_dataset, batch_size=current_loader.batch_size, shuffle=True)


def train_model(model, data_loaders, tasks_test, optimizer, criterion, epochs=5, replay_buffer=None, path="test_animation", animation=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_anim = []
    
    for task_id, loader in enumerate(data_loaders):
        print(f"Training on Task {task_id + 1}")
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            if replay_buffer is not None:
                loader = replay_buffer.get_replay_loader(loader)

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            labs, feas = [], []
            for i in range(task_id, 5):
                test_datasets = torch.utils.data.ConcatDataset([tasks_test[i] for i in range(task_id+1)])
                test_loader = DataLoader(test_datasets, batch_size=100, shuffle=True)
                correct = 0
                total = 0
                lab, fea = [], []
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs, features = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        lab.append(labels)
                        fea.append(features)
                test_acc = 100 * correct / total
                labs.append(torch.concat(lab).cpu().numpy())
                feas.append(torch.concat(fea).cpu().numpy())
            data_anim.append((np.concatenate(labs),np.concatenate(feas)))            

        if replay_buffer is not None:
            replay_buffer.update(loader)
    
    with open(f'animations/{path}.pkl', 'wb') as file:
        pickle.dump(data_anim, file)
    
    if animation:
        create_animation(path)

def create_animation(path):
    with open(f'animations/{path}.pkl', 'rb') as file:
        data_anim = pickle.load(file)

    TASKS = 5
    EPOCHS = len(data_anim) // TASKS

    color_palette = [
        'red', 'blue', 'green', 'purple', 'orange', 
        'cyan', 'magenta', 'yellow', 'brown', 'pink'
    ]

    fig, axs = plt.subplots(1, TASKS, figsize=(25, 5))

    all_features = np.concatenate([d[1] for d in data_anim])
    x_min, x_max = all_features[:, 0].min() - 50, all_features[:, 0].max() + 50
    y_min, y_max = all_features[:, 1].min() - 50, all_features[:, 1].max() + 50

    scatters = []
    for task in range(TASKS):
        scatter = {}
        current_data = data_anim[0]
        unique_classes = np.unique(current_data[0])
        for i, c in enumerate(unique_classes):
            scatter[c] = axs[task].scatter(
                current_data[1][:, 0][current_data[0] == c],
                current_data[1][:, 1][current_data[0] == c],
                color=color_palette[i],
                label=f'Class {c}'
            )
        
        axs[task].set_xlim(x_min, x_max)
        axs[task].set_ylim(y_min, y_max)
        
        axs[task].set_title(f"Task {task}, Epoch 0")
        axs[task].legend()
        scatters.append(scatter)

    def update(frame):    
        for task in range(frame//EPOCHS, TASKS):
            current_data = data_anim[frame]
            unique_classes = np.unique(current_data[0])
            
            for existing_scatter in scatters[task].values():
                existing_scatter.remove()
            
            scatters[task] = {}
            for i, c in enumerate(unique_classes):
                scatters[task][c] = axs[task].scatter(
                    current_data[1][:, 0][current_data[0] == c],
                    current_data[1][:, 1][current_data[0] == c],
                    color=color_palette[i],
                    label=f'Class {c}'
                )
            
            axs[task].set_xlim(x_min, x_max)
            axs[task].set_ylim(y_min, y_max)
            
            axs[task].set_title(f"Task {task}, Epoch {frame}")
            axs[task].legend()
        
        return [scatter for task_scatters in scatters for scatter in task_scatters.values()]

    ani = FuncAnimation(
        fig, 
        update, 
        frames=TASKS * EPOCHS, 
        interval=1000, 
        repeat=False, 
        blit=False
    )

    ani.save(f'animations/{path}.gif', writer='pillow', fps=1)