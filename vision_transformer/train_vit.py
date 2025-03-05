import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset Class
class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Load dataset
def load_dataset(data_dir):
    image_paths = []
    labels = []
    for image_name in os.listdir(data_dir):
        if image_name.endswith('.png'):
            parts = image_name.split('_')
            tumor_class = parts[1]  # Extract class from filename (e.g., 'M' for malignant, 'B' for benign)
            image_paths.append(os.path.join(data_dir, image_name))
            labels.append(1 if tumor_class == 'M' else 0)  # 1 for malignant, 0 for benign
    return image_paths, labels

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "../dataset/all_images"  # Path to the dataset
image_paths, labels = load_dataset(data_dir)

# Split dataset into training and validation sets (80-20 split)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Create datasets
train_dataset = BreastCancerDataset(train_paths, train_labels, transform)
val_dataset = BreastCancerDataset(val_paths, val_labels, transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Load and cache the model weights before training
print("Caching model weights...")
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, 2)  # Modify final layer for binary classification
model = model.to(device)
print("Model weights cached successfully.")

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(true_labels, predictions)
    epoch_f1 = f1_score(true_labels, predictions, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    predictions = []
    true_labels = []
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(dataloader)
    val_acc = accuracy_score(true_labels, predictions)
    val_f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Check if both classes are present in true_labels
    if len(np.unique(true_labels)) == 2:
        val_auc = roc_auc_score(true_labels, predictions)
    else:
        print("Warning: Only one class present in validation set. Skipping ROC AUC calculation.")
        val_auc = None
    
    return val_loss, val_acc, val_f1, val_auc, predictions, true_labels

# Plotting functions
def plot_metrics(train_metrics, val_metrics, metric_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} over epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
 
def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Training loop
num_epochs = 10
best_val_acc = 0.0

# Store metrics for plotting
train_losses = []
val_losses = []
train_accs = []
val_accs = []
train_f1s = []
val_f1s = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Train
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc, val_f1, val_auc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)
    
    # Update learning rate
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_vit_model.pth')
    
    # Print metrics
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}", end=" ")
    if val_auc is not None:
        print(f"Val AUC: {val_auc:.4f}")
    else:
        print("Val AUC: N/A (Only one class present)")

# Load best model
model.load_state_dict(torch.load('best_vit_model.pth'))

# Final evaluation
val_loss, val_acc, val_f1, val_auc, val_preds, val_labels = validate(model, val_loader, criterion, device)
print(f"Final Validation Metrics - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}", end=" ")
if val_auc is not None:
    print(f"AUC: {val_auc:.4f}")
else:
    print("AUC: N/A (Only one class present)")

# Confusion matrix
cm = confusion_matrix(val_labels, val_preds)
print("Confusion Matrix:")
print(cm)

# Plotting the metrics
plot_metrics(train_losses, val_losses, 'Loss')
plot_metrics(train_accs, val_accs, 'Accuracy')
plot_metrics(train_f1s, val_f1s, 'F1 Score')

# Plot confusion matrix
plot_confusion_matrix(val_labels, val_preds)