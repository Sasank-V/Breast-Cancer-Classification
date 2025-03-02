import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # First Conv Block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Second Conv Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Third Conv Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Fourth Conv Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

def get_pretrained_model(model_name, num_classes=2):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model

class ModelTrainer:
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_score = 0
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        predictions = []
        true_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(true_labels, predictions)
        epoch_f1 = f1_score(true_labels, predictions, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def evaluate(self, val_loader):
        self.model.eval()
        predictions = []
        true_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(true_labels, predictions)
        val_f1 = f1_score(true_labels, predictions, average='weighted')
        val_auc = roc_auc_score(true_labels, predictions)
        
        return val_loss, val_acc, val_f1, val_auc, predictions, true_labels

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

def train_and_evaluate(data_dir, model_type='basic_cnn', num_epochs=30):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare data
    image_paths = []
    labels = []
    for image_name in os.listdir(data_dir):
        if image_name.endswith('.png'):
            parts = image_name.split('_')
            tumor_class = parts[1]
            image_paths.append(os.path.join(data_dir, image_name))
            labels.append(1 if tumor_class == 'M' else 0)

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create datasets
    train_dataset = BreastCancerDataset(train_paths, train_labels, transform_train)
    val_dataset = BreastCancerDataset(val_paths, val_labels, transform_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model selection
    if model_type == 'basic_cnn':
        model = BasicCNN()
    else:
        model = get_pretrained_model(model_type)
    
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # Initialize trainer
    trainer = ModelTrainer(model, device, criterion, optimizer, scheduler)

    # Training history
    train_losses, train_accs, train_f1s = [], [], []
    val_losses, val_accs, val_f1s, val_aucs = [], [], [], []

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_loss, val_acc, val_f1, val_auc, predictions, true_labels = trainer.evaluate(val_loader)
        
        # Update learning rate
        if trainer.scheduler:
            trainer.scheduler.step(val_acc)

        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')

        # Save best model
        if val_acc > trainer.best_val_score:
            trainer.best_val_score = val_acc
            torch.save(model.state_dict(), f'best_model_{model_type}.pth')

    # Plot metrics
    plot_metrics(train_losses, val_losses, 'Loss')
    plot_metrics(train_accs, val_accs, 'Accuracy')
    plot_metrics(train_f1s, val_f1s, 'F1 Score')
    plot_confusion_matrix(true_labels, predictions)

    return model, trainer.best_val_score

def hyperparameter_optimization(data_dir, model_type='basic_cnn'):
    learning_rates = [0.001]
    batch_sizes = [16]
    best_params = {'lr': None, 'batch_size': None}
    best_score = 0

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\nTrying lr={lr}, batch_size={batch_size}")
            _, val_score = train_and_evaluate(
                data_dir,
                model_type=model_type,
                num_epochs=5  # Reduced epochs for faster optimization
            )
            
            if val_score > best_score:
                best_score = val_score
                best_params['lr'] = lr
                best_params['batch_size'] = batch_size

    print("\nBest parameters:")
    print(f"Learning rate: {best_params['lr']}")
    print(f"Batch size: {best_params['batch_size']}")
    return best_params

if __name__ == "__main__":
    data_dir = "../datasets/all_images"
    
    # Train and evaluate basic CNN
    print("Training Basic CNN...")
    train_and_evaluate(data_dir, model_type='basic_cnn')
    
    # Train and evaluate pretrained models
    for model_type in ['resnet50', 'vgg16', 'mobilenet']:
        print(f"\nTraining {model_type}...")
        train_and_evaluate(data_dir, model_type=model_type)
    
    # Perform hyperparameter optimization
    print("\nOptimizing hyperparameters...")
    best_params = hyperparameter_optimization(data_dir)