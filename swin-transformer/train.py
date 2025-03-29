import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModelForImageClassification  # For ViT

# -----------------------------
# Utility Functions
# -----------------------------
def extract_magnification_level(filename):
    """
    Extract the magnification level from a filename.
    Expected format:
      <BIOPSY_PROCEDURE>_<TUMOR_CLASS>_<TUMOR_TYPE>_<YEAR>-<SLIDE_ID>-<MAGNIFICATION>-<SEQ>.png
    """
    name_without_ext, _ = os.path.splitext(filename)
    parts = name_without_ext.split('_')
    if parts:
        sub_parts = parts[-1].split('-')
        if len(sub_parts) >= 3:
            return sub_parts[-2]
    return "unknown"

# -----------------------------
# PyTorch Dataset & Transforms
# -----------------------------
class HistopathologyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and convert to RGB
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_transforms():
    # Resize to 224x224, convert to tensor, and apply ImageNet normalization.
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# -----------------------------
# Model Definitions
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_labels=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        # After four pooling layers, the image size 224 becomes 224/16 = 14.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_swin_transformer_model(num_labels=2, fine_tune=True):
    """
    Build the Swin Transformer Tiny model using TorchVision.
    This method loads the pretrained swin_t model and replaces its head.
    """
    # Load pretrained Swin Transformer Tiny from TorchVision.
    model = models.swin_t(weights="DEFAULT")
    # Optionally fine-tune all layers.
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
    # Replace the classification head to match the desired number of labels.
    # Note: model.head.in_features provides the input feature size of the current head.
    model.head = nn.Linear(in_features=model.head.in_features, out_features=num_labels, bias=True)
    return model

# -----------------------------
# Training & Evaluation Function
# -----------------------------
def train_model(model, train_loader, val_loader, num_epochs, device, model_name="model", magnification="NA"):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    # To store metrics per epoch
    train_metrics = {"loss": [], "accuracy": [], "f1": []}
    val_metrics   = {"loss": [], "accuracy": [], "f1": []}
    
    for epoch in range(num_epochs):
        # ----- Training phase -----
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # For models like ViT from Hugging Face, extract logits if needed.
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs)
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs  # Fallback if output is already a tensor
                
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_train_labels, all_train_preds)
        epoch_f1 = f1_score(all_train_labels, all_train_preds, average="weighted")
        train_metrics["loss"].append(epoch_loss)
        train_metrics["accuracy"].append(epoch_acc)
        train_metrics["f1"].append(epoch_f1)
        
        # ----- Validation phase -----
        model.eval()
        val_running_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs)
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                loss = criterion(logits, labels)
                val_running_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = accuracy_score(all_val_labels, all_val_preds)
        val_epoch_f1 = f1_score(all_val_labels, all_val_preds, average="weighted")
        val_metrics["loss"].append(val_epoch_loss)
        val_metrics["accuracy"].append(val_epoch_acc)
        val_metrics["f1"].append(val_epoch_f1)
        
        print(f"{model_name} ({magnification}) Epoch {epoch+1}/{num_epochs} => "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, F1: {val_epoch_f1:.4f}")
    
    # Compute confusion matrix on entire validation set (after final epoch)
    conf_matrix = confusion_matrix(all_val_labels, all_val_preds)
    return train_metrics, val_metrics, conf_matrix

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.close()

# -----------------------------
# Experiment Function
# -----------------------------
def run_experiments(data_dir, magnification_levels=["40", "100", "200", "400"], num_epochs=5, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms()
    
    # Organize images by magnification level.
    images_by_level = {mag: [] for mag in magnification_levels}
    labels_by_level = {mag: [] for mag in magnification_levels}
    
    for fname in os.listdir(data_dir):
        if fname.endswith('.png'):
            mag = extract_magnification_level(fname)
            if mag in magnification_levels:
                parts = fname.split('_')
                if len(parts) >= 2:
                    tumor_class = parts[1]
                    images_by_level[mag].append(os.path.join(data_dir, fname))
                    labels_by_level[mag].append(1 if tumor_class == "M" else 0)
                    
    results = {}
    
    for mag in magnification_levels:
        if len(images_by_level[mag]) < 5:
            print(f"Not enough images for magnification level {mag}, skipping...")
            continue
        
        print("=" * 50)
        print(f"Magnification level: {mag}x")
        print("=" * 50)
        
        img_paths = images_by_level[mag]
        labs = labels_by_level[mag]
        
        # Split into training and validation sets (stratified)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            img_paths, labs, test_size=0.2, random_state=42, stratify=labs
        )
        
        train_dataset = HistopathologyDataset(train_paths, train_labels, transform=transform)
        val_dataset   = HistopathologyDataset(val_paths, val_labels, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        results[mag] = {}
        
        # # ----- Traditional CNN -----
        print("Training Traditional CNN...")
        cnn_model = SimpleCNN(num_labels=2)
        cnn_train_metrics, cnn_val_metrics, cnn_cm = train_model(
            cnn_model, train_loader, val_loader, num_epochs, device, model_name="CNN", magnification=f"{mag}x"
        )
        plot_confusion_matrix(cnn_cm, f"Confusion Matrix: CNN {mag}x", f"results_CNN_{mag}x.png")
        results[mag]["CNN"] = {
            "train": cnn_train_metrics,
            "val": cnn_val_metrics,
            "confusion_matrix": cnn_cm.tolist()
        }
        
        # ----- Vision Transformer (ViT) -----
        print("Training ViT...")
        vit_model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=2,
            id2label={0: "Benign", 1: "Malignant"},
            label2id={"Benign": 0, "Malignant": 1},
            ignore_mismatched_sizes=True
        )
        vit_model.to(device)
        vit_train_metrics, vit_val_metrics, vit_cm = train_model(
            vit_model, train_loader, val_loader, num_epochs, device, model_name="ViT", magnification=f"{mag}x"
        )
        plot_confusion_matrix(vit_cm, f"Confusion Matrix: ViT {mag}x", f"results_ViT_{mag}x.png")
        results[mag]["ViT"] = {
            "train": vit_train_metrics,
            "val": vit_val_metrics,
            "confusion_matrix": vit_cm.tolist()
        }
        
        # ----- Swin Transformer (TorchVision) -----
        print("Training Swin Transformer...")
        swin_model = build_swin_transformer_model(num_labels=2, fine_tune=True)
        swin_model.to(device)
        swin_train_metrics, swin_val_metrics, swin_cm = train_model(
            swin_model, train_loader, val_loader, num_epochs, device, model_name="Swin", magnification=f"{mag}x"
        )
        plot_confusion_matrix(swin_cm, f"Confusion Matrix: Swin {mag}x", f"results_Swin_{mag}x.png")
        results[mag]["Swin"] = {
            "train": swin_train_metrics,
            "val": swin_val_metrics,
            "confusion_matrix": swin_cm.tolist()
        }
        
    # Save overall comparative results to JSON
    with open("comparative_results_pytorch.json", "w") as f:
        json.dump(results, f, indent=4)
    return results

def composite_results_plot(magnifications=["40x", "100x", "200x", "400x"], models=["CNN", "ViT", "Swin"], results_dir="."):
    import matplotlib.image as mpimg
    n_rows = len(magnifications)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    for i, mag in enumerate(magnifications):
        for j, model in enumerate(models):
            filename = os.path.join(results_dir, f"results_{model}_{mag}.png")
            if os.path.exists(filename):
                img = mpimg.imread(filename)
                axes[i, j].imshow(img)
                axes[i, j].axis("off")
            else:
                axes[i, j].text(0.5, 0.5, f"Missing\n{filename}", ha="center", va="center")
                axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(model, fontsize=16)
        axes[i, 0].set_ylabel(mag, fontsize=16)
    
    plt.tight_layout()
    plt.savefig("composite_results_pytorch.png")
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Set the path to your images folder (adjust as necessary)
    data_dir = "../datasets/all_images"
    
    # Run experiments on the desired magnification levels
    results = run_experiments(data_dir, magnification_levels=["40", "100", "200", "400"], num_epochs=5, batch_size=8)
    
    # Create a composite plot for the confusion matrices
    composite_results_plot(magnifications=["40x", "100x", "200x", "400x"], models=["CNN", "ViT", "Swin"])
    
    print("Experiments complete. Check the JSON result file and composite_results_pytorch.png for comparisons.")
