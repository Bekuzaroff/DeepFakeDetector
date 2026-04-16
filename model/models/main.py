import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment and (split == 'train')
        self.samples = self._load_samples(split)

        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])

    def _load_samples(self, split):
        samples = []
        for label_name, label in [('real', 0), ('fake', 1)]:
            class_dir = self.data_dir / split / label_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        samples.append((str(img_path), label))
        logger.info(f"Loaded {len(samples)} samples for {split} split")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.augment:
                image = self.augment_transform(image)
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.long)

class ViTModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=2, dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class SwinModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=2, dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class DeiTModel(nn.Module):
    def __init__(self, model_name='deit_base_patch16_224', num_classes=2, dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class ModelManager:
    def __init__(self):
        self.model_configs = {
            'vit_base_patch16_224': ViTModel(),
            'swin_base_patch4_window7_224': SwinModel(),
            'deit_base_patch16_224': DeiTModel()
        }
        self.models = {name: model.to(device) for name, model in self.model_configs.items()}
        logger.info(f"Initialized models: {list(self.models.keys())}")

    def get_model(self, name):
        return self.models[name]

    def train_all(self, data_dir, num_epochs=10, batch_size=16, lr=1e-4):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = DeepfakeDataset(data_dir, 'train', transform, augment=True)
        val_dataset = DeepfakeDataset(data_dir, 'val', transform, augment=False)

        class_counts = np.bincount([s[1] for s in train_dataset.samples])
        weights = [float(1.0 / class_counts[label]) for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for name, model in self.models.items():
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            best_auc = 0
            print(f"\nTraining {name}...")
            for epoch in range(num_epochs):
                model.train()
                total_loss, correct, total = 0, 0, 0
                for data, target in tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{num_epochs}"):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    pred = output.argmax(1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                acc = 100. * correct / total
                val_loss, val_acc, val_auc = self.evaluate(model, val_loader, criterion)
                print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}")
                if val_auc > best_auc:
                    torch.save(model.state_dict(), f"best_{name}.pth")
                    best_auc = val_auc

    def evaluate(self, model, loader, criterion):
        model.eval()
        val_loss = 0
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                probs = torch.softmax(output, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(probs.argmax(dim=1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        acc = accuracy_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)
        return val_loss / len(loader), acc * 100, auc

if __name__ == '__main__':
    data_path = "deepfake_dataset"
    manager = ModelManager()
    manager.train_all(data_path, num_epochs=10)
