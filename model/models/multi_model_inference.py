"""
main model inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection with advanced preprocessing"""
    
    def __init__(self, data_dir, split='train', transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment and (split == 'train')
        
        # Load image paths and labels
        self.samples = self._load_samples(split)
        
        # Advanced augmentation for training
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
    
    def _load_samples(self, split):
        """Load image paths and labels from directory structure"""
        samples = []
        
        # Expected structure: data_dir/split/real/, data_dir/split/fake/
        real_dir = self.data_dir / split / 'real'
        fake_dir = self.data_dir / split / 'fake'
        
        # Load real images
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append((str(img_path), 0))  # 0 for real
        
        # Load fake images
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append((str(img_path), 1))  # 1 for fake
        
        logger.info(f"Loaded {len(samples)} samples for {split} split")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            
            # Apply augmentation if training
            if self.augment:
                image = self.augment_transform(image)
            
            # Apply main transform
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return a default image
            default_img = torch.zeros(3, 224, 224)
            return default_img, torch.tensor(label, dtype=torch.long)

class EfficientNetModel(nn.Module):
    """EfficientNet-based deepfake detector"""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        # Replace classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ResNetModel(nn.Module):
    """ResNet-based deepfake detector"""
    
    def __init__(self, model_name='resnet50', num_classes=2, dropout=0.3):
        super().__init__()
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class EnhancedDeepfakeDetector:
    """Enhanced ensemble deepfake detection system"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _default_config(self):
        return {
            'models': {
                'efficientnet_b0': {'dropout': 0.3},
                'resnet50': {'dropout': 0.3},
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 20,
                'weight_decay': 1e-5,
                'patience': 5,
            },
            'ensemble': {
                'method': 'weighted_average',  # 'simple_average', 'weighted_average', 'stacking'
                'weights': None  # Will be computed based on validation performance
            }
        }
    
    def build_models(self):
        """Build all models in the ensemble"""
        logger.info("Building ensemble models...")
        
        for model_name, model_config in self.config['models'].items():
            try:
                if 'efficientnet' in model_name:
                    model = EfficientNetModel(model_name, dropout=model_config['dropout'])
                elif 'resnet' in model_name:
                    model = ResNetModel(model_name, dropout=model_config['dropout'])
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                model = model.to(device)
                self.models[model_name] = model
                
                # Setup optimizer and scheduler
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config['training']['learning_rate'],
                    weight_decay=self.config['training']['weight_decay']
                )
                self.optimizers[model_name] = optimizer
                
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=3, factor=0.5
                )
                self.schedulers[model_name] = scheduler
                
                logger.info(f"Built model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error building {model_name}: {e}")
    
    def create_data_loaders(self, data_dir):
        """Create data loaders for training and validation"""
        dataset_class = DeepfakeDataset
        
        # Create datasets
        train_dataset = dataset_class(data_dir, 'train', self.train_transform, augment=True)
        val_dataset = dataset_class(data_dir, 'val', self.val_transform, augment=False)
        
        # Handle class imbalance with weighted sampling
        train_labels = [sample[1] for sample in train_dataset.samples]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [float(class_weights[label]) for label in train_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        return train_loader, val_loader
    
    def train_single_model(self, model_name, train_loader, val_loader):
        """Train a single model"""
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        scheduler = self.schedulers[model_name]
        
        criterion = nn.CrossEntropyLoss()
        best_val_auc = 0
        patience_counter = 0
        
        # Создаем папку model если её нет
        os.makedirs("./model", exist_ok=True)
        
        logger.info(f"Training {model_name}...")
        
        for epoch in range(self.config['training']['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100. * train_correct / train_total:.2f}%"
                })
            
            # Validation phase
            val_loss, val_acc, val_auc = self.evaluate_model(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            logger.info(f"{model_name} - Epoch {epoch+1}: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), f"./model/best_{model_name}.pth")
                logger.info(f"Saved best model to ./model/best_{model_name}.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['patience']:
                    logger.info(f"Early stopping for {model_name}")
                    break
        
        
        best_model_path = f"./model/best_{model_name}.pth"
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f"Loaded best model from {best_model_path}")
        else:
            logger.warning(f"Best model file not found: {best_model_path}")
        
        return best_val_auc
    
    def evaluate_model(self, model, data_loader, criterion):
        """Evaluate a single model"""
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake
        
        val_loss /= len(data_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        val_auc = roc_auc_score(all_targets, all_probs)
        
        return val_loss, val_acc, val_auc
    
    def train_ensemble(self, data_dir):
        """Train the entire ensemble"""
        logger.info("Starting ensemble training...")
        
        # Build models
        self.build_models()
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(data_dir)
        
        # Train each model
        model_performance = {}
        for model_name in self.models.keys():
            val_auc = self.train_single_model(model_name, train_loader, val_loader)
            model_performance[model_name] = val_auc
        
        # Compute ensemble weights based on performance
        if self.config['ensemble']['method'] == 'weighted_average':
            total_performance = sum(model_performance.values())
            self.config['ensemble']['weights'] = {
                name: perf / total_performance 
                for name, perf in model_performance.items()
            }
        
        logger.info("Ensemble training completed!")
        logger.info(f"Model performance: {model_performance}")
        logger.info(f"Ensemble weights: {self.config['ensemble']['weights']}")
    
    def predict_ensemble(self, image_path):
        """Make ensemble prediction on a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.val_transform(image).unsqueeze(0).to(device)
            
            predictions = {}
            all_probs = []
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    output = model(image_tensor)
                    probs = torch.softmax(output, dim=1)
                    fake_prob = probs[0, 1].item()
                    pred = int(fake_prob > 0.5)
                    
                    predictions[model_name] = {
                        'prediction': pred,
                        'fake_probability': fake_prob,
                        'confidence': max(fake_prob, 1 - fake_prob),
                        'label': 'FAKE' if pred == 1 else 'REAL'
                    }
                    
                    all_probs.append(fake_prob)
            
            # Ensemble prediction
            if self.config['ensemble']['method'] == 'simple_average':
                ensemble_prob = np.mean(all_probs)
            elif self.config['ensemble']['method'] == 'weighted_average':
                weights = [self.config['ensemble']['weights'][name] for name in self.models.keys()]
                ensemble_prob = np.average(all_probs, weights=weights)
            else:
                ensemble_prob = np.mean(all_probs)
            
            ensemble_pred = int(ensemble_prob > 0.5)
            ensemble_confidence = max(float(ensemble_prob), 1 - float(ensemble_prob))
            
            predictions['Ensemble'] = {
                'prediction': ensemble_pred,
                'fake_probability': ensemble_prob,
                'confidence': ensemble_confidence,
                'label': 'FAKE' if ensemble_pred == 1 else 'REAL'
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {e}")
            return {}
    
    def evaluate_ensemble(self, data_dir, split='test'):
        """Evaluate ensemble on test set"""
        logger.info(f"Evaluating ensemble on {split} set...")
        
        # Create test dataset
        test_dataset = DeepfakeDataset(data_dir, split, self.val_transform, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_targets = []
        all_ensemble_preds = []
        all_ensemble_probs = []
        model_predictions = {name: [] for name in self.models.keys()}
        
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            all_targets.extend(target.cpu().numpy())
            
            model_probs = []
            for model_name, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    fake_probs = probs[:, 1].cpu().numpy()
                    model_predictions[model_name].extend(fake_probs)
                    model_probs.append(fake_probs)
            
            # Ensemble prediction
            model_probs = np.array(model_probs)
            if self.config['ensemble']['method'] == 'weighted_average':
                weights = [self.config['ensemble']['weights'][name] for name in self.models.keys()]
                ensemble_probs = np.average(model_probs, axis=0, weights=weights)
            else:
                ensemble_probs = np.mean(model_probs, axis=0)
            
            all_ensemble_probs.extend(ensemble_probs)
            all_ensemble_preds.extend((ensemble_probs > 0.5).astype(int))
        
        # Generate comprehensive evaluation report
        self.generate_evaluation_report(
            all_targets, all_ensemble_preds, all_ensemble_probs, model_predictions
        )
    
    def generate_evaluation_report(self, targets, ensemble_preds, ensemble_probs, model_predictions):
        """Generate comprehensive evaluation report with visualizations"""
        
        # Create results directory
        results_dir = Path("./model/evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Ensemble metrics
        ensemble_acc = accuracy_score(targets, ensemble_preds)
        ensemble_auc = roc_auc_score(targets, ensemble_probs)
        
        print(f"\n{'='*60}")
        print("ENSEMBLE EVALUATION REPORT")
        print(f"{'='*60}")
        print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
        print(f"Ensemble AUC: {ensemble_auc:.4f}")
        
        # Individual model performance
        print(f"\n{'Individual Model Performance'}")
        print(f"{'-'*40}")
        
        model_metrics = {}
        for model_name, probs in model_predictions.items():
            preds = (np.array(probs) > 0.5).astype(int)
            acc = accuracy_score(targets, preds)
            auc = roc_auc_score(targets, probs)
            model_metrics[model_name] = {'accuracy': acc, 'auc': auc}
            print(f"{model_name:<20} | Acc: {acc:.4f} | AUC: {auc:.4f}")
        
        # Confusion Matrix
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(targets, ensemble_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(targets, ensemble_probs)
        plt.plot(fpr, tpr, label=f'Ensemble (AUC = {ensemble_auc:.4f})', linewidth=2)
        
        for model_name, probs in model_predictions.items():
            fpr_model, tpr_model, _ = roc_curve(targets, probs)
            auc_model = model_metrics[model_name]['auc']
            plt.plot(fpr_model, tpr_model, label=f'{model_name} (AUC = {auc_model:.4f})', alpha=0.7)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        # Precision-Recall Curve
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(targets, ensemble_probs)
        plt.plot(recall, precision, label='Ensemble', linewidth=2)
        
        for model_name, probs in model_predictions.items():
            precision_model, recall_model, _ = precision_recall_curve(targets, probs)
            plt.plot(recall_model, precision_model, label=model_name, alpha=0.7)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed report
        report = {
            'ensemble_metrics': {
                'accuracy': ensemble_acc,
                'auc': ensemble_auc,
                'classification_report': classification_report(targets, ensemble_preds, output_dict=True)
            },
            'individual_models': model_metrics,
            'config': self.config
        }
        
        with open(results_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed results saved to {results_dir}/")
    
    def save_model(self, save_path):
        """Save the entire ensemble"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_data = {
            'config': self.config,
            'model_state_dicts': {name: model.state_dict() for name, model in self.models.items()}
        }
        torch.save(save_data, save_path)
        logger.info(f"Ensemble saved to {save_path}")
    
    def load_model(self, load_path):
        """Load a saved ensemble"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        save_data = torch.load(load_path, map_location=device)
        self.config = save_data['config']
        
        # Rebuild models
        self.build_models()
        
        # Load state dicts
        for name, state_dict in save_data['model_state_dicts'].items():
            if name in self.models:
                self.models[name].load_state_dict(state_dict)
                logger.info(f"Loaded model: {name}")
        
        logger.info(f"Ensemble loaded from {load_path}")
def main():
    """Main execution function with comprehensive workflow"""
    print("Enhanced Deepfake Detection System")
    print("=====================================")
    
    # folder structure
    os.makedirs("./model", exist_ok=True)
    os.makedirs("./model/evaluation_results", exist_ok=True)
    
    # Configuration
    config = {
        'models': {
            'efficientnet_b0': {'dropout': 0.3},
            'resnet50': {'dropout': 0.3},
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'weight_decay': 1e-5,
            'patience': 3,
        },
        'ensemble': {
            'method': 'weighted_average',
            'weights': None
        }
    }
    
    # Initialize detector
    detector = EnhancedDeepfakeDetector(config)
    
    data_dir = "./model/data"
    
    if os.path.exists(data_dir):
        print(f"Training ensemble models from: {data_dir}")
        detector.train_ensemble(data_dir)
        
        print("Saving trained models...")
        detector.save_model("./model/enhanced_deepfake_ensemble.pth")
        
        print("Evaluating on test set...")
        detector.evaluate_ensemble(data_dir, 'test')
        
    else:
        print(f"Dataset directory '{data_dir}' not found.")
        print(f"Current directory: {os.getcwd()}")
        print("\nRequired structure:")
        print("model/")
        print("├── data/")
        print("│   ├── train/")
        print("│   │   ├── real/")
        print("│   │   └── fake/")
        print("│   ├── val/")
        print("│   │   ├── real/")
        print("│   │   └── fake/")
        print("│   └── test/")
        print("│       ├── real/")
        print("│       └── fake/")
        print("├── evaluation_results/")
        print("└── (models will be saved here)")
    
    # Demo inference
    sample_image = "./model/cathedral.jpg"
    if os.path.exists(sample_image):
        print(f"Testing on sample image: {sample_image}")
        results = detector.predict_ensemble(sample_image)
        
        if results:
            print("\n" + "="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            
            for model_name, result in results.items():
                res = "-" if result['label'] == 'FAKE' else "+"
                print(f"{res} {model_name:<20} | {result['label']:<4} | "
                      f"Confidence: {result['confidence']:.4f} | "
                      f"Fake Prob: {result['fake_probability']:.4f}")
        else:
            print(" No predictions generated")
    else:
        print(f"\nSample image not found: {sample_image}")
    
    print("\nEnhanced Deepfake Detection System Ready!")
if __name__ == "__main__":
    main()