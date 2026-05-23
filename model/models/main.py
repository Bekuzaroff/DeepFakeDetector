"""
inference of our ensemble
"""

from multi_model_inference import EnhancedDeepfakeDetector
import torch
import os

if __name__ == "__main__":
    detector = EnhancedDeepfakeDetector()
    
    # create architecture of models
    detector.build_models()
    
    # save weights
    models_loaded = False
    
    if os.path.exists("model/best_efficientnet_b0.pth"):
        detector.models['efficientnet_b0'].load_state_dict(
            torch.load("model/best_efficientnet_b0.pth", map_location='cuda')
        )
        detector.models['efficientnet_b0'].eval()
        print("Loaded EfficientNet model")
        models_loaded = True
    
    if os.path.exists("model/best_resnet50.pth"):
        detector.models['resnet50'].load_state_dict(
            torch.load("model/best_resnet50.pth", map_location='cuda')
        )
        detector.models['resnet50'].eval()
        print("Loaded ResNet50 model")
        models_loaded = True
    
    if not models_loaded:
        print("No model checkpoints found. Please train models first:")
        print("   - best_efficientnet_b0.pth")
        print("   - best_resnet50.pth")
        exit(1)
    
    # ensemble method
    detector.config['ensemble']['method'] = 'simple_average'
    
    # prediction
    im_path = "model/real.webp"
    if os.path.exists(im_path):
        result = detector.predict_ensemble(im_path)
        print(result)
    else:
        print(f"Image {im_path} not found")