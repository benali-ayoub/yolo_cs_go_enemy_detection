import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from torchvision.models import resnet50
import torch.nn as nn
import matplotlib.pyplot as plt

class EnemyDetector(nn.Module):
    def __init__(self, num_classes=4):
        super(EnemyDetector, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)

def test_single_image(image_path, model_path):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare model
    model = EnemyDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        bbox = model(image_tensor).cpu().numpy()[0]
    
    # Convert normalized coordinates to pixel coordinates
    img_w, img_h = original_image.size
    x_center, y_center, box_w, box_h = bbox
    
    x1 = int((x_center - box_w/2) * img_w)
    y1 = int((y_center - box_h/2) * img_h)
    x2 = int((x_center + box_w/2) * img_w)
    y2 = int((y_center + box_h/2) * img_h)
    
    # Convert PIL image to OpenCV format for drawing
    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Draw bounding box
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add confidence text (assuming full confidence)
    cv2.putText(image_cv, 'Enemy', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Print detection coordinates
    print(f"\nDetection Results:")
    print(f"Normalized coordinates (x_center, y_center, width, height): {bbox}")
    print(f"Pixel coordinates (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")
    
    # Display image
    cv2.imshow('Detection Result', image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    output_path = 'detection_result.jpg'
    cv2.imwrite(output_path, image_cv)
    print(f"\nResult saved as: {output_path}")

if __name__ == "__main__":
    # Replace these paths with your actual paths
    IMAGE_PATH = "1.jpg"
    MODEL_PATH = "models/resnet50.pth"
    
    test_single_image(IMAGE_PATH, MODEL_PATH)