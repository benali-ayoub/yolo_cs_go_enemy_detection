import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn

class VOCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.annotations = []
        
        # Get all XML files
        ann_dir = os.path.join(data_dir, 'data/Annotations')
        img_dir = os.path.join(data_dir, 'data/JPEGImages')
        
        for xml_file in os.listdir(ann_dir):
            if xml_file.endswith('.xml'):
                ann_path = os.path.join(ann_dir, xml_file)
                img_path = os.path.join(img_dir, xml_file.replace('.xml', '.jpg'))
                
                if os.path.exists(img_path):
                    self.annotations.append(ann_path)
                    self.images.append(img_path)
    
    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size for normalization
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        # Get first enemy object (assuming one enemy per image)
        obj = root.find('object')
        if obj is None:
            return None
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text) / width
        ymin = float(bbox.find('ymin').text) / height
        xmax = float(bbox.find('xmax').text) / width
        ymax = float(bbox.find('ymax').text) / height
        
        # Convert to center format (x_center, y_center, width, height)
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        box_width = xmax - xmin
        box_height = ymax - ymin
        
        return [x_center, y_center, box_width, box_height]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        ann_path = self.annotations[idx]
        
        image = Image.open(img_path).convert('RGB')
        bbox = self.parse_voc_xml(ann_path)
        
        if bbox is None:
            bbox = [0, 0, 0, 0]  # Default values if no object found
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(bbox, dtype=torch.float32)

def create_data_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create train/val splits
    dataset = VOCDataset(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

class EnemyDetector(nn.Module):
    def __init__(self, num_classes=4):
        super(EnemyDetector, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Bound outputs between 0 and 1 for normalized coordinates
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(data_dir, batch_size)
    
    # Initialize model and training components
    model = EnemyDetector().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.1
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, boxes in train_loader:
            images = images.to(device)
            boxes = boxes.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, boxes)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, boxes in val_loader:
                images = images.to(device)
                boxes = boxes.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, boxes)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def run_detection(model_path):
    import mss
    sct = mss.mss()
    monitor = sct.monitors[1]  # Primary monitor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnemyDetector().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    while True:
        # Capture screen
        screenshot = np.array(sct.grab(monitor))
        image = Image.fromarray(screenshot)
        
        # Prepare image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get detection
        with torch.no_grad():
            bbox = model(input_tensor).cpu().numpy()[0]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = screenshot.shape[:2]
        x_center, y_center, box_w, box_h = bbox
        
        x1 = int((x_center - box_w/2) * w)
        y1 = int((y_center - box_h/2) * h)
        x2 = int((x_center + box_w/2) * w)
        y2 = int((y_center + box_h/2) * h)
        
        # Draw bbox
        cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display result
        cv2.imshow('Detection', screenshot)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()