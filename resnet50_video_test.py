import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from torchvision.models import resnet50
import torch.nn as nn
import time

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

class VideoDetector:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = EnemyDetector().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def process_frame(self, frame):
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            bbox = self.model(image_tensor).cpu().numpy()[0]
        
        return bbox
    
    def draw_detection(self, frame, bbox):
        h, w = frame.shape[:2]
        x_center, y_center, box_w, box_h = bbox
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_center - box_w/2) * w)
        y1 = int((y_center - box_h/2) * h)
        x2 = int((x_center + box_w/2) * w)
        y2 = int((y_center + box_h/2) * h)
        
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(frame, 'Enemy', (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

def process_video(input_path, model_path, output_path=None, show_fps=True):
    """
    Process video file or webcam stream with enemy detection.
    
    Args:
        input_path: Path to video file or camera index (e.g., 0 for webcam)
        model_path: Path to trained model
        output_path: Path to save output video (optional)
        show_fps: Whether to display FPS counter
    """
    # Initialize detector
    detector = VideoDetector(model_path)
    
    # Initialize video capture
    if isinstance(input_path, int) or input_path.isdigit():
        cap = cv2.VideoCapture(int(input_path))
    else:
        cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize FPS counter
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            bbox = detector.process_frame(frame)
            
            # Draw detection
            frame = detector.draw_detection(frame, bbox)
            
            # Calculate and display FPS
            if show_fps:
                fps_counter += 1
                if time.time() - fps_start_time > 1:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                cv2.putText(frame, f'FPS: {current_fps}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to output video
            if output_path:
                out.write(frame)
            
            # Display frame
            cv2.imshow('Enemy Detection', frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "models/resnet50.pth"
    
    # For video file:
    VIDEO_PATH = "videos/Counter-Strike_ Global Offensive - Direct3D 9 2024-11-01 18-09-24.mp4"
    OUTPUT_PATH = "output_video.mp4"
    process_video(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH)
    