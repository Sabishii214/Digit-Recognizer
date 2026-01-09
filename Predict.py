import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import gradio as gr
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Load Model on CPU
device = torch.device('cpu')
model = SimpleCNN().to(device)
MODEL_PATH = "mnist_cnn.pth" 

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")

def predict(image_data):
    if image_data is None: return None, None
    
    # Gradio Sketchpad returns a dict with 'composite'
    img = image_data['composite'] 
    
    # Convert to PIL Image
    img = Image.fromarray(img.astype('uint8'))
    
    # STEP 1: Convert to Grayscale
    img = img.convert('L')
    
    # STEP 2: Handle Inversion 
    # MNIST is White Digit (255) on Black Background (0).
    # If the drawing is mostly white background, invert it.
    stat = np.array(img)
    if np.mean(stat) > 127:
        img = ImageOps.invert(img)
    
    # STEP 3: Binarize (make it high contrast)
    # This helps the model see the shape clearly
    img = img.point(lambda p: p > 50 and 255) 
    
    # STEP 4: Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Preprocessing for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
    
    results = {str(i): float(probs[i]) for i in range(10)}
    
    return results, img

# 3. Enhanced UI
interface = gr.Interface(
    fn=predict,
    # We add a larger brush to help the model see the lines
    inputs=gr.Sketchpad(label="Draw Digit (0-9)", type="numpy"),
    # We add an output image so you can see what the model "sees"
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions"),
        gr.Image(label="What the model sees (28x28)", image_mode="L")
    ],
    title="Digit Recognizer"
)

if __name__ == "__main__":
    interface.launch()