import torch
from models.experimental import attempt_load

weights = './yolov5s.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = attempt_load(weights, map_location=device) 

img_size = 640
img = torch.zeros((1, 3, img_size, img_size), device=device)

model.eval()

with torch.no_grad():
    output = model(img)
    print(f"Model output shape: {output.shape}")

print(f"Input size: {img_size}x{img_size}")
print(f"Model architecture: {model}")

