import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gerekli kütüphaneleri yükle
model_type = "DPT_Large"  # Alternatifler: "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Görüntüyü yükle
img = cv2.imread("C:/Users/iremolcun/Desktop/depth_sensing/object.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Görüntüyü dönüştür
input_batch = transform(img).unsqueeze(0)

# input_batch şekli kontrolü
print(f"Input batch shape: {input_batch.shape}")

# input_batch boyutunu doğru şekle getirme
input_batch = input_batch.squeeze(0)
print(f"Corrected input batch shape: {input_batch.shape}")

# Modeli değerlendirme moduna geçir
midas.eval()

# GPU kullanımı kontrolü (varsa)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
input_batch = input_batch.to(device)

# Derinlik tahmini
with torch.no_grad():
    prediction = midas(input_batch)
    
    # prediction şekli kontrolü
    print(f"Prediction shape: {prediction.shape}")
    
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# Derinlik haritasını göster
plt.imshow(output, cmap='plasma')
plt.colorbar()
plt.show()