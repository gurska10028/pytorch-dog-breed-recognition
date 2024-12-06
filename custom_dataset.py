import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

#Klasa CustomDataset została stworzona w celu efektywnego załadowania i przetworzenia zbioru danych, który zawiera obrazy i odpowiadające im etykiety.
#Jest to kluczowy element projektu, ponieważ standardowe klasy Dataset w PyTorch nie spełniają wymagań specyficznych dla tego projektu,
#takich jak obsługa plików JSON zawierających etykiety czy przetwarzanie obrazów w określony sposób. 
#Dzięki CustomDataset możliwe jest dostosowanie procesu wczytywania danych, co ma istotne znaczenie dla efektywnego szkolenia modeli w PyTorch 
#oraz późniejszej analizy i klasyfikacji obrazów.

class CustomDataset(Dataset):
    

    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = [file for file in os.listdir(images_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # Zwraca pojedynczy obraz i jego etykietę na podstawie indeksu 'i'
        
        img_name = self.images[i]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Konwersja obrazu do formatu RGB w celu zachowania spójności

        label_name = img_name.replace('.jpg', '.json')
        label_path = os.path.join(self.labels_dir, label_name)

        if os.path.exists(label_path):
             with open(label_path, 'r') as file:
                label_data = json.load(file)
                label = label_data["obecnosc_znaku"]
        else:
            label = 0  # W przypadku braku pliku etykiety

        # Przekształcenia obrazu
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([label], dtype=torch.float32)
        
transform = transforms.Compose([    
    transforms.Resize(256),   #obraz 256x256
    transforms.CenterCrop(224), #Przycinanie obrazu do kwadratu 224x224        
    transforms.ToTensor(), #konwersja obrazu do formatu tensora             
    transforms.Normalize(              
    mean=[0.485, 0.456, 0.406],        
    std=[0.229, 0.224, 0.225] # normalizacja obrazu na podstawie średniej i odchylenia standardowego          
 )])

dataset = CustomDataset(images_dir='data/obrazy', labels_dir='data/etykiety', transform=transform)

