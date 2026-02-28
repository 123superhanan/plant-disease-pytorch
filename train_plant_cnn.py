import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def predict_image(model_path, img_path, class_names):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return None

    # Load model
    model = SmallCNN(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        top3_prob, top3_idx = torch.topk(probabilities, 3)

    results = []
    for i in range(3):
        pred_class = class_names[top3_idx[i].item()]
        parts = pred_class.split('___')
        plant = parts[0].replace('_', ' ')
        if len(parts) > 1:
            disease = parts[1].replace('_', ' ')
            status = 'Healthy' if 'healthy' in disease.lower() else 'Diseased'
        else:
            plant = pred_class.replace('_', ' ')
            disease = 'None'
            status = 'Healthy'

        results.append({
            'plant_name': plant,
            'status': status,
            'disease': disease,
            'confidence': top3_prob[i].item() * 100,
            'full_class': pred_class
        })

    return results

def main():
    device = torch.device("cpu")  # force CPU
    print("Using device:", device)

    data_root = r"C:\Users\Admin\Desktop\pytorch\torch_env\PlantVillage"

    if not os.path.exists(data_root):
        print(f"Dataset folder not found: {data_root}")
        return

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_root, transform=transform)

    print("Total Images:", len(dataset))
    print("Classes:", dataset.classes)
    print("Number of classes:", len(dataset.classes))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    model = SmallCNN(len(dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot
    plt.plot(range(1, epochs+1), losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Save model
    torch.save(model.state_dict(), "plant_cnn.pth")
    print("Model saved as plant_cnn.pth")

    # Example prediction
    test_img = r"C:\Users\Admin\Desktop\pytorch\torch_env\PlantVillage\Pepper__bell___Bacterial_spot\0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"  
    if os.path.exists(test_img):
        result = predict_image("plant_cnn.pth", test_img, dataset.classes)
        if result:
            print("\nPrediction Results (Top 3):")
            for i, r in enumerate(result, 1):
                print(f"{i}. {r['plant_name']} - {r['status']} ({r['disease']})")
                print(f"   Confidence: {r['confidence']:.2f}%")
                print(f"   Full class: {r['full_class']}\n")
    else:
        print("Add a test image path to see prediction.")

if __name__ == "__main__":
    main()