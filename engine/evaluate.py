import torch
def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item()*batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy    