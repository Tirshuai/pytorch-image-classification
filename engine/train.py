import torch
def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predicted = outputs.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item()*batch_size


    avg_loss = total_loss/total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy

