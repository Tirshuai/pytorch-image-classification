from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="outputs/runs/demo")

for epoch in range(10):
    train_loss = 1.0 / (epoch + 1)
    train_acc = 50 + epoch * 4

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)

writer.close()

print("TensorBoard logs saved.")