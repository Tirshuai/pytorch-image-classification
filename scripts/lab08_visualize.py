# import os
# import matplotlib.pyplot as plt

# os.makedirs("outputs", exist_ok=True)

# # Loss 曲线
# plt.figure()
# plt.plot(history["train_loss"], label="Train Loss")
# plt.plot(history["test_loss"], label="Test Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss Curve")
# plt.legend()
# plt.grid()
# plt.savefig("outputs/loss_curve.png")
# plt.close()

# # Accuracy 曲线
# plt.figure()
# plt.plot(history["train_acc"], label="Train Accuracy")
# plt.plot(history["test_acc"], label="Test Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy Curve")
# plt.legend()
# plt.grid()
# plt.savefig("outputs/accuracy_curve.png")
# plt.close()