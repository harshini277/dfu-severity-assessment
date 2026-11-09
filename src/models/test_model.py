
import torch

from coral_resnet import (
    CoralResNet18,
    coral_loss,
    coral_predict
)

# Test on CPU first.
device = torch.device("cpu")

model = CoralResNet18(
    num_classes=4,
    pretrained=False
).to(device)

# Simulate a batch of 32 RGB images.
images = torch.randn(32, 3, 224, 224)

# Simulate labels from Grade 1 to Grade 4.
labels = torch.randint(0, 4, (32,))

logits = model(images)
loss = coral_loss(logits, labels)
predictions = coral_predict(logits)

print("Logits shape:", logits.shape)
print("Loss:", loss.item())
print("Predictions shape:", predictions.shape)
print("Predicted classes:", predictions.tolist())

assert logits.shape == (32, 3)
assert predictions.shape == (32,)
assert torch.isfinite(loss)
probabilities = torch.sigmoid(logits)

print("Learned threshold values:", model.thresholds.detach().tolist())

assert torch.all(model.thresholds[1:] > model.thresholds[:-1])
assert torch.all(
    probabilities[:, :-1] >= probabilities[:, 1:]
)

print("Threshold ordering: PASSED")
print("Cumulative probability ordering: PASSED")
print("\nMODEL TEST PASSED!")
