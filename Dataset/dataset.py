
from datasets import load_from_disk
import matplotlib.pyplot as plt

# Load dataset
dataset = load_from_disk(
    "D:/SparseVLM/Dataset/OCT-summary-Dataset"
)

# Select first training sample
sample = dataset["train"][0]

# Extract information
image = sample["image"]
label = sample["label"]
description = sample["desc"]

# Display image, label and description
plt.figure(figsize=(10, 7))

plt.imshow(image, cmap="gray")
plt.title(f"OCT Image\nLabel: {label}", fontsize=13)

plt.xlabel(f"Description: {description}", fontsize=10, labelpad=15)

plt.axis("off")
plt.tight_layout()
plt.show()

# Print details in terminal
print("Label:", label)
print("Description:", description)
