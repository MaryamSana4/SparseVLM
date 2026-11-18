
import os
import io
import csv
import json
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    auc
)

# ==========================================
# 1. SETTINGS
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 2e-5

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading OCT dataset...")

ds = load_dataset("QIAIUNCC/OCT-summary-Dataset")

print(ds)
print("Columns:", ds["train"].column_names)

train_raw = ds["train"]

if "test" in ds:
    test_raw = ds["test"]
elif "validation" in ds:
    test_raw = ds["validation"]
else:
    raise ValueError("No test or validation split found.")


# ==========================================
# 3. TOKENIZER AND IMAGE TRANSFORM
# ==========================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. LABEL ENCODING
# ==========================================

encoder = LabelEncoder()
encoder.fit(train_raw["label"])

class_names = encoder.classes_
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Number of classes:", num_classes)


# ==========================================
# 5. DATASET CLASS
# ==========================================

class OCTDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        # Load image
        image = item["image"]

        if isinstance(image, dict):

            if image.get("bytes") is not None:
                image = Image.open(
                    io.BytesIO(image["bytes"])
                ).convert("RGB")

            elif image.get("path") is not None:
                image = Image.open(
                    image["path"]
                ).convert("RGB")

            else:
                raise ValueError("Image path and bytes are missing.")

        else:
            image = image.convert("RGB")

        image = transform(image)

        # Load description
        description = str(item["desc"] or "")

        text = tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Encode label
        label = encoder.transform([item["label"]])[0]

        return (
            image,
            text["input_ids"].squeeze(0),
            text["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long)
        )


# ==========================================
# 6. MULTIMODAL MODEL
# ==========================================

class MultimodalModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # Image Encoder: ResNet50
        self.image_model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        self.image_model.fc = nn.Identity()

        # Text Encoder: BERT
        self.text_model = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        # Feature Fusion + MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):

        # Image features: 2048 dimensions
        image_features = self.image_model(image)

        # Text features: 768 dimensions
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_features = text_output.pooler_output

        # Feature fusion
        combined_features = torch.cat(
            (image_features, text_features),
            dim=1
        )

        # Classification
        output = self.classifier(combined_features)

        return output


# ==========================================
# 7. DATA LOADERS
# ==========================================

train_data = OCTDataset(train_raw)
test_data = OCTDataset(test_raw)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("\nTraining images:", len(train_data))
print("Testing images:", len(test_data))


# ==========================================
# 8. INITIALIZE MODEL
# ==========================================

model = MultimodalModel(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 9. COMPUTATIONAL METRICS
# ==========================================

# Total parameters
total_params = sum(
    p.numel() for p in model.parameters()
)

# Trainable parameters
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

# Non-trainable parameters
non_trainable_params = total_params - trainable_params

print("\n========== MODEL COMPLEXITY ==========")

print("Total Parameters:", f"{total_params:,}")
print("Trainable Parameters:", f"{trainable_params:,}")
print("Non-trainable Parameters:", f"{non_trainable_params:,}")

# Estimate MACs and FLOPs using THOP
macs = None
flops = None

try:

    from thop import profile

    dummy_image = torch.randn(
        1, 3, 224, 224
    ).to(device)

    dummy_ids = torch.ones(
        1, 128, dtype=torch.long
    ).to(device)

    dummy_mask = torch.ones(
        1, 128, dtype=torch.long
    ).to(device)

    model.eval()

    macs, thop_params = profile(
        model,
        inputs=(
            dummy_image,
            dummy_ids,
            dummy_mask
        ),
        verbose=False
    )

    # Conventional approximation: 1 MAC = 2 FLOPs
    flops = 2 * macs

    print("Estimated MACs per image:", f"{macs / 1e9:.4f} G")
    print("Estimated FLOPs per image:", f"{flops / 1e9:.4f} G")

except Exception as error:

    print("FLOPs estimation unavailable:", error)


    # del dummy_image if False else None


# ==========================================
# 10. TRAINING WITH PROGRESS BAR
# ==========================================

train_losses = []
epoch_times = []

print("\n========== STARTING TRAINING ==========")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)

training_start = time.perf_counter()

for epoch in range(EPOCHS):

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    epoch_start = time.perf_counter()

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch + 1}/{EPOCHS}]",
        unit="batch"
    )

    for batch_idx, (image, input_ids, attention_mask, labels) in enumerate(progress_bar):

        image = image.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            image,
            input_ids,
            attention_mask
        )

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / (batch_idx + 1):.4f}"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    epoch_time = time.perf_counter() - epoch_start

    avg_loss = total_loss / len(train_loader)

    train_losses.append(avg_loss)
    epoch_times.append(epoch_time)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Time: {epoch_time:.2f} seconds"
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

total_training_time = time.perf_counter() - training_start

average_epoch_time = np.mean(epoch_times)

print("\nTotal Training Time:", total_training_time, "seconds")
print("Average Epoch Time:", average_epoch_time, "seconds")


# ==========================================
# 11. SAVE TRAINING LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o"
)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig(
    RESULTS_DIR / "training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 12. TESTING AND INFERENCE TIME
# ==========================================

print("\n========== STARTING TESTING ==========")

model.eval()

predictions = []
actual_labels = []
all_probabilities = []

if torch.cuda.is_available():
    torch.cuda.synchronize()

inference_start = time.perf_counter()

with torch.no_grad():

    progress_bar = tqdm(
        test_loader,
        desc="Testing",
        unit="batch"
    )

    for image, input_ids, attention_mask, labels in progress_bar:

        image = image.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(
            image,
            input_ids,
            attention_mask
        )

        probabilities = torch.softmax(outputs, dim=1)

        predicted = torch.argmax(outputs, dim=1)

        predictions.extend(predicted.cpu().numpy())
        actual_labels.extend(labels.numpy())

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

if torch.cuda.is_available():
    torch.cuda.synchronize()

total_inference_time = time.perf_counter() - inference_start

predictions = np.array(predictions)
actual_labels = np.array(actual_labels)
all_probabilities = np.array(all_probabilities)

number_of_test_images = len(actual_labels)

average_inference_time = (
    total_inference_time / number_of_test_images
)

inference_latency_ms = average_inference_time * 1000

throughput = (
    number_of_test_images / total_inference_time
)

print("\nTotal Inference Time:", total_inference_time, "seconds")
print("Average Inference Time:", inference_latency_ms, "ms/image")
print("Inference Throughput:", throughput, "images/second")


# ==========================================
# 13. CLASSIFICATION METRICS
# ==========================================

accuracy = accuracy_score(
    actual_labels,
    predictions
)

balanced_acc = balanced_accuracy_score(
    actual_labels,
    predictions
)

precision_macro = precision_score(
    actual_labels,
    predictions,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    actual_labels,
    predictions,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    actual_labels,
    predictions,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    actual_labels,
    predictions,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    actual_labels,
    predictions,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    actual_labels,
    predictions,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    actual_labels,
    predictions
)

kappa = cohen_kappa_score(
    actual_labels,
    predictions
)


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

labels_range = np.arange(num_classes)

cm = confusion_matrix(
    actual_labels,
    predictions,
    labels=labels_range
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(9, 7))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(
    labels_range,
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    labels_range,
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(num_classes):
    for j in range(num_classes):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 15. SENSITIVITY AND SPECIFICITY
# ==========================================

class_metrics = []

for i, class_name in enumerate(class_names):

    tp = cm[i, i]

    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp

    tn = cm.sum() - (tp + fn + fp)

    sensitivity = (
        tp / (tp + fn) if (tp + fn) > 0 else 0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0
    )

    class_metrics.append({
        "Class": str(class_name),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity)
    })

    print(f"\n{class_name}")
    print("Sensitivity:", round(sensitivity, 4))
    print("Specificity:", round(specificity, 4))

macro_specificity = np.mean([
    item["Specificity"] for item in class_metrics
])


# ==========================================
# 16. ROC-AUC
# ==========================================

roc_auc = None

y_true_binary = label_binarize(
    actual_labels,
    classes=labels_range
)

try:

    roc_auc = roc_auc_score(
        y_true_binary,
        all_probabilities,
        average="macro"
    )

    print("\nMacro ROC-AUC:", round(roc_auc, 4))

    plt.figure(figsize=(9, 7))

    for i, class_name in enumerate(class_names):

        if len(np.unique(y_true_binary[:, i])) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(
            y_true_binary[:, i],
            all_probabilities[:, i]
        )

        class_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC = {class_auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.title("Multiclass ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(
        RESULTS_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

except ValueError as error:

    print("ROC-AUC unavailable:", error)


# ==========================================
# 17. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    actual_labels,
    predictions,
    labels=labels_range,
    target_names=class_names.astype(str),
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ==========================================
# 18. GPU MEMORY METRICS
# ==========================================

if torch.cuda.is_available():

    peak_gpu_memory = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 3)
    )

    peak_reserved_memory = (
        torch.cuda.max_memory_reserved(device)
        / (1024 ** 3)
    )

    print("\nPeak GPU Memory Allocated:", peak_gpu_memory, "GB")
    print("Peak GPU Memory Reserved:", peak_reserved_memory, "GB")

else:

    peak_gpu_memory = None
    peak_reserved_memory = None


# ==========================================
# 19. SAVE MODEL
# ==========================================

model_path = RESULTS_DIR / "multimodal_baseline.pth"

torch.save(
    model.state_dict(),
    model_path
)

model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

print("\nModel saved:", model_path)
print("Model Size:", model_size_mb, "MB")


# ==========================================
# 20. FINAL METRICS
# ==========================================

metrics = {

    # Classification metrics
    "Accuracy": float(accuracy),
    "Balanced Accuracy": float(balanced_acc),

    "Macro Precision": float(precision_macro),
    "Macro Recall": float(recall_macro),
    "Macro F1": float(f1_macro),

    "Weighted Precision": float(precision_weighted),
    "Weighted Recall": float(recall_weighted),
    "Weighted F1": float(f1_weighted),

    "Macro Specificity": float(macro_specificity),

    "MCC": float(mcc),
    "Cohen Kappa": float(kappa),

    "Macro ROC-AUC": (
        float(roc_auc) if roc_auc is not None else None
    ),

    # Model complexity
    "Total Parameters": int(total_params),
    "Trainable Parameters": int(trainable_params),
    "Non-trainable Parameters": int(non_trainable_params),

    "Estimated MACs per Image": (
        float(macs) if macs is not None else None
    ),

    "Estimated FLOPs per Image": (
        float(flops) if flops is not None else None
    ),

    # Training computation
    "Total Training Time (seconds)": float(total_training_time),
    "Average Epoch Time (seconds)": float(average_epoch_time),

    # Inference computation
    "Total Inference Time (seconds)": float(total_inference_time),
    "Average Inference Time (ms/image)": float(inference_latency_ms),
    "Inference Throughput (images/second)": float(throughput),

    # GPU
    "Peak GPU Memory Allocated (GB)": (
        float(peak_gpu_memory) if peak_gpu_memory is not None else None
    ),

    "Peak GPU Memory Reserved (GB)": (
        float(peak_reserved_memory) if peak_reserved_memory is not None else None
    ),

    # Saved model
    "Model Size (MB)": float(model_size_mb)
}


# ==========================================
# 21. SAVE METRICS JSON
# ==========================================

with open(
    RESULTS_DIR / "evaluation_metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(metrics, file, indent=4)


# ==========================================
# 22. SAVE CLASS-WISE METRICS
# ==========================================

with open(
    RESULTS_DIR / "class_wise_metrics.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Class",
            "Sensitivity",
            "Specificity"
        ]
    )

    writer.writeheader()
    writer.writerows(class_metrics)


# ==========================================
# 23. SAVE PREDICTIONS
# ==========================================

with open(
    RESULTS_DIR / "predictions.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Actual Label",
        "Predicted Label",
        "Confidence"
    ])

    for i in range(len(predictions)):

        writer.writerow([
            class_names[actual_labels[i]],
            class_names[predictions[i]],
            float(np.max(all_probabilities[i]))
        ])


# ==========================================
# 24. SAVE EPOCH TIMES
# ==========================================

with open(
    RESULTS_DIR / "epoch_times.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Training Loss",
        "Epoch Time (seconds)"
    ])

    for i in range(EPOCHS):

        writer.writerow([
            i + 1,
            train_losses[i],
            epoch_times[i]
        ])


# ==========================================
# 25. SAVE CLASS NAMES
# ==========================================

with open(
    RESULTS_DIR / "class_names.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names.tolist(),
        file,
        indent=4
    )


# ==========================================
# 26. FINAL RESULTS
# ==========================================

print("\n==========================================")
print("MULTIMODAL OCT BASELINE COMPLETED")
print("==========================================")

for key, value in metrics.items():

    if value is None:
        print(f"{key}: Not available")

    elif isinstance(value, int):
        print(f"{key}: {value:,}")

    elif isinstance(value, float):
        print(f"{key}: {value:.4f}")

print("\nAll results saved in:")
print(RESULTS_DIR)
