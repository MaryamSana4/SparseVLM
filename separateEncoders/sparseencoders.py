
import os
import io
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import models, transforms
from PIL import Image

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
from tqdm import tqdm


# ==========================================
# 1. CONFIGURATION
# ==========================================

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 2e-5

NUM_CLASSES = 4
MAX_LENGTH = 128

SPARSITY_WEIGHT = 0.001
GATE_TEMPERATURE = 0.1

RANDOM_SEED = 42

DATASET_NAME = "QIAIUNCC/OCT-summary-Dataset"
TEXT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\nLoading OCT dataset...")

ds = load_dataset(DATASET_NAME)

train_data = ds["train"]
test_data = ds["test"]

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))


# ==========================================
# 4. LABEL ENCODING
# ==========================================

label_encoder = LabelEncoder()

train_labels = [
    str(x) for x in train_data["label"]
]

test_labels = [
    str(x) for x in test_data["label"]
]

label_encoder.fit(train_labels)

if not set(test_labels).issubset(set(label_encoder.classes_)):
    raise ValueError("Test set contains labels not found in training set.")

train_encoded = label_encoder.transform(train_labels)
test_encoded = label_encoder.transform(test_labels)

NUM_CLASSES = len(label_encoder.classes_)

print("\nClasses:", list(label_encoder.classes_))


# ==========================================
# 5. IMAGE TRANSFORMS
# ==========================================

image_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# 6. DATASET CLASS
# ==========================================

class OCTDataset(Dataset):

    def __init__(self, dataset, encoded_labels):

        self.dataset = dataset
        self.labels = encoded_labels

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]

        # Handle Hugging Face image formats

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

                raise ValueError("Invalid image format.")

        else:

            image = image.convert("RGB")

        image = image_transform(image)

        description = item["desc"]

        if description is None:
            description = ""

        description = str(description)

        label = int(self.labels[index])

        return image, description, label


# ==========================================
# 7. TEXT TOKENIZER
# ==========================================

print("\nLoading PubMedBERT...")

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)


# ==========================================
# 8. COLLATE FUNCTION
# ==========================================

def collate_fn(batch):

    images = torch.stack([
        item[0] for item in batch
    ])

    descriptions = [
        item[1] for item in batch
    ]

    labels = torch.tensor(
        [item[2] for item in batch],
        dtype=torch.long
    )

    text_inputs = tokenizer(
        descriptions,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return (
        images,
        text_inputs["input_ids"],
        text_inputs["attention_mask"],
        labels
    )


# ==========================================
# 9. TRAIN / VALIDATION SPLIT
# ==========================================

all_indices = np.arange(len(train_data))

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=0.10,
    random_state=RANDOM_SEED,
    stratify=train_encoded
)

full_train_dataset = OCTDataset(
    train_data,
    train_encoded
)

test_dataset = OCTDataset(
    test_data,
    test_encoded
)

train_dataset = Subset(
    full_train_dataset,
    train_indices
)

val_dataset = Subset(
    full_train_dataset,
    val_indices
)


# ==========================================
# 10. DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available()
)


# ==========================================
# 11. SPARSE VLM MODEL
# ==========================================

class SparseVLM(nn.Module):

    def __init__(self, num_classes):

        super(SparseVLM, self).__init__()

        # ----------------------------------
        # IMAGE ENCODER: RESNET50
        # ----------------------------------

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # Output:
        # [B, 2048, 7, 7]

        # ----------------------------------
        # TEXT ENCODER: PUBMEDBERT
        # ----------------------------------

        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL
        )

        # Text dimension = 768
        # Visual dimension = 2048

        self.visual_projection = nn.Linear(
            2048, 768
        )

        # ----------------------------------
        # CROSS-MODAL ATTENTION
        # ----------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            batch_first=True
        )

        # ----------------------------------
        # SEMANTIC RELEVANCE ESTIMATOR
        # ----------------------------------

        self.relevance_network = nn.Sequential(

            nn.Linear(768 * 2, 256),

            nn.ReLU(),

            nn.Linear(256, 1),

            nn.Sigmoid()

        )

        # ----------------------------------
        # ADAPTIVE THRESHOLD NETWORK
        # ----------------------------------

        self.threshold_network = nn.Sequential(

            nn.Linear(3, 64),

            nn.ReLU(),

            nn.Linear(64, 1),

            nn.Sigmoid()

        )

        # ----------------------------------
        # VISUAL CLASSIFIER ONLY
        # ----------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(2048, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, num_classes)

        )

    def forward(
        self,
        images,
        input_ids,
        attention_mask,
        hard_mask=False
    ):

        # ==================================
        # A. EXTRACT VISUAL FEATURES
        # ==================================

        visual_map = self.image_encoder(images)

        batch_size = visual_map.size(0)

        # [B, 2048, 7, 7]
        # -> [B, 49, 2048]

        visual_tokens = visual_map.flatten(2)

        visual_tokens = visual_tokens.transpose(1, 2)

        # Project visual features to 768

        visual_projected = self.visual_projection(
            visual_tokens
        )

        # ==================================
        # B. EXTRACT SEMANTIC TEXT FEATURES
        # ==================================

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_tokens = text_output.last_hidden_state

        # [B, L, 768]

        # ==================================
        # C. SEMANTIC CROSS-ATTENTION
        # ==================================

        # Visual tokens are queries.
        # Text tokens are keys and values.

        semantic_context, _ = self.cross_attention(

            query=visual_projected,

            key=text_tokens,

            value=text_tokens,

            key_padding_mask=~attention_mask.bool(),

            need_weights=False

        )

        # [B, 49, 768]

        # ==================================
        # D. FEATURE RELEVANCE
        # ==================================

        combined_features = torch.cat(
            [
                visual_projected,
                semantic_context
            ],
            dim=-1
        )

        relevance_scores = self.relevance_network(
            combined_features
        )

        # [B, 49, 1]

        # ==================================
        # E. ADAPTIVE THRESHOLD
        # ==================================

        score_mean = relevance_scores.mean(
            dim=1
        )

        score_std = relevance_scores.std(
            dim=1,
            unbiased=False
        )

        score_max = relevance_scores.max(
            dim=1
        ).values

        score_statistics = torch.cat(
            [
                score_mean,
                score_std,
                score_max
            ],
            dim=1
        )

        adaptive_threshold = self.threshold_network(
            score_statistics
        )

        # Ensure at least one region can be retained.
        # This does NOT impose a fixed Top-K.

        maximum_score = relevance_scores.max(
            dim=1
        ).values

        adaptive_threshold = torch.minimum(
            adaptive_threshold,
            maximum_score - 1e-4
        ).clamp(min=0.0)

        # ==================================
        # F. ADAPTIVE FEATURE CLIPPING
        # ==================================

        soft_gate = torch.sigmoid(

            (
                relevance_scores -
                adaptive_threshold.unsqueeze(1)
            ) / GATE_TEMPERATURE

        )

        if hard_mask:

            # Binary mask during validation/testing.
            # 1 = retain, 0 = remove.

            visual_gate = (
                relevance_scores >=
                adaptive_threshold.unsqueeze(1)
            ).float()

        else:

            # Differentiable soft gating during training.

            visual_gate = soft_gate

        # ==================================
        # G. REMOVE IRRELEVANT VISUAL FEATURES
        # ==================================

        selected_visual_features = (
            visual_tokens * visual_gate
        )

        # [B, 49, 2048]

        # ==================================
        # H. VISUAL FEATURE AGGREGATION
        # ==================================

        # Max aggregation across spatial locations.
        # No text features are used here.

        pooled_visual_features = selected_visual_features.max(
            dim=1
        ).values

        # [B, 2048]

        # ==================================
        # I. FINAL CLASSIFICATION
        # ==================================

        predictions = self.classifier(
            pooled_visual_features
        )

        return (
            predictions,
            soft_gate,
            adaptive_threshold
        )


# ==========================================
# 12. INITIALIZE MODEL
# ==========================================

print("\nInitializing SparseVLM...")

model = SparseVLM(
    num_classes=NUM_CLASSES
).to(DEVICE)


# ==========================================
# 13. LOSS AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ==========================================
# 14. COMPUTATIONAL METRICS
# ==========================================

total_parameters = sum(
    p.numel() for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

non_trainable_parameters = (
    total_parameters - trainable_parameters
)

print("\nTotal parameters:", total_parameters)

print("Trainable parameters:", trainable_parameters)

print("Non-trainable parameters:", non_trainable_parameters)


# ==========================================
# 15. TRAINING
# ==========================================

train_losses = []
val_losses = []

epoch_times = []

best_val_loss = float("inf")

best_model_path = os.path.join(
    RESULTS_DIR,
    "best_sparsevlm.pth"
)

last_checkpoint_path = os.path.join(
    RESULTS_DIR,
    "last_checkpoint.pth"
)

if torch.cuda.is_available():

    torch.cuda.reset_peak_memory_stats()

if torch.cuda.is_available():
    torch.cuda.synchronize()

training_start = time.perf_counter()


for epoch in range(EPOCHS):

    epoch_start = time.perf_counter()

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        unit="batch"
    )

    for images, input_ids, attention_mask, labels in progress_bar:

        images = images.to(DEVICE)

        input_ids = input_ids.to(DEVICE)

        attention_mask = attention_mask.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        predictions, gates, thresholds = model(
            images,
            input_ids,
            attention_mask,
            hard_mask=False
        )

        classification_loss = criterion(
            predictions,
            labels
        )

        # Encourage suppression of irrelevant features.
        # This is a soft sparsity penalty.

        sparsity_loss = gates.mean()

        loss = (
            classification_loss +
            SPARSITY_WEIGHT * sparsity_loss
        )

        loss.backward()

        optimizer.step()

        running_loss += classification_loss.item()

        progress_bar.set_postfix(
            loss=f"{classification_loss.item():.4f}",
            gate=f"{gates.mean().item():.3f}"
        )

    train_loss = running_loss / len(train_loader)

    train_losses.append(train_loss)

    # ==================================
    # VALIDATION
    # ==================================

    model.eval()

    val_running_loss = 0.0

    with torch.no_grad():

        for images, input_ids, attention_mask, labels in val_loader:

            images = images.to(DEVICE)

            input_ids = input_ids.to(DEVICE)

            attention_mask = attention_mask.to(DEVICE)

            labels = labels.to(DEVICE)

            predictions, gates, thresholds = model(
                images,
                input_ids,
                attention_mask,
                hard_mask=True
            )

            val_loss = criterion(
                predictions,
                labels
            )

            val_running_loss += val_loss.item()

    average_val_loss = (
        val_running_loss / len(val_loader)
    )

    val_losses.append(average_val_loss)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    epoch_time = time.perf_counter() - epoch_start

    epoch_times.append(epoch_time)

    print("\n----------------------------------")

    print(f"Epoch: {epoch + 1}/{EPOCHS}")

    print(f"Training Loss: {train_loss:.4f}")

    print(f"Validation Loss: {average_val_loss:.4f}")

    print(f"Epoch Time: {epoch_time:.2f} seconds")

    print("----------------------------------")

    # ==================================
    # SAVE LATEST CHECKPOINT
    # ==================================

    checkpoint = {

        "epoch": epoch + 1,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "train_loss": train_loss,

        "val_loss": average_val_loss,

        "classes": list(label_encoder.classes_)

    }

    torch.save(
        checkpoint,
        last_checkpoint_path
    )

    # ==================================
    # SAVE BEST MODEL
    # ==================================

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("Best model saved!")


if torch.cuda.is_available():
    torch.cuda.synchronize()

total_training_time = (
    time.perf_counter() - training_start
)

print("\nTraining completed.")

print(
    "Total training time:",
    total_training_time,
    "seconds"
)


# ==========================================
# 16. PLOT TRAINING CURVES
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("SparseVLM Training and Validation Loss")

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(RESULTS_DIR, "loss_curve.png"),
    dpi=300
)

plt.close()


# ==========================================
# 17. LOAD BEST MODEL
# ==========================================

print("\nLoading best SparseVLM model...")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# ==========================================
# 18. TEST EVALUATION
# ==========================================

y_true = []

y_pred = []

y_prob = []

test_gates = []

inference_start = time.perf_counter()

if torch.cuda.is_available():
    torch.cuda.synchronize()

with torch.no_grad():

    for images, input_ids, attention_mask, labels in tqdm(
        test_loader,
        desc="Testing",
        unit="batch"
    ):

        images = images.to(DEVICE)

        input_ids = input_ids.to(DEVICE)

        attention_mask = attention_mask.to(DEVICE)

        predictions, gates, thresholds = model(
            images,
            input_ids,
            attention_mask,
            hard_mask=True
        )

        probabilities = torch.softmax(
            predictions,
            dim=1
        )

        predicted_labels = torch.argmax(
            probabilities,
            dim=1
        )

        y_true.extend(labels.numpy())

        y_pred.extend(
            predicted_labels.cpu().numpy()
        )

        y_prob.extend(
            probabilities.cpu().numpy()
        )

        test_gates.extend(
            gates.cpu().numpy()
        )

if torch.cuda.is_available():
    torch.cuda.synchronize()

total_inference_time = (
    time.perf_counter() - inference_start
)

y_true = np.array(y_true)

y_pred = np.array(y_pred)

y_prob = np.array(y_prob)

test_gates = np.array(test_gates)

number_of_test_images = len(y_true)

average_inference_ms = (
    total_inference_time /
    number_of_test_images
) * 1000

inference_throughput = (
    number_of_test_images /
    total_inference_time
)


# ==========================================
# 19. CLASSIFICATION METRICS
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_true,
    y_pred
)

kappa = cohen_kappa_score(
    y_true,
    y_pred
)


# ==========================================
# 20. SPECIFICITY
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

specificities = []

sensitivities = []

for i in range(NUM_CLASSES):

    TP = cm[i, i]

    FN = cm[i, :].sum() - TP

    FP = cm[:, i].sum() - TP

    TN = cm.sum() - (TP + FN + FP)

    sensitivity = TP / (TP + FN) if (TP + FN) else 0

    specificity = TN / (TN + FP) if (TN + FP) else 0

    sensitivities.append(sensitivity)

    specificities.append(specificity)

macro_specificity = np.mean(specificities)


# ==========================================
# 21. ROC-AUC
# ==========================================

try:

    y_true_binary = label_binarize(
        y_true,
        classes=np.arange(NUM_CLASSES)
    )

    macro_roc_auc = roc_auc_score(
        y_true_binary,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

except ValueError:

    macro_roc_auc = None

    print("ROC-AUC could not be calculated.")


# ==========================================
# 22. SAVE ROC CURVE
# ==========================================

if macro_roc_auc is not None:

    plt.figure(figsize=(8, 6))

    for i in range(NUM_CLASSES):

        fpr, tpr, _ = roc_curve(
            y_true_binary[:, i],
            y_prob[:, i]
        )

        class_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{label_encoder.classes_[i]} AUC={class_auc:.3f}"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("SparseVLM ROC Curve")

    plt.legend()

    plt.grid()

    plt.savefig(
        os.path.join(RESULTS_DIR, "roc_curve.png"),
        dpi=300
    )

    plt.close()


# ==========================================
# 23. CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(8, 6))

plt.imshow(cm, cmap="Blues")

plt.title("SparseVLM Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.xticks(
    range(NUM_CLASSES),
    label_encoder.classes_,
    rotation=30
)

plt.yticks(
    range(NUM_CLASSES),
    label_encoder.classes_
)

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()


# ==========================================
# 24. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\nClassification Report:")

print(report)

with open(
    os.path.join(RESULTS_DIR, "classification_report.txt"),
    "w"
) as file:

    file.write(report)


# ==========================================
# 25. COMPUTATIONAL METRICS
# ==========================================

model_parameters = sum(
    p.numel() for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

model_path = os.path.join(
    RESULTS_DIR,
    "best_sparsevlm.pth"
)

model_size_mb = (
    os.path.getsize(model_path) /
    (1024 * 1024)
)

if torch.cuda.is_available():

    peak_gpu_allocated = (
        torch.cuda.max_memory_allocated(DEVICE) /
        (1024 ** 3)
    )

    peak_gpu_reserved = (
        torch.cuda.max_memory_reserved(DEVICE) /
        (1024 ** 3)
    )

else:

    peak_gpu_allocated = 0

    peak_gpu_reserved = 0


# ==========================================
# 26. FEATURE SELECTION METRICS
# ==========================================

# Hard mask retention rate for each image

hard_retention = (
    test_gates >= 0.5
).mean(axis=1)

average_retention_rate = float(
    hard_retention.mean()
)

average_suppression_rate = (
    1.0 - average_retention_rate
)

# ==========================================
# 27. FINAL RESULTS
# ==========================================

results = {

    "Accuracy": float(accuracy),

    "Balanced Accuracy": float(balanced_accuracy),

    "Macro Precision": float(macro_precision),

    "Macro Recall": float(macro_recall),

    "Macro F1": float(macro_f1),

    "Weighted Precision": float(weighted_precision),

    "Weighted Recall": float(weighted_recall),

    "Weighted F1": float(weighted_f1),

    "Macro Specificity": float(macro_specificity),

    "MCC": float(mcc),

    "Cohen Kappa": float(kappa),

    "Macro ROC-AUC": (
        float(macro_roc_auc)
        if macro_roc_auc is not None
        else None
    ),

    "Total Parameters": int(model_parameters),

    "Trainable Parameters": int(trainable_parameters),

    "Non-trainable Parameters": int(
        model_parameters - trainable_parameters
    ),

    "Total Training Time (seconds)": float(
        total_training_time
    ),

    "Average Epoch Time (seconds)": float(
        np.mean(epoch_times)
    ),

    "Total Inference Time (seconds)": float(
        total_inference_time
    ),

    "Average Inference Time (ms/image)": float(
        average_inference_ms
    ),

    "Inference Throughput (images/second)": float(
        inference_throughput
    ),

    "Peak GPU Memory Allocated (GB)": float(
        peak_gpu_allocated
    ),

    "Peak GPU Memory Reserved (GB)": float(
        peak_gpu_reserved
    ),

    "Model Size (MB)": float(model_size_mb),

    "Average Visual Feature Retention Rate": float(
        average_retention_rate
    ),

    "Average Visual Feature Suppression Rate": float(
        average_suppression_rate
    )

}


# ==========================================
# 28. SAVE RESULTS
# ==========================================

with open(
    os.path.join(RESULTS_DIR, "metrics.json"),
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


pd.DataFrame(
    [results]
).to_csv(
    os.path.join(RESULTS_DIR, "metrics.csv"),
    index=False
)


pd.DataFrame({

    "Epoch": range(1, EPOCHS + 1),

    "Training Loss": train_losses,

    "Validation Loss": val_losses,

    "Epoch Time": epoch_times

}).to_csv(
    os.path.join(RESULTS_DIR, "training_history.csv"),
    index=False
)


pd.DataFrame({

    "True Label": [
        label_encoder.classes_[x]
        for x in y_true
    ],

    "Predicted Label": [
        label_encoder.classes_[x]
        for x in y_pred
    ]

}).to_csv(
    os.path.join(RESULTS_DIR, "predictions.csv"),
    index=False
)


# ==========================================
# 29. PRINT FINAL RESULTS
# ==========================================

print("\n===================================")

print("SPARSEVLM FINAL RESULTS")

print("===================================")

for key, value in results.items():

    print(f"{key}: {value}")

print("\nAll results saved in:")

print(RESULTS_DIR)

print("\nSparseVLM training and evaluation completed.")
