# google/vit-base-patch16-224 fine tune of the last layer and check accuracy

#To fine-tune the last layer of the Hugging Face model google/vit-base-patch16-224 (Vision Transformer) and check accuracy, you can follow this step-by-step example using PyTorch and transformers.
# ✅ Requirements

# Install required packages:

# pip install transformers datasets torchvision evaluate

from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import evaluate
import torch

# Load the dataset (use a small subset or your own)
dataset = load_dataset("beans")  # This is a small image classification dataset (3 classes)

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Define transforms
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    normalize,
])

# Apply transforms
def transform_examples(example):
    example['pixel_values'] = transform(example['image'])
    return example

dataset = dataset.with_transform(lambda x: {'pixel_values': transform(x['image']), 'label': x['labels']})

# Load model with adjusted head for the correct number of classes
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=3,  # Change this to match your dataset
    ignore_mismatched_sizes=True  # Important to replace classification head
)

# Freeze all layers except classification head
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# Accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-finetuned-beans",
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

# Train the model
trainer.train()

# Evaluate on test set
metrics = trainer.evaluate(dataset['test'])
print("Test Accuracy:", metrics["eval_accuracy"])

# To fine-tune the entire model (google/vit-base-patch16-224) — not just the last layer — you simply need to remove the layer-freezing step so that all parameters are updated during training.
# Here's the complete working script for full fine-tuning, using the Hugging Face transformers library and a sample dataset (beans). You can replace the dataset with your own if needed.


from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import evaluate
import torch

# Load a small sample dataset (3-class classification)
dataset = load_dataset("beans")  # Use your own dataset if needed

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Define image transformations
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    normalize,
])

# Transform the dataset
def transform_examples(example):
    example['pixel_values'] = transform(example['image'])
    return example

dataset = dataset.with_transform(lambda x: {'pixel_values': transform(x['image']), 'label': x['labels']})

# Load model and replace the classification head
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=3,  # Update based on your dataset
    ignore_mismatched_sizes=True
)

# ⚠️ NOTE: No layers are frozen — full fine-tuning

# Define metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-finetuned-full",
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

# Train the entire model
trainer.train()

# Evaluate on test set
metrics = trainer.evaluate(dataset['test'])
print("Test Accuracy:", metrics["eval_accuracy"])


# ✅ Key Differences From Last-Layer Fine-Tuning:
# No param.requires_grad = False — all layers are trainable.

# Higher training time, but better final accuracy if you have enough data.

# More prone to overfitting on small datasets.

"""
Fine-tuning some layers of google/vit-base-patch16-224 (instead of just the last layer or all layers) is a great strategy when:

Your dataset is medium-sized.

You want better generalization than freezing all but the final layer.

You want to reduce training time and memory compared to full fine-tuning.

✅ Strategy Overview
We'll do the following:

Freeze early layers (lower transformer blocks).

Unfreeze later layers (higher transformer blocks and classification head).

Adjust training hyperparameters to improve performance:

Learning rate

Weight decay

Epochs

Scheduler


"""
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import evaluate
import torch

# Load dataset
dataset = load_dataset("beans")

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Transform images
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    normalize,
])

# Apply transformation
dataset = dataset.with_transform(lambda x: {'pixel_values': transform(x['image']), 'label': x['labels']})

# Load the model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=3,
    ignore_mismatched_sizes=True
)

# ✅ Freeze the first N transformer blocks (e.g., up to block 8)
for name, param in model.named_parameters():
    if any(layer in name for layer in [f"encoder.layer.{i}" for i in range(0, 8)]):
        param.requires_grad = False

# Keep layer 8, 9, 10, 11, and classifier head trainable

# Define metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# Training arguments with tuned hyperparameters
training_args = TrainingArguments(
    output_dir="./vit-partial-finetune",
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=7,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.05,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    lr_scheduler_type="cosine",
    report_to="none"
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor
)

# Train the model
trainer.train()

# Evaluate on test set
metrics = trainer.evaluate(dataset["test"])
print("Test Accuracy:", metrics["eval_accuracy"])

"""
✅ Summary of What This Code Does:
Freezes layers 0–7 in the Vision Transformer.

Trains layers 8–11 and the classifier head.

Uses cosine learning rate scheduler with warmup steps.

Uses weight decay and 7 training epochs for better generalization.
"""



