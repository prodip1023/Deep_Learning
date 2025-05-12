# main.py - auto-generated
from data_loader import load_datasets
from train import build_and_train_model
from evaluate import evaluate_model
from config import *

train_dir = "dataset/train"
val_dir = "dataset/val"

train_ds, val_ds = load_datasets(train_dir, val_dir)

model = build_and_train_model(train_ds, val_ds)

evaluate_model(model, val_ds)
model.save("saved_model/image_classifier")