from pathlib import Path

MODEL_DIR = Path('models')
PLOTS_DIR = Path('plots')
RANDOM_STATE = 42

# Model
IMG_SIZE = 28
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Training
EPOCHS = 20
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1

# Inference
CANVAS_SIZE = 280  # frontend canvas size in pixels — scaled down to 28x28 for model
CONFIDENCE_THRESHOLD = 0.5
# Predictions below this confidence are flagged as uncertain.
# CNN is very confident on clean digits — low confidence usually means
# the drawing is ambiguous or the stroke is too thin.

# Azure
AZURE_APP_NAME = 'digit-recognizer-xoc'
AZURE_RESOURCE_GROUP = 'digit-recognizer-rg'
AZURE_LOCATION = 'westeurope'
