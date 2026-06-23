
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

import evaluate

# =====================================================
# LOAD DATASET
# =====================================================

CSV_PATH = r"emotion_detection_dataset.csv"

df = pd.read_csv(CSV_PATH)

print("Dataset Shape:", df.shape)
print(df.head())
print("\nColumns:", df.columns.tolist())

# =====================================================
# FIND LABEL COLUMN
# =====================================================

LABEL_COL_CANDIDATES = [
    "label",
    "emotion",
    "Emotion",
    "Label",
    "sentiment",
    "category"
]

label_col = next(
    (c for c in LABEL_COL_CANDIDATES if c in df.columns),
    None
)

if label_col is None:
    raise ValueError(
        f"No label column found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

print(f"\nUsing label column: {label_col}")

# =====================================================
# ENCODE LABELS
# =====================================================

label_encoder = LabelEncoder()

df["label_id"] = label_encoder.fit_transform(df[label_col])

print("df[label_col].value_counts()",df[label_col].value_counts())

id2label = {
    i: label
    for i, label in enumerate(label_encoder.classes_)
}

label2id = {
    label: i
    for i, label in enumerate(label_encoder.classes_)
}

print("\nLabel Mapping:")
for k, v in id2label.items():
    print(f"{k} -> {v}")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_id"]
)

print("\nTrain Size:", len(train_df))
print("Test Size :", len(test_df))

# =====================================================
# CONVERT TO HF DATASET
# =====================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label_id"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label_id"]]
)

print("test_df[label_col].value_counts()",test_df[label_col].value_counts())

# =====================================================
# TOKENIZER
# =====================================================

MODEL_NAME = "distilbert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =====================================================
# TOKENIZATION
# =====================================================

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

# Rename for Trainer

train_dataset = train_dataset.rename_column(
    "label_id",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "label_id",
    "labels"
)

# Torch format

train_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

test_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

# Dynamic padding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# =====================================================
# MODEL
# =====================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_),
    id2label=id2label,
    label2id=label2id
)

print("id2label:",id2label)
# =====================================================
# METRICS
# =====================================================

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

# =====================================================
# TRAINING ARGUMENTS
# =====================================================

training_args = TrainingArguments(
    output_dir="./emotion5_model",

    num_train_epochs=3,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    eval_strategy="epoch",
    save_strategy="epoch",

    save_total_limit=1,

    logging_steps=10,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    report_to="none"
)

# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =====================================================
# TRAIN
# =====================================================

print("\nStarting training...\n")

trainer.train()

# =====================================================
# SAVE MODEL
# =====================================================

SAVE_DIR = "./emotion5_model"

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\nModel saved to: {SAVE_DIR}")

# =====================================================
# EVALUATION
# =====================================================

print("\nRunning evaluation...")

pred_output = trainer.predict(test_dataset)

true_ids = pred_output.label_ids
predicted_ids = np.argmax(pred_output.predictions, axis=-1)

from sklearn.metrics import classification_report

labels = list(range(len(label_encoder.classes_)))
print("label_encoder.classes_: ",label_encoder.classes_)

print(classification_report(
    true_ids,
    predicted_ids,
    labels=labels,
    target_names=label_encoder.classes_,
    zero_division=0
))




from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

accuracy = accuracy_score(true_ids, predicted_ids)
precision = precision_score(
    true_ids,
    predicted_ids,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    true_ids,
    predicted_ids,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    true_ids,
    predicted_ids,
    average="weighted",
    zero_division=0
)
print("np.unique(true_ids)",np.unique(true_ids))
print("\n===== OVERALL RESULTS =====")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
# =====================================================
# INFERENCE FUNCTION
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)

def predict_emotion(texts):

    model.eval()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    probs = torch.softmax(
        outputs.logits,
        dim=-1
    ).cpu().numpy()

    pred_ids = np.argmax(
        probs,
        axis=-1
    )

    results = []

    for i, text in enumerate(texts):

        results.append({
            "text": text,
            "emotion": id2label[pred_ids[i]],
            "confidence": round(
                float(
                    probs[i][pred_ids[i]]
                ) * 100,
                2
            )
        })

    return results

# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

sample_sentences = [
    "I just got promoted at work!",
    "I can't stop crying today.",
    "How dare they do this to me?",
    "There's someone outside my house.",
    "That food was disgusting.",
    "Wait, you're throwing me a party?"
]

print("\nSample Predictions:\n")

results = predict_emotion(
    sample_sentences
)

for r in results:
    print(
        f"[{r['emotion']:^12}] "
        f"{r['confidence']:6.2f}%  "
        f"{r['text']}"
    )
