
# Emotion Detection Application

## Project Structure
Project Folder
│
├── Backend
│   ├── app.py
│   └── training_model.py
│
├── Frontend UI
│   └── Emotion_UI.html
│
└── Dataset
    └── emotion_detection_dataset.csv


## Setup Instructions

### Step 1: Open the Project

1. Open the project folder in VS Code.
2. Open the VS Code Terminal.
3. Select **Command Prompt** as the terminal.

---

### Step 2: Create and Activate Virtual Environment

python -m venv venv
venv\Scripts\activate.bat

---

### Step 3: Install Dependencies

```bash
pip install uv
pip install fastapi
pip install uvicorn
pip install pandas
pip install datasets
pip install torch
pip install evaluate
pip install transformers
pip install scikit-learn
pip install accelerate
pip install -U tokenizers transformers
pip install sentencepiece
```

---

## Running the Application

### Train the Model (Run Only Once)

```bash
python training_model.py
```

After successful training, a folder named:
"emotion_training_model"
will be created containing the trained model files.


### Start the Backend

Run:

```bash
uvicorn app:app --reload
```

Expected output:

```text
Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal window open.

---

### Start the Frontend

After the backend is running:

1. Open File Explorer.
2. Navigate to the project folder.
3. Locate: "Emotion_UI.html"
4. Right-click the file.
5. Select **Open With**.
6. Choose **Google Chrome** or **Microsoft Edge**.

---

## Supported Emotions

- 😊 Joy
- 😢 Sadness
- 😡 Anger
- 😨 Fear
- 🤢 Disgust
- 😲 Surprise

---

## Technology Stack

### Backend
- FastAPI
- Transformers
- PyTorch
- Scikit-learn

### Frontend
- HTML
- CSS
- JavaScript

### Machine Learning
- DistilBERT Multilingual
- Hugging Face Transformers

### Dataset
- Custom Emotion Detection Dataset

---

## Usage

1. Enter text in the input box.
2. Click **Analyze Emotion**.
3. The application predicts:
   - Emotion
   - Confidence Score
4. Results are displayed instantly on the UI.
````

This will render cleanly on GitHub, VS Code Markdown Preview, and documentation tools.
