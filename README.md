# Fashion Item Classification with CNN (DL-002)

**Domain:** Computer Vision / Retail  
**Project ID:** DL-002  
**Dataset:** Fashion-MNIST (10 Classes, 70,000 Grayscale 28×28 images)  

A clean, production-grade full-stack computer vision application for classifying clothing and footwear items from the real **Fashion-MNIST** dataset using a Deep Convolutional Neural Network (CNN).

> **Note on Machine Learning Authenticity:**  
> This project is designed exclusively for genuine machine-learning workflows. It uses the real Fashion-MNIST dataset and authentic model checkpoints; no synthetic predictions, fabricated accuracy numbers, or placeholder metrics are used.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Recharts, Lucide React |
| **Backend API** | Python (3.11 / 3.12), FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | TensorFlow / Keras, NumPy, scikit-learn, Matplotlib, Pillow |
| **Research / Exploration** | Jupyter Notebook (`notebooks/`) |

---

## 📁 Project Structure

```text
fashion-item-classification/
├── backend/
│   ├── app/
│   │   ├── __init__.py           # Package definition & metadata
│   │   ├── config.py             # App configuration, directories & class names
│   │   ├── dataset.py            # Real Fashion-MNIST loader, split (54k/6k) & preprocessing
│   │   ├── evaluator.py          # Metrics, confusion matrix, and chart generation module
│   │   ├── model.py              # CNN architecture (Conv2D -> MaxPool -> Dropout -> Dense)
│   │   └── main.py               # FastAPI server & artifact-serving endpoints
│   ├── artifacts/                # Real training history & evaluation artifacts
│   │   ├── accuracy_curve.png    # Training vs validation accuracy plot
│   │   ├── artifact_metadata.json# Run metadata, layer config, seed, consistency flags
│   │   ├── class_distributions.json # Exact sample counts across train/val/test
│   │   ├── confusion_matrix.png  # 10-Class annotated confusion matrix heatmap
│   │   ├── evaluation_metrics.json  # Held-out test accuracy, Macro F1, classification report
│   │   ├── loss_curve.png        # Training vs validation loss plot
│   │   ├── misclassified_grid.png# Grid of authentic misclassified test samples
│   │   ├── sample_predictions.json  # Correct and misclassified test samples with probabilities
│   │   ├── training_curves.json  # Epoch-by-epoch loss & accuracy arrays
│   │   ├── training_history.json # Epoch history dictionary
│   │   └── training_summary.json # High-level training execution summary
│   ├── data/                     # Downloaded / cached Fashion-MNIST data files
│   ├── evaluate.py               # CLI runner for test set evaluation & artifact generation
│   ├── models/                   # Saved trained model files (fashion_mnist_cnn.keras)
│   ├── .env.example              # Example environment variables (CORS_ORIGINS)
│   ├── requirements.txt          # Python ML & server dependencies
│   └── train.py                  # CLI script for running real CNN training pipeline
├── frontend/
│   ├── index.html                # Entry HTML
│   ├── .env.example              # Example environment variables (VITE_API_URL)
│   ├── package.json              # Frontend dependencies & npm scripts
│   ├── postcss.config.js         # PostCSS configuration
│   ├── tailwind.config.js        # Tailwind CSS styling configuration
│   ├── tsconfig.json             # TypeScript configuration
│   ├── vite.config.ts            # Vite configuration & backend proxy
│   └── src/
│       ├── App.tsx               # Main dashboard controller & routing
│       ├── index.css             # Tailwind directives & global styling
│       ├── main.tsx              # React application entry point
│       ├── vite-env.d.ts         # Vite TypeScript type declarations
│       ├── components/           # UI components
│       │   ├── EmptyState.tsx    # "Run training first" screen if artifacts are missing
│       │   ├── ErrorBanner.tsx   # API connection error alert
│       │   ├── Header.tsx        # Top navbar with model status indicator
│       │   ├── LoadingSpinner.tsx# Async data loading spinner
│       │   └── Sidebar.tsx       # Desktop & mobile navigation drawer
│       ├── pages/                # The 6 primary dashboard views
│       │   ├── ArchitecturePage.tsx       # Layer execution flow & hyperparameters
│       │   ├── DatasetExplorerPage.tsx    # Split sizes & class distributions chart
│       │   ├── ErrorAnalysisPage.tsx      # Failure modes, misclassified gallery & modal
│       │   ├── OverviewPage.tsx           # Retail tagging use case & 10 classes
│       │   ├── RealPredictionPage.tsx     # Image upload dropzone & probability chart
│       │   └── TrainingEvaluationPage.tsx # Accuracy/loss curves & confusion matrix
│       ├── services/
│       │   └── api.ts            # Typed API client functions
│       └── types/
│           └── api.ts            # TypeScript interfaces matching FastAPI schemas
├── notebooks/
│   └── fashion_mnist_cnn.ipynb   # Interactive training pipeline notebook
├── .gitignore                    # Git ignore rules for Python, Node, data & weights
└── README.md                     # Project documentation & setup instructions
```

---

## 🚀 Setup & Installation Instructions

### 1. Prerequisites
- **Python 3.11+ or 3.12** (TensorFlow compatible)
- **Node.js 18+ or 20+** and **npm**

---

### 2. Backend Setup (FastAPI & ML Pipeline)

1. Open your terminal and navigate to the `backend/` directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     py -3.12 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3.12 -m venv .venv
     source .venv/bin/activate
     ```

3. Upgrade `pip` and install project dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Execute the real CNN training pipeline:
   ```bash
   python train.py --epochs 15 --batch-size 64 --seed 42
   ```
   - Automatically downloads the Fashion-MNIST dataset from Keras on the first run.
   - Partitions 60,000 samples into **54,000 training** and **6,000 validation** images (stratified).
   - Holds out the official **10,000 test images** for final evaluation.
   - Saves the trained model to `backend/models/fashion_mnist_cnn.keras`.
   - Saves loss & accuracy metrics to `backend/artifacts/training_history.json` and `training_summary.json`.
   - Automatically triggers the evaluation pipeline to produce all JSON artifacts and visualization plots.

5. (Optional) Run the dedicated evaluation script anytime:
   ```bash
   python evaluate.py
   ```
   - Re-evaluates `fashion_mnist_cnn.keras` on all 10,000 held-out test images.
   - Verifies internal mathematical consistency (e.g. confusion matrix sum = 10,000).
   - Updates all structured JSON files and PNG charts in `backend/artifacts/`.

6. Run the automated API test suite:
   ```bash
   python test_api.py
   ```
   - Verifies all 7 REST endpoints with automated unit assertions.
   - Tests file uploads with real images and checks validation for invalid formats.

7. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

8. Verify the backend is running:
   - Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
   - Model Info: [http://127.0.0.1:8000/api/model-info](http://127.0.0.1:8000/api/model-info)
   - Dataset Info: [http://127.0.0.1:8000/api/dataset-info](http://127.0.0.1:8000/api/dataset-info)
   - Real Evaluation Metrics: [http://127.0.0.1:8000/api/metrics](http://127.0.0.1:8000/api/metrics)
   - Confusion Matrix: [http://127.0.0.1:8000/api/confusion-matrix](http://127.0.0.1:8000/api/confusion-matrix)
   - Test Samples: [http://127.0.0.1:8000/api/samples?type=all&limit=20](http://127.0.0.1:8000/api/samples?type=all&limit=20)
   - Interactive Swagger API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 3. Frontend Setup (React Dashboard + Vite + Tailwind CSS)

1. In a separate terminal, navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```

4. Open the interactive dashboard in your browser:
   - **URL:** [http://localhost:5173](http://localhost:5173)

The dashboard contains 6 dedicated pages:
- **Project Overview:** E-commerce retail product-tagging use case, live model/API status, 10 Fashion-MNIST classes.
- **Dataset Explorer:** Stratified split sizes (54,000 train, 6,000 val, 10,000 test), 28×28 grayscale preprocessing explanation, and interactive class distributions bar chart.
- **CNN Architecture:** Visual layer execution flow (Conv2D &rarr; MaxPool &rarr; Dropout &rarr; Dense Softmax), parameter counts (225,034), and deterministic hyperparameters.
- **Training & Evaluation:** Held-out test accuracy (90.73%), macro F1 (90.81%), Recharts accuracy & loss curves over 15 epochs, interactive 10×10 confusion matrix heatmap, and class-wise classification report table.
- **Real Prediction:** Interactive image dropzone, file selector, preset sample picker from test set, live inference via `POST /api/predict`, and 10-class probability distribution bar chart.
- **Error Analysis:** Common retail confusion insights, gallery of authentic test misclassifications, class filters, and individual sample probability inspector modal.

---

### 4. Jupyter Notebook Support

To explore the dataset and run the training pipeline interactively:

1. Activate your virtual environment:
   ```powershell
   cd backend
   .\.venv\Scripts\Activate.ps1
   ```

2. Launch Jupyter Lab or Notebook:
   ```bash
   jupyter lab ../notebooks/
   ```
   Or:
   ```bash
   jupyter notebook ../notebooks/
   ```

---

## 🌐 Deployment (Separate Backend & Frontend)

This project supports independent deployment of the FastAPI backend and React frontend (e.g., Render, Railway, Fly.io, Vercel, Netlify):

1. **Deploy the FastAPI backend first:**
   - Deploy the Python application from the `backend/` directory.
   - **Crucial:** The deployment bundle must include the trained model file `backend/models/fashion_mnist_cnn.keras` and all JSON/PNG files in `backend/artifacts/` (or run `python train.py` and `python evaluate.py` during your host build step).
   - Start command:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```

2. **Configure Backend CORS:**
   - Once your frontend domain is known (e.g., `https://my-fashion-app.vercel.app`), set the `CORS_ORIGINS` environment variable on the backend host:
     ```env
     CORS_ORIGINS=https://my-fashion-app.vercel.app,https://www.my-fashion-app.vercel.app
     ```
   - Multiple origins can be comma-separated. Whitespace is automatically trimmed.
   - Do not use wildcards (`*`) in production.

3. **Deploy the React frontend:**
   - Deploy the frontend from the `frontend/` directory (e.g., to Vercel, Netlify, Cloudflare Pages).
   - Build command: `npm run build`
   - Output directory: `dist`
   - Set the build environment variable pointing to your deployed backend's `/api` prefix:
     ```env
     VITE_API_URL=https://YOUR-BACKEND-DOMAIN/api
     ```
   - **Security note:** `VITE_API_URL` is baked into the public client-side JavaScript bundle and must never contain private tokens or secrets.

4. **Post-Deployment Verification:**
   - Visit `https://YOUR-BACKEND-DOMAIN/api/health` to confirm the backend reports `"status": "healthy"` with `"model_available": true`.
   - Open your deployed frontend URL and verify the live status badge displays `Model & API Online`.
   - Test dashboard metrics, confusion matrix rendering, and upload a test image on the **Real Prediction** page to verify end-to-end inference.

---

## 🧠 CNN Architecture Specification

| Layer | Type | Specifications | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| `input_image` | Input | Grayscale image input | `(None, 28, 28, 1)` | 0 |
| `conv2d_1` | Conv2D | 32 filters, 3×3 kernel, ReLU | `(None, 26, 26, 32)` | 320 |
| `maxpool_1` | MaxPooling2D | 2×2 pool size | `(None, 13, 13, 32)` | 0 |
| `conv2d_2` | Conv2D | 64 filters, 3×3 kernel, ReLU | `(None, 11, 11, 64)` | 18,496 |
| `maxpool_2` | MaxPooling2D | 2×2 pool size | `(None, 5, 5, 64)` | 0 |
| `flatten` | Flatten | Flattens 2D feature maps | `(None, 1600)` | 0 |
| `dense_features` | Dense | 128 hidden units, ReLU | `(None, 128)` | 204,928 |
| `dropout` | Dropout | 30% dropout rate | `(None, 128)` | 0 |
| `output_probabilities` | Dense | 10 units, Softmax activation | `(None, 10)` | 1,290 |

**Total Parameters:** 225,034 Trainable (879 KB)

---

## 📊 Authentic Model Benchmark Results

> **Verification Statement:**  
> **All dashboard metrics, confusion matrices, training curves, and sample predictions are generated from the genuine Fashion-MNIST model training run executed on the held-out test partition.**  
> Zero simulated values, fabricated accuracies, or mock placeholders are used anywhere in the codebase.

The following metrics are the **real results** measured from the trained CNN evaluated on all 10,000 held-out test images:

- **Training Samples:** 54,000 (stratified split, seed 42)
- **Validation Samples:** 6,000 (stratified split, seed 42)
- **Held-Out Test Samples:** 10,000 (strictly non-overlapping)
- **Batch Size:** 64
- **Epochs:** 15
- **Optimizer:** Adam ($LR = 0.001$)
- **Final Training Accuracy:** **95.15%** (Loss: `0.1285`)
- **Final Validation Accuracy:** **92.18%** (Loss: `0.2543`)
- **Held-Out Test Accuracy:** **`90.73%`** (Loss: `0.2910`)
- **Macro F1-Score:** **`90.81%`**
- **Model Checkpoint:** `backend/models/fashion_mnist_cnn.keras` (2.74 MB)

---

## 🏷️ Fashion-MNIST Target Classes

The model classifies 28×28 grayscale images into 10 retail clothing and shoe classes:

0. `T-shirt/top` — Casual short-sleeve tops
1. `Trouser` — Jeans, slacks, and pants
2. `Pullover` — Sweaters, hoodies, and knit jumpers
3. `Dress` — One-piece garments
4. `Coat` — Overcoats, jackets, and winter outerwear
5. `Sandal` — Open-toe warm-weather footwear
6. `Shirt` — Collared button-down shirts
7. `Sneaker` — Athletic and running shoes
8. `Bag` — Handbags, backpacks, and totes
9. `Ankle boot` — High-top boots and winter footwear

---

## 🔄 End-to-End Workflow Verification

```text
[1] Ingest & Split (54k/6k/10k) 
      └──> [2] Train CNN (15 Epochs) 
             └──> [3] Save Checkpoint (fashion_mnist_cnn.keras)
                    └──> [4] Evaluate Held-Out Test Set (10k images)
                           └──> [5] Save Artifacts (JSON & PNG)
                                  └──> [6] Launch FastAPI Server (Port 8000)
                                         └──> [7] Launch Vite React Dashboard (Port 5173)
                                                └──> [8] Live Image Upload & Real Inference
```

---

## 🛠️ Troubleshooting Guide

### 1. PowerShell Script Execution Policy
If `.\.venv\Scripts\Activate.ps1` produces a script execution restriction error on Windows:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Port Conflicts (Port 8000 or 5173 already occupied)
- **Backend:** Change port via CLI:
  ```bash
  uvicorn app.main:app --reload --port 8080
  ```
  *(Update `frontend/vite.config.ts` proxy target if port 8000 is modified)*
- **Frontend:** Vite automatically prompts to use the next open port (e.g., `5174`).

### 3. Missing Model Weights / "Run Training First" Screen
If the React dashboard displays the **"Model or Evaluation Artifacts Not Found"** empty state screen:
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python train.py --epochs 15 --batch-size 64 --seed 42
python evaluate.py
```
This downloads Fashion-MNIST from Keras, trains the CNN, exports all evaluation metrics, and refreshes the dashboard automatically.

### 4. Uploading Real Color Product Photos (Distribution Shift)
The model was trained on **28×28 grayscale** centered benchmark garments on dark backgrounds. For real retail product photos:
- Use the **Auto** or **Invert** background adaptation button on the Prediction page to invert white e-commerce backgrounds.
- High-resolution complex lifestyle photos with humans or noisy backgrounds differ from Fashion-MNIST's training distribution; cropping to the garment ensures best results.
