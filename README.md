# MedScanAI

MedScanAI is an AI-powered medical imaging application with a React (Vite + TypeScript + shadcn-ui) frontend and a FastAPI backend. It supports skin lesion classification, brain MRI tumor classification, X-ray report analysis, and utility features like HealthScan and meal planning.

## Features

- **Skin Analysis (PyTorch)**: EfficientNet-B0, ResNet18, and a DermNet ResNet50 model
- **Brain MRI Classification**: 4 classes (glioma, meningioma, pituitary, notumor)
- **X-ray Report Analysis**: Text-based analysis using a fine-tuned BERT model
- **Gemini AI (optional)**: Enhanced analysis when `GEMINI_API_KEY` is set
- **Health tools**: HealthScan, nutrition analysis, and meal planner (frontend pages)

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, shadcn-ui, React Router, TanStack Query
- **Backend**: FastAPI, Pydantic, Python 3.11, PyTorch, torchvision, transformers, Pillow, NumPy, python-dotenv, (optional) google-generativeai

## Project Structure

```
MedScanAI/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                 # FastAPI app
│  │  ├─ models_loader.py        # Model loading helpers
│  │  ├─ schemas.py              # Pydantic schemas
│  │  ├─ utils/                  # GradCAM, OCR utils
│  │  ├─ model/                  # .pth model files
│  │  └─ ...
│  ├─ .env                       # Backend environment (optional)
│  └─ ...
├─ frontend/
│  ├─ src/                       # React app (pages, components, hooks)
│  ├─ package.json               # Vite + TS React app
│  └─ ...
└─ README.md
```

## Prerequisites

- Node.js 18+ and npm
- Python 3.11 (recommended) with pip
- (Optional) Google Gemini API key for enhanced MRI analysis

## Backend — Setup & Run

1. Create a virtual environment (recommended) and activate it.
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn[standard] python-dotenv pillow numpy torch torchvision transformers google-generativeai
   ```
3. Ensure model files exist in `backend/app/model/`:
   - `model.pth` (EfficientNet skin)
   - `resnet18_skin_cancer.pth`
   - `dermnet_resnet50_full.pth`
   - `brain_tumor_classifier_final.pth`
4. Configure environment (optional) in `backend/.env`:
   ```env
   GEMINI_API_KEY=your_key_here
   ```
5. Start the backend from repository root:
   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Key API Endpoints

- `GET /` — Health check and service info
- `POST /predict/efficientnet` — Skin lesion classification
- `POST /predict/resnet18` — Skin lesion classification
- `POST /predict/dermnet_resnet50` — Skin conditions (DermNet classes)
- `POST /predict/xray-image` — X-ray image/report analysis
- `POST /predict/mri` — Brain MRI tumor classification (+ optional Gemini enhancement)

Note: Endpoints may include additional routes for HealthScan, nutrition analysis, and meal planning if enabled in the backend.

## Frontend — Setup & Run

1. Navigate to `frontend/` and install dependencies:
   ```bash
   npm install
   ```
2. Start the dev server:
   ```bash
   npm run dev
   ```
3. The app routes include:
   - `/` — Landing page
   - `/xray-analysis` — X-ray Analysis UI
   - `/mri-analysis` — MRI Analysis UI
   - `/health-hub` — Health Hub
   - `/health-scan` — HealthScan
   - `/meal-planner` — Meal Planner

## Environment Variables

- **Backend**: `GEMINI_API_KEY` (optional) in `backend/.env`
- **Frontend**: If needed, add Vite env vars in `frontend/.env` using `VITE_` prefix

## Model Notes

- PyTorch models are loaded from `backend/app/model/`
- Ensure the class lists in `models_loader.py` match the checkpoints used
- Grad-CAM is applied when supported layers are available

## Security & Privacy

- This project is for educational/research purposes and is not a medical device
- Do not upload personally identifiable information (PII)

## Contributing

This repository is owned by the project author. External attributions and unrelated third‑party promotional content have been removed. If you wish to contribute, please open an issue first to discuss your proposal.

## License

All rights reserved unless a license is added. Contact the owner for permissions.

## Maintainer

- MedScanAI — Project Owner

---

Tip: A repo summary file `.zencoder/rules/repo.md` is missing. I can generate it to improve future maintenance and automated assistance if you want.
