# Cottonil AI Operations Hub

<div align="center">

A complete AI-powered operations system for Cottonil retail data with **7 machine learning models** and an **interactive Gradio web application**.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![Gradio](https://img.shields.io/badge/Gradio-6.31+-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

**[🚀 Live Demo](#-live-demo)** • **[📖 Documentation](#-documentation)** • **[💡 Features](#-features)**

</div>

---

## 💡 Project Idea

Cottonil AI Operations Hub is an end-to-end retail analytics platform designed to optimize operations across the entire retail value chain. The system uses machine learning to answer critical business questions:

- **Demand Forecasting**: Predict weekly sales per branch and category with confidence intervals
- **Inventory Management**: Identify stockout risks and generate prioritized restock plans
- **Customer Intelligence**: Predict churn, segment customers (RFM), and identify high-value at-risk customers
- **Cross-Selling**: Recommend products at checkout based on basket contents and purchase patterns
- **Return Risk**: Predict which order lines are likely to be returned
- **Supplier Risk**: Assess purchase order delivery risks and suggest backup suppliers
- **Marketing Optimization**: Allocate marketing budget across channels for maximum ROI

All models are trained in Jupyter notebooks following a **train → test → evaluate → save** pattern, and served through a user-friendly web interface. The system automatically detects real vs synthetic data and marks all artifacts accordingly.

---

## 🌐 Live Demo

> **[Click here to try the live demo](https://huggingface.co/spaces/YOUR_USERNAME/cottonil-ai-hub)**
---

## ✨ Features

### 📊 Web Application (9 Interactive Tabs)

| Tab | Description |
|-----|-------------|
| **Overview** | KPI dashboard with yearly metrics and key performance charts |
| **Forecast** | 4-week demand forecasts with P10-P90 confidence bands |
| **Restock** | Prioritized transfer recommendations based on forecast + stock levels |
| **Cross-sell** | Real-time product recommendations based on basket contents |
| **Customers** | Churn risk analysis, RFM segments, and win-back call lists |
| **Returns** | Return risk prediction and concentration analysis |
| **Suppliers** | Purchase order risk assessment and backup supplier suggestions |
| **Marketing** | Budget allocation optimizer with channel efficiency analysis |
| **About** | Model cards, data quality checks, and known limitations |

### 🤖 Machine Learning Models

1. **Demand Forecasting** - Gradient Boosting with Poisson loss for 1-4 week horizons
2. **Stockout Risk** - Classifier to predict branch×category stockouts
3. **Customer Churn** - Churn probability + RFM segmentation
4. **Cross-Sell Recommender** - Item-item co-occurrence with margin-aware ranking
5. **Return Risk** - Classifier with signal quality verification
6. **Supplier PO Risk** - Partial receipt risk + lead time regression
7. **Marketing Optimizer** - Log-log response curves + budget allocation

### 🎯 Key Capabilities

- **Egyptian Retail Calendar** - Ramadan, Eid al-Fitr, Eid al-Adha, Black Friday features
- **Data Validation** - Automatic schema validation and data quality checks
- **Model Quality** - Every model tested against baselines on time-based splits
- **Transparent Policies** - Explainable restock rules and recommendation logic
- **Synthetic Data Generation** - Schema-identical test data for development

---

## 📁 Project Structure

```
cottonil-ai/
├── app.py                      # Gradio web application entry point
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development dependencies
├── cottonil/                   # Core business logic package
│   ├── config.py              # Configuration & artifact management
│   ├── data_io.py             # Data loading & validation
│   ├── services.py            # All app logic (views, charts, predictions)
│   ├── forecast.py            # Demand forecasting
│   ├── restock.py             # Restock planning
│   ├── crosssell.py           # Cross-sell recommender
│   ├── customers.py           # Customer analytics (RFM, churn)
│   ├── returns.py             # Return risk features
│   ├── suppliers.py           # Supplier risk features
│   ├── marketing.py           # Marketing optimizer
│   ├── calendar_features.py   # Egyptian retail calendar
│   ├── synth.py               # Synthetic data generator
│   └── visuals.py             # UI visuals (SVG tiles, banners)
├── notebooks/                 # Model training notebooks (00-07)
├── scripts/                   # Utility scripts
│   ├── check_data.py          # Data validation
│   ├── run_all_notebooks.py   # Batch training
│   ├── make_synthetic_data.py # Synthetic data generator
│   └── deploy_to_hf.py        # Hugging Face deployment
├── tests/                     # Automated tests (14 tests)
├── artifacts/                 # Trained models & serving data
│   ├── models/                # .joblib model files
│   ├── serving/               # Runtime CSV files
│   └── metrics/               # Performance metrics JSON
└── COTTONIL_Data/             # Data directory
    ├── 01_CATALOG/           # Product catalog
    └── 02_TABLES/             # Main data tables
```
---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ahmedayad0168/cottonil-HVIA
cd cottonil-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app opens at `http://127.0.0.1:7860` with pre-trained demo models.

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest -q tests

# Run specific test
python -m pytest tests/test_core.py::test_calendar_flags
```

The test suite includes 14 tests covering:
- Calendar features (Ramadan, Eid, Black Friday)
- Schema validation
- Data statistics
- Forecast leakage detection
- Restock policy
- Cross-sell logic
- Churn labels
- Marketing optimizer
- App services

---

## ⚠️ Important Notes

- **Synthetic Data**: Shipped models trained on synthetic data. Orange banner in app indicates this. Retrain with real data for production.
- **Model Quality**: Every model compared against simple baseline on time-based test split. Trust baseline if model doesn't beat it.
- **Marketing Attribution**: Campaign-attributed orders ~45× larger than sales ledger. Use relative efficiency only.
- **Data Source**: System auto-detects real vs synthetic data and marks all artifacts accordingly.

---

## 🔧 Development

### Adding New Models

1. Create notebook in `notebooks/` following `XX_model_name.ipynb` convention
2. Implement training, testing, evaluation
3. Save model with `cottonil.config.save_model()`
4. Save metrics with `cottonil.config.save_metrics()`
5. Add view function to `cottonil/services.py`
6. Wire UI in `app.py`

### Code Style

- Compact, idiomatic Python
- Add tests for new features
- Run `python -m pytest -q tests` before committing
- Ensure notebooks run with `python scripts/run_all_notebooks.py`

---

## 📝 Design Decisions

The project deliberately simplified some ideas from the original blueprint:

- **YOLOv10 packing camera** - No images in data to train vision model
- **Siamese GNN for anonymous orders** - No card/phone/device fields to match on
- **Bayesian MMM** - Insufficient data for adstock/saturation identification
- **Temporal Fusion Transformer** - Gradient boosting used for simplicity and explainability
- **ROI projections** - Only measured test metrics reported, not projections

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Models not found" | Run `python scripts/run_all_notebooks.py` |
| "Missing required columns" | Run `python scripts/check_data.py` |
| Kernel not found | `pip install ipykernel && python -m ipykernel install --user --name python3` |
| Out of memory | Close other apps or use smaller synthetic dataset |
| Model load failure on HF | Ensure scikit-learn version matches (pinned to 1.8.0) |

---

## 📄 License

This project is proprietary to Cottonil. All data and models are confidential.
