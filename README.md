# Turkish Spam Detection System

A hybrid spam detection system for Turkish e-commerce reviews using BERTurk + TF-IDF feature fusion with weak supervision.

![Label Distribution](results/1_label_distribution.png)

## Description

This project implements a machine learning pipeline for detecting spam in Turkish e-commerce product reviews. The system combines:

- **Weak Supervision**: Automatically generates training labels from heuristic spam signals (brevity, emoji, URL presence, excessive capitalization, repeated characters, and punctuation patterns). Note: since labels are rule-generated rather than human-annotated, reported metrics reflect how well the models learn the labeling heuristics, not agreement with human ground truth.
- **Hybrid Feature Fusion**: Concatenates TF-IDF features with PCA-reduced BERT embeddings (from BERTurk) for a richer text representation
- **Class Balancing**: Random oversampling of the minority (spam) class on the training split only
- **Multi-Model Evaluation**: Compares 5 classification models (Logistic Regression, ANN, Random Forest, LightGBM, CART)
- **Comprehensive Visualizations**: Confusion matrices, AUC curves, t-SNE plots, word clouds, and feature importance analysis
- **Serving**: A FastAPI service (`api.py`) that loads the persisted artifacts and returns spam predictions with rule-based explanations

![Oversampling Effect](results/2_oversampling_effect.png)

## Dataset

The raw dataset (`data/veri_seti_200k.csv`, ~203k Turkish e-commerce reviews) is **not included in the repository** due to its size. Place your own CSV at `data/veri_seti_200k.csv` (one review per line, single header row) before running the pipeline. The path is configurable via `DATA_PATH` in `config.py`.

## Installation

```bash
# Clone or navigate to the project directory
cd turkish-spam-detection_1

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- LightGBM
- matplotlib, seaborn
- wordcloud (optional)

## Usage

### Running the Full Pipeline

```bash
# Execute the complete pipeline
python main.py
```

The pipeline will:
1. Load data from `data/veri_seti_200k.csv`
2. Apply weak supervision labeling based on spam signals
3. Extract hybrid features (TF-IDF + BERT with PCA)
4. Train and evaluate 5 classification models
5. Generate visualizations in the `results/` directory

### Training & Persisting a Model for Serving

```bash
# Train the ANN pipeline and save all artifacts to models/
python save_model.py
```

This writes `tfidf_vectorizer.joblib`, `pca_model.joblib`, `scaler.joblib`, `best_model.joblib`, and `model_info.json` into `models/`.

### Running the API

```bash
# Requires the artifacts produced by save_model.py
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints: `POST /predict`, `POST /predict/batch` (max 50 texts), `GET /health`, `GET /info`.

### Ablation Study

```bash
# Compares TF-IDF only vs. BERT only vs. hybrid features
python ablation.py
```

### Configuration

Edit `config.py` to customize the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_PATH` | `data/veri_seti_200k.csv` | Input data file path |
| `BERT_MODEL` | `dbmdz/bert-base-turkish-cased` | BERT model for embeddings |
| `SAMPLE_SIZE` | `None` | Subsample size (`None` = use full dataset) |
| `SPAM_THRESHOLD` | `1.0` | Min. number of spam signals to label a text as spam |
| `TFIDF_FEATURES` | `500` | Maximum TF-IDF features |
| `PCA_COMPONENTS` | `756` | BERT dimensions after PCA |
| `OVERSAMPLING_RATIO` | `1.0` | Minority:majority ratio after oversampling |

### Model Comparison Results

![Model Comparison](results/4_top3_comparison.png)

![AUC Curve](results/5_auc_curve.png)

### Visualizations Generated

The system generates 9 analysis plots saved to `results/`:

| File | Description |
|------|-------------|
| `1_label_distribution.png` | Distribution of spam vs ham labels |
| `2_oversampling_effect.png` | Impact of random oversampling |
| `3_best_model_cm.png` | Confusion matrix of best model |
| `4_top3_comparison.png` | Performance comparison of top 3 models |
| `5_auc_curve.png` | ROC-AUC curves for all models |
| `6_wordclouds.png` | Word clouds for spam and ham reviews |
| `7_tsne_plot.png` | t-SNE visualization of feature space |
| `8_pca_variance.png` | PCA variance explanation |
| `9_feature_importance.png` | Feature importance analysis |

![Word Clouds](results/6_wordclouds.png)

![t-SNE Plot](results/7_tsne_plot.png)

![Feature Importance](results/9_feature_importance.png)

## Project Structure

```
turkish-spam-detection_1/
├── config.py              # Configuration settings
├── preprocessing.py       # Turkish text cleaning and stemming
├── labeling.py            # Weak supervision heuristics
├── features.py            # TF-IDF and BERT feature extraction
├── train.py               # Model training and evaluation
├── visualize.py           # Analysis plots generation
├── main.py                # Pipeline orchestration (train + evaluate + plot)
├── save_model.py          # Train ANN pipeline and persist artifacts
├── api.py                 # FastAPI serving layer
├── ablation.py            # TF-IDF vs. BERT vs. hybrid ablation
├── frontend-integration/  # Example React/TypeScript client
├── requirements.txt       # Python dependencies
├── data/                  # Input data (not included in repo)
├── models/                # Persisted model artifacts
└── results/               # Generated visualizations
```

## Contributing

We welcome contributions! To contribute:

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/YourFeature`)
3. **Make your changes** and ensure code follows existing patterns
4. **Test thoroughly** with the existing pipeline
5. **Commit** your changes (`git commit -m 'Add some feature'`)
6. **Push** to the branch (`git push origin feature/YourFeature`)
7. **Open a Pull Request** with a clear description of your changes

### Guidelines

- Follow PEP 8 style guidelines for Python code
- Add docstrings to new functions and classes
- Ensure new features work with the existing configuration system
- Test changes with the full pipeline before submitting

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Turkish Spam Detection Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

*Built with BERTurk + TF-IDF hybrid feature fusion for Turkish language spam detection.*
