# Smart Ticket Classifier (ML + NLP)

End-to-end **customer support ticket AI** that predicts:

- **Priority** (Low / Medium / High / Urgent) — SLA-aligned triage
- **Issue category**
- **Closest product / subject**
- **Estimated resolution time (hours)**

Built with **TF-IDF + Calibrated LinearSVC**, Streamlit UI, and a reproducible train → save → infer pipeline.

## Highlights

| Metric | Result |
|--------|--------|
| Priority accuracy (hold-out) | **~98%** |
| Macro F1 | **~0.91** |
| Top-2 accuracy | **~99.8%** |

### Why accuracy jumped from ~30% → ~98%

Historical priority labels were noisy: a few duplicated complaint templates (~200k rows) were labeled almost randomly across classes. This project:

1. Removes multi-label noisy templates  
2. Deduplicates unique complaints  
3. Learns an **SLA triage priority** from text urgency + category risk policy  
4. Trains a calibrated LinearSVC with engineered keyword features  

That data-quality story is intentional — it mirrors real support-ops ML work.

## Project structure

```
smart-ticket-classifier/
├── README.md
├── requirements.txt
├── app.py                  # Fast Streamlit inference (uses models/)
├── train_models.py         # Train once & save artifacts
├── train_and_save.py       # Alias for train_models.py
├── ML_Final_project.py     # Runtime-train Streamlit demo
├── ML_Project.py           # Earlier iterative version
├── Customer Support Ticket Classifier.ipynb
└── customer_support_tickets.csv   # Sample data (optional)
```

> Full ~200k dataset (`customer_support_tickets_200k.csv` / `.zip`) is not committed (size). Place it next to `train_models.py` before training.

## Setup

```bash
git clone https://github.com/<your-username>/smart-ticket-classifier.git
cd smart-ticket-classifier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
# Put customer_support_tickets_200k.zip (or .csv) in this folder first
python train_models.py
```

Creates `models/` with vectorizer, classifiers, encoders, and metrics.

## Run

```bash
streamlit run app.py
```

Or runtime training (slower first launch):

```bash
streamlit run ML_Final_project.py
```

## Tech stack

- Python, pandas, scikit-learn, SciPy, NLTK  
- Streamlit  
- NLP: cleaning, TF-IDF (1–2 grams), urgency keyword features  

## License

Personal / portfolio project.
