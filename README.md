# Fantasy Premier League Dashboard

A dashboard to:
- Analyze player efficiency (points per million, points per 90, form)
- Predict next gameweek's top scorers
- Build the optimal 15-man squad and starting XI under budget constraints

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Fetch latest data:
   ```bash
   python src/fetch_data.py
   ```
2. Build features:
   ```bash
   python src/features.py
   ```
3. Train prediction model:
   ```bash
   python src/train_model.py
   ```
4. Run the optimizer standalone (optional):
   ```bash
   python src/optimizer.py
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```

## Project Structure

```
fantasypl/
  data/
    raw/            # raw API pulls (bootstrap-static, fixtures, gw history)
    processed/       # engineered feature tables
  src/
    fetch_data.py    # pulls data from official FPL API
    features.py      # feature engineering (efficiency scores, form, fixture difficulty)
    train_model.py   # trains next-GW point prediction model
    optimizer.py      # ILP-based squad/XI optimizer using PuLP
  models/            # saved trained models (.pkl)
  app/
    dashboard.py     # Streamlit dashboard
```
