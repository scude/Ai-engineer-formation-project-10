# My Content - Hybrid Co-visitation Recommendation MVP

A production-ready MVP demonstrating the tuned **E3-1** hybrid co-visitation + popularity recommender for Globo.com user interactions. The project builds offline artifacts from the provided dataset, exposes recommendations via an Azure Function HTTP API, and offers a simple Flask UI that consumes the API.

## Data location (read-only)

The dataset is expected at `data/news-portal-user-interactions-by-globocom/` relative to the repository root, exactly as listed in `tree.txt`. The project reads:
- `clicks/` directory containing `clicks_hour_XXX.csv` files
- `articles_embeddings.pickle`

The online inference layer exposes the **E3-1** recommender configuration with tuned hyperparameters:

- `covisit_hybrid_alpha = 0.7350738721058192`
- `covisit_top_n_neighbors = 20`

No files under `/data` should be modified.

The evaluation notebook now compares four strategies side by side: native SVD (Surprise),
session-size-weighted SVD, a LightFM-style item-to-item model leveraging user context,
and a 60/40 hybrid between the two families.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate artifacts (offline)

Artifacts are written to `artifacts/` (override with `ARTIFACTS_DIR`).

```bash
python -m src.train.build_artifacts
```

Generated files:
- `popular_articles.npy`
- `popularity_scores.pkl`
- `covisit_similarity.pkl`
- `user_clicks.pkl`

## Azure Function (local)

1. Install Azure Functions Core Tools and activate the virtualenv.
2. Copy `azure_function/RecommendFunction/local.settings.json.example` to `local.settings.json` and update `ARTIFACTS_DIR` if needed.
3. Start the function host:

```bash
cd azure_function/RecommendFunction
func start
```

The HTTP endpoint will be available at `http://localhost:7071/api/recommend`.

### Example curl

```bash
curl -X POST \
  http://localhost:7071/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345}'
```

## Flask UI

The Flask UI loads available user IDs from the artifacts and calls the Azure Function to display recommendations.

```bash
export AZURE_FUNCTION_URL="http://localhost:7071/api/recommend"  # optional override
flask --app app/app.py run
```

Open http://localhost:5000 to interact with the UI.

## Project structure

```
src/                 # Data, modeling, training, and inference code
azure_function/      # Azure Function entrypoint
app/                 # Flask UI
artifacts/           # Generated recommendation artifacts (not committed)
```

