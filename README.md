# My Content - Hybrid Recommendation MVP

A production-ready MVP demonstrating a **Hybrid SVD++ (session weighting) + Content-Based (cosine on embeddings)** recommender for Globo.com user interactions. The project builds offline artifacts from the provided dataset, exposes recommendations via an Azure Function HTTP API, and offers a simple Flask UI that consumes the API.

## Data location (read-only)

The dataset is expected at `data/news-portal-user-interactions-by-globocom/` relative to the repository root, exactly as listed in `tree.txt`. The project reads:
- `clicks/` directory containing `clicks_hour_XXX.csv` files
- `articles_embeddings.pickle`

The online inference layer exposes the hybrid configuration validated in the notebook:

- SVD++ with session-size weighting (`rating = 1 + log1p(avg_session_size)`)
- Content-based cosine similarity on PCA-reduced embeddings (32 components)
- Rank-fusion weights: 60% collaborative / 40% content

No files under `/data` should be modified.

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
- `article_ids.npy`
- `article_embeddings_pca_32.npy`
- `popular_articles.npy`
- `popularity_scores.pkl`
- `user_clicks.pkl`
- `svdpp_session_weighted.pkl`
- `svdpp_items.npy`

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
