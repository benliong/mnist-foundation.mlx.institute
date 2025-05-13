# MNIST Digit Recognizer 🖌️🔢

A full‑stack, containerised application that lets anyone draw a digit in the browser and instantly see the model's prediction and confidence—all while logging real‑world feedback to continuously improve the model.

> **Built as the Foundation Project for the MLX Institute six‑week Machine‑Learning residency.**

---

## 🌐 Live Demo

[http://37.27.205.20:8501](http://37.27.205.20:8501)

*(Running on a self‑managed Hetzner CX22 instance, Docker‑orchestrated.)*

---

## 🚀 Key Features

| Feature                 | Where it happens                                   | Notes                                                                                                    |
| ----------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Draw & Predict**      | `app.py` (Streamlit + `streamlit‑drawable‑canvas`) | Digit canvas → 28 × 28 tensor → model → top‑1 digit & softmax confidence                                 |
| **Feedback Loop**       | `app.py → PostgreSQL`                              | Users can correct the model's guess; entries stored in **`prediction_logs`** table for future retraining |
| **CNN Training**        | `train.py` (PyTorch)                               | 2 × Conv + ReLU → MaxPool → FC → 10‑way softmax; 98 % test accuracy in 5 epochs                          |
| **Containerised stack** | `docker-compose.yml`                               | `web` (Streamlit) • `db` (PostgreSQL) • optional `trainer` for periodic retraining                        |
| **One‑click redeploy**  | GitHub → `docker compose pull && up -d`            | Stateless web layer : model weights baked into image (`model.pt`)                                        |

---

## 🗂️ Repository Layout

```
├── app.py               # Streamlit UI + inference + logging
├── train.py             # MNIST training script — saves model.pt
├── config.py            # Hyper‑parameters & DB defaults
├── Dockerfile           # Builds web image with model
├── docker-compose.yml   # Web + PostgreSQL services
├── requirements.txt     # Python deps
└── README.md            # ← you are here
```

---

## 📄 How `app.py` Works

1. **Load model** — `torch.load("model.pt")` on CPU (≈ 4 MB).
2. **Canvas** — `st_canvas()` collects a 280 × 280 drawing, down‑sampled to 28 × 28.
3. **Pre‑process** — normalise with `MNIST_MEAN / MNIST_STD` from `config.py`.
4. **Inference** — forward pass through the frozen `Net` CNN defined in *train.py*.
5. **Display** — predicted digit & bar chart of per‑class probabilities.
6. **Logging** — `save_prediction()` writes `(timestamp, prediction, confidence, true_label, image_data)` to **PostgreSQL** via `psycopg2`.

Environment variables override the defaults in `config.py`, so the same image runs in dev and prod without changes.

---

## 🏋️‍♂️ Training (`train.py`)

```bash
python train.py            # ~2 minutes on laptop CPU
```

*Hyper‑params*

| Var (config.py)   | Value |
| ----------------- | ----- |
| `BATCH_SIZE`      | 128   |
| `EPOCHS`          | 5     |
| `DEVICE`          | Auto-detects CUDA, falls back to CPU |
| `LR` (hard‑coded) | 1e‑3  |

`train.py` saves *model.pt* in the project root. The architecture is:

```text
Conv2d(1→32,3) → ReLU → Conv2d(32→64,3) → ReLU → MaxPool(2)
→ Flatten → Linear(64·12·12 → 128) → ReLU → Linear(128 → 10)
```

---

## 🐳 Running with Docker

```bash
git clone https://github.com/benliong/mnist-foundation.mlx.institute.git
cd mnist-foundation.mlx.institute

# Build images
docker compose build

# Edit .env if you want custom DB creds/ports
cp .env.example .env && nano .env

# IMPORTANT: Train the model first - this creates model.pt
docker compose run --rm trainer

# Fire it up
docker compose up -d web # http://localhost:8501
```

Compose brings up:

| Service   | Port | Image                     | Purpose              |
| --------- | ---- | ------------------------- | -------------------- |
| `web`     | 8501 | `digit-recognizer:latest` | Streamlit UI + model |
| `db`      | 5432 | `postgres:16-alpine`      | PostgreSQL datastore |

> **Tip:** To inspect logged rows:
> `docker exec -it db psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT * FROM prediction_logs LIMIT 5;"`

---

## 🔄 Iterative Improvement

1. Periodically export new drawings + labels from `prediction_logs`.
2. Append them to the original training set.
3. Re‑run `train.py` → produces a fresh `model.pt`.
4. `docker compose build web && docker compose up -d web` to hot‑swap the model.

Automating this with a **`trainer`** service is sketched in *docker-compose.yml* (commented‑out).

---

## 📑 Application Checklist

✔️ Meets all deliverables: training, UI, logging, Docker & deployment citeturn0file0

---

## ✍️ Author

[Ben Liong](https://github.com/benliong) — *MLX Institute 2025 applicant*

---

*Questions or suggestions? Open an issue or ping me at [benliong@gmail.com](mailto:benliong@gmail.com)*
