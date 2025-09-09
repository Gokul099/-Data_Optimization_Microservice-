import pandas as pd
import json, uuid, datetime, os
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from utils import mask_names
from qlearning import QLearningAgent, bucketize_quality, sentiment_sign, target_from_sentiment, reward_fn
from storage_simulator import save_to_blob

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/blob", exist_ok=True)

# load spaCy model if available; otherwise use blank
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")

# Use DistilBERT-based sentiment model explicitly
# transformers 'sentiment-analysis' pipeline often defaults to a DistilBERT-based SST-2 model,
# but we explicitly load a known model for determinism.
try:
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception:
    # fallback to default pipeline model (if offline, this may fail)
    sentiment_model = pipeline("sentiment-analysis")

# Simple sequential asset ID generator (asset_001 style)
def _make_asset_ids(n, prefix="asset"):
    return [f"{prefix}_{str(i+1).zfill(3)}" for i in range(n)]

def process_data_pipeline(data):
    """
    1) Ingest & clean
    2) Metadata extraction via spaCy
    3) Sentiment + Q-learning refinement
    4) Save outputs + simulate blob storage
    """
    # Step 1: Data Ingestion & Cleaning
    df = pd.DataFrame(data)
    # ensure required columns exist
    if "text" not in df.columns:
        raise ValueError("Input must contain 'text' field for each record")
    # Fill rating: if all null, default 5
    if "rating" not in df.columns:
        df["rating"] = None
    if df["rating"].dropna().empty:
        df["rating"] = 5
    else:
        df["rating"] = df["rating"].fillna(df["rating"].mean())

    # timestamps: if missing use now
    df["timestamp"] = df.get("timestamp", None)
    df["timestamp"] = df["timestamp"].fillna(datetime.datetime.utcnow().isoformat())

    # assign sequential asset IDs
    df["asset_id"] = _make_asset_ids(len(df), prefix="asset")

    df.to_json("outputs/cleaned_data.json", orient="records", indent=2)

    # Step 2: Metadata Extraction
    metadata = []
    for entry in df["text"]:
        doc = nlp(str(entry))
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        metadata.append({"text": entry, "entities": entities})
    with open("outputs/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 3: ML + Q-learning Refinement
    agent = QLearningAgent()
    results, logs = [], []

    for _, row in df.iterrows():
        orig_text = str(row["text"])
        masked_text = mask_names(orig_text)

        # run sentiment model
        try:
            sent = sentiment_model(masked_text, truncation=True)[0]
            label, conf = sent.get("label"), float(sent.get("score", 0.0))
        except Exception:
            # fallback
            label, conf = "NEUTRAL", 0.0

        base_quality = float(row["rating"]) / 10.0  # scale 0..1 from 0..10
        state = (bucketize_quality(base_quality), sentiment_sign(label) if label else 0)

        action = agent.policy(state)
        new_quality = agent.act_adjust(base_quality, action)

        tgt = target_from_sentiment(label) if label else 0.0
        r = reward_fn(base_quality, new_quality, tgt)
        next_state = (bucketize_quality(new_quality), sentiment_sign(label) if label else 0)

        agent.update(state, action, r, next_state)

        refined = {
            "asset_id": row["asset_id"],
            "text": masked_text,
            "orig_rating": row["rating"],
            "refined_quality": round(new_quality * 10, 2),
            "sentiment": label,
            "confidence": conf,
            "timestamp": row["timestamp"]
        }
        results.append(refined)
        logs.append({"asset_id": row["asset_id"], "original": orig_text, "masked": masked_text, "action": action, "reward": r})

    # write outputs
    with open("outputs/refined_data.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("outputs/log.json", "w") as f:
        json.dump(logs, f, indent=2)
    agent.dump("outputs/q_table.json")

    # Step 4: Simulate external storage (Azure or local)
    try:
        save_to_blob(results, use_azure=True)
    except Exception:
        # ensure fallback saved locally by storage_simulator
        save_to_blob(results, use_azure=False)

    return {"status": "success", "records": len(results)}

def load_results():
    with open("outputs/refined_data.json") as f:
        return json.load(f)