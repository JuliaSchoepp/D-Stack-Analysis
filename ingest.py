#!/usr/bin/env python3
"""
Automated ingestion pipeline for D-Stack feedback issues.
Fetches issues from GitLab, enriches them with sentiment, labels, and organization attribution,
then saves to Parquet for analysis.

This script is designed to run in CI/CD pipelines (e.g., GitHub Actions) on a schedule.
Supports incremental updates: only processes new issues since last successful run.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from google import genai
from google.auth import default
from google.auth.transport.requests import Request
from google.cloud import language_v2, storage
from google.genai import types
from google.oauth2 import service_account

import utils as u

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
KEYWORDS_VERSION = 1
PROJECT_PATH = "dstack/d-stack-home"
DATA_DIR = Path("data")
KEYWORDS_PATH = Path(f"keyword_lists/keywords_config_v{KEYWORDS_VERSION}.txt")

DATA_DIR.mkdir(exist_ok=True)

POSTPROCESSED_PATH = DATA_DIR / "issues_postprocessed"
POSTPROCESSED_PATH.mkdir(exist_ok=True)
RUN_METADATA_PATH = DATA_DIR / "run_metadata.json"

PROJECT_ID = "project-8415b93b-4a16-4c2b-901"
LOCATION = "europe-west3"
MODEL = "gemini-2.0-flash"

BATCH_SIZE = 10
RATE_LIMIT_SLEEP = 2
RATE_LIMIT_ERROR_SLEEP = 5

# Configuration constants
IDS_TO_EXCLUDE = list(range(1, 9))
COLUMNS_TO_KEEP = [
    "iid", "title", "description", "state", "created_at", "updated_at",
    "closed_at", "author_id", "author_name", "author_state",
    "user_notes_count", "upvotes", "downvotes", "references"
]
DESC_TO_EXCLUDE = ["", "test", "Test"]
EXCLUDE_PAGES = ['/beteiligung?utm_source=chatgpt.com', '/wtf', '/landkarte/ Tech-Stack Aufnahmekriterien & Prozess']

# Load labels from config
LABELS = [
    line.strip()
    for line in KEYWORDS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

logger.info(f"Loaded {len(LABELS)} labels from {KEYWORDS_PATH}")


# Run Metadata Management
def load_run_metadata() -> dict:
    """Load metadata from last successful run."""
    if RUN_METADATA_PATH.exists():
        try:
            with open(RUN_METADATA_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load run metadata: {e}")
    return {"last_successful_run": None, "last_fetched_issues": 0}


def save_run_metadata(metadata: dict):
    """Save metadata from current run."""
    try:
        with open(RUN_METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save run metadata: {e}")


def upload_to_gcs(partition_path: Path, bucket_name: str = "dstack-feedback") -> bool:
    """Upload partition to GCS bucket."""
    try:
        creds_json = os.environ.get("GCP_CREDENTIALS")
        if not creds_json:
            logger.warning("GCP_CREDENTIALS not set, skipping GCS upload")
            return False
        
        # Parse credentials from JSON string
        import tempfile
        creds_dict = json.loads(creds_json)
        
        # Create client with credentials
        client = storage.Client.from_service_account_info(creds_dict)
        bucket = client.bucket(bucket_name)
        
        # Upload the parquet file
        parquet_file = partition_path / "issues.parquet"
        if parquet_file.exists():
            blob_path = f"data/{partition_path.name}/issues.parquet"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(parquet_file))
            logger.info(f"Uploaded {blob_path} to gs://{bucket_name}/")
            return True
        else:
            logger.warning(f"Parquet file not found: {parquet_file}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}", exc_info=True)
        return False


# Initialize clients
def init_sentiment_client():
    """Initialize Google Cloud Language API client for sentiment analysis."""
    try:
        return language_v2.LanguageServiceClient()
    except Exception as e:
        logger.error(f"Failed to initialize sentiment client: {e}")
        raise


def init_genai_client():
    """Initialize Vertex AI GenAI client for labeling and attribution."""
    try:
        # Explicitly pass Project ID and Location
        return genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {e}")
        raise


# Sentiment Analysis
def get_sentiment_score(client, text_content: str) -> float:
    """
    Analyze sentiment of text using Google Cloud Language API.
    Returns score between -1.0 and 1.0. Returns 0.0 on error.
    """
    if not text_content:
        return 0.0

    try:
        document = language_v2.Document(
            content=text_content,
            type_=language_v2.Document.Type.PLAIN_TEXT
        )
        response = client.analyze_sentiment(document=document)
        return response.document_sentiment.score
    except Exception as e:
        logger.warning(f"Error analyzing sentiment: {e}")
        return 0.0


def add_sentiment_scores(df: pl.DataFrame, client) -> pl.DataFrame:
    """Add sentiment scores to all issues."""
    logger.info("Adding sentiment scores...")
    df = df.with_columns(
        pl.col("desc_clean")
        .map_elements(lambda x: get_sentiment_score(client, x), return_dtype=pl.Float64)
        .alias("sentiment")
    )
    return df


# Multi-label Classification
LABELING_SYSTEM_INSTRUCTION = (
    "Du bekommst GitLab-Issues aus dem Deutschland-Stack-Konsultationsverfahren. "
    "Du bist ein Klassifizierungs-Experte. Deine Aufgabe ist es, die Beschreibung zu analysieren "
    "und sie anhand der Labels in der Liste zu klassifizieren. "
    "Nutze NUR die zur Verfügung gestellten Labels. Erfinde keine neuen Labels! "
    "Stelle das Ergebnis als Komma-separierten String zur Verfügung. "
    "Der String enthält NUR die von dir vergebenen Labels (eins oder bis zu 5). "
    "Versuche nur so viele Labels wie nötig zu vergeben. "
    "Wenn kein Label passt, nutze das Label Unklar"
)


def validate_labels(labels_str: str, allowed_labels: list) -> list:
    """Filter labels to only those in allowed_labels."""
    labels = [label.strip() for label in labels_str.split(",")]
    return [label for label in labels if label in allowed_labels]


def classify_issue_multilabel(client, issue_text: str, labels: list) -> list:
    """Classify a single issue with multiple labels using GenAI."""
    user_prompt = f"""
Labels:
{", ".join(labels)}

Issue:
{issue_text}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=LABELING_SYSTEM_INSTRUCTION + "\n\n" + user_prompt)],
                ),
            ],
        )
        labels_str = response.text.strip()
        return validate_labels(labels_str, labels)
    except Exception as e:
        logger.error(f"Error classifying issue: {e}")
        return []


def add_labels(df: pl.DataFrame, client) -> pl.DataFrame:
    """Add multi-label classification to issues."""
    logger.info("Adding labels...")

    df_labeled = df.clone()
    labels_list = [[] for _ in range(len(df))]

    # Process in batches
    for i in range(0, len(df_labeled), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(df_labeled))
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"Labeling batch {batch_num}: rows {i} to {end_idx - 1}")

        for j in range(end_idx - i):
            row_idx = i + j
            desc = df_labeled["desc_clean"][row_idx]

            try:
                labels_result = classify_issue_multilabel(client, desc, LABELS)
                labels_list[row_idx] = labels_result
                time.sleep(RATE_LIMIT_SLEEP)
            except Exception as e:
                logger.error(f"Error processing row {row_idx}: {e}")
                labels_list[row_idx] = []
                time.sleep(RATE_LIMIT_ERROR_SLEEP)

        # Update with current batch
        df_labeled = df_labeled.with_columns(
            pl.Series(f"labels_v{KEYWORDS_VERSION}", labels_list)
        )
        logger.info(f"Completed batch {batch_num}")

    return df_labeled


# Organization Attribution
ORG_SYSTEM_INSTRUCTION = (
    "Du bekommst GitLab-Issues aus dem Deutschland-Stack-Konsultationsverfahren. "
    "Deine Aufgabe ist es, die Issues einer Organisation zuzuordnen, wenn möglich. "
    "Ordne sie NUR EINER ORGANISATION zu. Erfinde keine neuen Organisationen! "
    "Stelle das Ergebnis als Text zur Verfügung. "
    "Wenn keine Organisation erkennbar ist, antworte mit 'Unklar'. "
    "Beispiel: "
    "Konsultationsbeitrag der publicplan GmbH zum Deutschland-Stack "
    "Deine Antwort: publicplan GmbH"
)


def add_issue_org_attribution(client, title: str, issue_text: str) -> str:
    """Identify organization for a single issue using GenAI."""
    user_prompt = f"""
Titel:
{title}

Inhalt:
{issue_text}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=ORG_SYSTEM_INSTRUCTION + "\n\n" + user_prompt)],
                ),
            ],
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error attributing organization: {e}")
        return "Unklar"


def add_organization_attribution(df: pl.DataFrame, client) -> pl.DataFrame:
    """Add organization attribution to issues."""
    logger.info("Adding organization attribution...")

    df_org = df.clone()
    org_list = ["" for _ in range(len(df))]

    # Process in batches
    for i in range(0, len(df_org), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(df_org))
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"Attribution batch {batch_num}: rows {i} to {end_idx - 1}")

        for j in range(end_idx - i):
            row_idx = i + j
            title = df_org["title"][row_idx]
            desc = df_org["desc_clean"][row_idx]

            try:
                org = add_issue_org_attribution(client, title, desc)
                org_list[row_idx] = org
                time.sleep(RATE_LIMIT_SLEEP)
            except Exception as e:
                logger.error(f"Error processing row {row_idx}: {e}")
                org_list[row_idx] = "Unklar"
                time.sleep(RATE_LIMIT_ERROR_SLEEP)

        # Update with current batch
        df_org = df_org.with_columns(pl.Series("org", org_list))
        logger.info(f"Completed batch {batch_num}")

    return df_org


# Main Pipeline
def main():
    """Run the complete ingestion pipeline."""
    logger.info("Starting D-Stack feedback ingestion pipeline")
    
    # Load metadata from last run
    metadata = load_run_metadata()
    last_run_iso = metadata.get("last_successful_run")

    try:
        # Step 1: Fetch issues from GitLab
        logger.info("Step 1: Fetching issues from GitLab...")
        issues = u.fetch_all_gitlab_issues(PROJECT_PATH)
        logger.info(f"Fetched {len(issues)} issues from GitLab")

        # Filter to only new issues (created after last successful run)
        if last_run_iso:
            logger.info(f"Filtering issues created after {last_run_iso}")
            last_run_dt = datetime.fromisoformat(last_run_iso.replace('Z', '+00:00'))
            
            new_issues = []
            for issue in issues:
                created_at_str = issue.get("created_at", "")
                created_at_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                if created_at_dt > last_run_dt:
                    new_issues.append(issue)
            
            logger.info(f"Filtered to {len(new_issues)} new issues")
            issues = new_issues

        if not issues:
            logger.info("No new issues to process. Exiting.")
            return True

        new_df = pl.DataFrame(issues)

        # Step 2: Clean and prepare data
        logger.info("Step 2: Cleaning and preparing data...")
        df_clean = u.clean_issues_df(new_df, COLUMNS_TO_KEEP, IDS_TO_EXCLUDE)
        df_prepared = u.prepare_issues_df(df_clean, DESC_TO_EXCLUDE)
        logger.info(f"Prepared {len(df_prepared)} issues for enrichment")

        # Step 3: Add sentiment scores
        logger.info("Step 3: Adding sentiment scores...")
        sentiment_client = init_sentiment_client()
        df_enriched = add_sentiment_scores(df_prepared, sentiment_client)

        # Step 4: Add labels
        logger.info("Step 4: Adding multi-label classification...")
        genai_client = init_genai_client()
        df_labeled = add_labels(df_enriched, genai_client)

        # Step 5: Add organization attribution
        logger.info("Step 5: Adding organization attribution...")
        df_org = add_organization_attribution(df_labeled, genai_client)

        # Step 6: Postprocess
        logger.info("Step 6: Postprocessing...")
        df_postprocessed = u.postprocess_issues(df_org, KEYWORDS_VERSION)

        # Step 7: Add feedback round based on creation date
        logger.info("Step 7: Adding feedback round indicator...")
        df_postprocessed = df_postprocessed.with_columns(
            pl.col("created_at").dt.year().map_elements(
                lambda year: 2 if year == 2026 else 1,
                return_dtype=pl.Int8
            ).alias("feedback_round")
        )

        # Step 8: Save to date-partitioned parquet
        logger.info("Step 8: Saving to date-partitioned storage...")
        processing_date = datetime.utcnow().strftime("%Y-%m-%d")
        partition_path = POSTPROCESSED_PATH / f"processing_date={processing_date}"
        partition_path.mkdir(parents=True, exist_ok=True)
        df_postprocessed.write_parquet(partition_path / "issues.parquet")
        logger.info(f"Pipeline complete. Saved {len(df_postprocessed)} issues to {partition_path}")

        # Step 9: Upload to GCS
        logger.info("Step 9: Uploading to GCS...")
        upload_to_gcs(partition_path)

        # Update run metadata: use the max created_at from the data
        max_created_at = df_postprocessed.select(pl.col("created_at")).max()[0, 0]
        metadata = {
            "last_successful_run": max_created_at.isoformat() + "Z" if max_created_at else None,
            "last_fetched_issues": len(issues),
            "total_issues_processed": len(df_postprocessed)
        }
        save_run_metadata(metadata)
        logger.info(f"Metadata saved. Next run will process issues created after {metadata['last_successful_run']}")

        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
