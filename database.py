"""
Supabase database module for storing human preference data.
Implements the stretch goal of persistent cloud storage (Supabase/Postgres).

Required Supabase table — run this SQL in the Supabase SQL Editor:

    CREATE TABLE preferences (
        id BIGSERIAL PRIMARY KEY,
        timestamp TEXT NOT NULL,
        prompt TEXT NOT NULL,
        response_a TEXT NOT NULL,
        response_b TEXT NOT NULL,
        preference TEXT NOT NULL CHECK (preference IN ('A', 'B', 'tie')),
        chosen TEXT NOT NULL,
        rejected TEXT NOT NULL,
        model_name TEXT NOT NULL,
        temperature REAL NOT NULL,
        max_tokens INTEGER NOT NULL,
        response_a_tokens INTEGER,
        response_b_tokens INTEGER,
        generation_time_a REAL,
        generation_time_b REAL,
        prompt_category TEXT,
        session_id TEXT,
        reasoning TEXT
    );
"""

import os
from dotenv import load_dotenv
from supabase._sync.client import Client, ClientOptions

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

TABLE_NAME = "preferences"

# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

_client: Client | None = None


def _get_client() -> Client:
    """Return the active Supabase client, creating it on first call."""
    global _client
    if _client is None:
        _client = Client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions())
    return _client


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def save_preference(record: dict):
    """Insert a single preference record into the Supabase table."""
    client = _get_client()
    client.table(TABLE_NAME).insert(record).execute()


def get_all_preferences() -> list[dict]:
    """Retrieve all preference records ordered by id."""
    client = _get_client()
    response = client.table(TABLE_NAME).select("*").order("id").execute()
    return response.data


def get_training_pairs() -> list[dict]:
    """
    Export data in the standard {prompt, chosen, rejected} format
    suitable for downstream RLHF / DPO training.
    """
    client = _get_client()
    response = (
        client.table(TABLE_NAME)
        .select("prompt, chosen, rejected, model_name, timestamp, preference")
        .order("id")
        .execute()
    )
    return response.data


def get_record_count() -> int:
    """Return the total number of preference records."""
    client = _get_client()
    response = client.table(TABLE_NAME).select("id", count="exact").execute()
    return response.count or 0


def get_preferred_for_prompt(prompt: str) -> dict | None:
    """
    Look up all previous preferences for this exact prompt.
    Returns a summary with the latest chosen response, total votes,
    and preference breakdown — or None if no match exists.
    """
    client = _get_client()
    response = (
        client.table(TABLE_NAME)
        .select("chosen, rejected, preference, model_name, timestamp")
        .eq("prompt", prompt)
        .order("id", desc=True)
        .execute()
    )
    rows = response.data
    if not rows:
        return None

    # Tally preference breakdown
    counts = {"A": 0, "B": 0, "tie": 0}
    for r in rows:
        counts[r["preference"]] = counts.get(r["preference"], 0) + 1

    latest = rows[0]
    return {
        "chosen_response": latest["chosen"],
        "model_name": latest["model_name"],
        "timestamp": latest["timestamp"],
        "total_votes": len(rows),
        "preference_breakdown": counts,
    }
