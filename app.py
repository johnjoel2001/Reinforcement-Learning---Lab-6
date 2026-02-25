"""
Human Preference Data Collection App
=====================================
A Streamlit app that generates two responses from the same LLM for a given prompt,
lets the user select which response they prefer (or mark a tie), and stores the
preference data in a SQLite database for downstream RLHF / DPO training.
"""

import streamlit as st
import json
import time
import uuid
import os
from datetime import datetime

import openai
import pandas as pd

from database import save_preference, get_all_preferences, get_training_pairs, get_record_count

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")

AVAILABLE_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-4o",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt_dataset() -> list[dict]:
    """Load the curated prompt dataset from disk."""
    with open(PROMPTS_FILE, "r") as f:
        return json.load(f)


def generate_response(client: openai.OpenAI, model: str, prompt: str,
                      temperature: float, max_tokens: int) -> dict:
    """
    Call the OpenAI API and return the response text along with metadata.
    """
    start = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start
    msg = completion.choices[0].message.content
    usage = completion.usage
    return {
        "text": msg,
        "tokens": usage.completion_tokens if usage else None,
        "time": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "response_a" not in st.session_state:
        st.session_state.response_a = None
    if "response_b" not in st.session_state:
        st.session_state.response_b = None
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "generation_done" not in st.session_state:
        st.session_state.generation_done = False
    if "preference_submitted" not in st.session_state:
        st.session_state.preference_submitted = False
    if "meta" not in st.session_state:
        st.session_state.meta = {}


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Preference Collector", layout="wide")
    init_session()

    st.title("🗳️ Human Preference Data Collector")
    st.caption("Generate two responses from the same LLM, pick the better one, and build an RLHF-ready dataset.")

    # ---- Sidebar: settings & data export ----
    with st.sidebar:
        st.header("⚙️ Settings")

        api_key = st.text_input("OpenAI API Key", type="password",
                                help="Your key is never stored; it lives only in this session.")

        model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.05,
                                help="Higher values → more diverse (and divergent) response pairs.")
        max_tokens = st.slider("Max tokens per response", 64, 1024, 512, 64)

        st.divider()
        st.header("📊 Dataset")
        count = get_record_count()
        st.metric("Recorded preferences", count)

        if count > 0:
            # --- Export: training pairs (prompt / chosen / rejected) ---
            pairs = get_training_pairs()
            pairs_json = json.dumps(pairs, indent=2)
            st.download_button(
                "⬇️ Download training pairs (JSON)",
                data=pairs_json,
                file_name="training_pairs.json",
                mime="application/json",
            )

            # --- Export: full records as CSV ---
            all_records = get_all_preferences()
            df = pd.DataFrame(all_records)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download full log (CSV)",
                data=csv,
                file_name="preference_log.csv",
                mime="text/csv",
            )

        st.divider()
        st.header("📚 Prompt Bank")
        st.caption("Click a prompt below to load it into the editor.")

    # ---- Sidebar: prompt bank buttons ----
    prompts_dataset = load_prompt_dataset()
    with st.sidebar:
        for p in prompts_dataset:
            label = f"[{p['category']}] {p['prompt'][:80]}..."
            if st.button(label, key=f"prompt_{p['id']}"):
                st.session_state.current_prompt = p["prompt"]
                st.session_state.prompt_category = p["category"]
                # Reset generation state when a new prompt is selected
                st.session_state.generation_done = False
                st.session_state.preference_submitted = False
                st.session_state.response_a = None
                st.session_state.response_b = None
                st.rerun()

    # ---- Main area ----
    prompt_text = st.text_area(
        "Enter your prompt",
        value=st.session_state.current_prompt,
        height=120,
        key="prompt_input",
    )

    col_gen, col_reset = st.columns([1, 1])

    with col_gen:
        generate_clicked = st.button("🚀 Generate Two Responses", type="primary",
                                     disabled=st.session_state.generation_done)

    with col_reset:
        if st.button("🔄 New Prompt"):
            st.session_state.generation_done = False
            st.session_state.preference_submitted = False
            st.session_state.response_a = None
            st.session_state.response_b = None
            st.session_state.current_prompt = ""
            st.rerun()

    # ---- Generation logic ----
    if generate_clicked and not st.session_state.generation_done:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            st.stop()
        if not prompt_text.strip():
            st.warning("Please enter a prompt first.")
            st.stop()

        client = openai.OpenAI(api_key=api_key)

        with st.spinner("Generating Response A …"):
            res_a = generate_response(client, model, prompt_text, temperature, max_tokens)
        with st.spinner("Generating Response B …"):
            res_b = generate_response(client, model, prompt_text, temperature, max_tokens)

        # Lock responses into session state so they are fixed
        st.session_state.response_a = res_a
        st.session_state.response_b = res_b
        st.session_state.current_prompt = prompt_text
        st.session_state.generation_done = True
        st.session_state.preference_submitted = False
        st.session_state.meta = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        st.rerun()

    # ---- Display responses side-by-side ----
    if st.session_state.generation_done:
        st.divider()
        col_a, col_b = st.columns(2)

        res_a = st.session_state.response_a
        res_b = st.session_state.response_b

        with col_a:
            st.subheader("Response A")
            st.markdown(res_a["text"])
            st.caption(f"Tokens: {res_a['tokens']}  |  Time: {res_a['time']}s")

        with col_b:
            st.subheader("Response B")
            st.markdown(res_b["text"])
            st.caption(f"Tokens: {res_b['tokens']}  |  Time: {res_b['time']}s")

        # ---- Preference selection ----
        st.divider()

        if st.session_state.preference_submitted:
            st.success("✅ Preference recorded! Click **New Prompt** to continue.")
        else:
            st.subheader("Which response do you prefer?")
            pref_cols = st.columns(3)

            with pref_cols[0]:
                if st.button("👈  Prefer A", use_container_width=True):
                    _record_preference("A")
            with pref_cols[1]:
                if st.button("🤝  Tie", use_container_width=True):
                    _record_preference("tie")
            with pref_cols[2]:
                if st.button("👉  Prefer B", use_container_width=True):
                    _record_preference("B")

    # ---- Footer ----
    st.divider()
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        **Purpose:** Collect human preference data for alignment research (RLHF / DPO).

        **How it works:**
        1. Enter a prompt (or pick one from the curated prompt bank in the sidebar).
        2. Two independent responses are generated from the *same* model with the *same* settings.
        3. You choose which response is better, or mark a tie.
        4. The result is saved to **Supabase** (cloud Postgres) and can be exported as
           `{prompt, chosen, rejected}` JSON pairs or as a full CSV log.

        **Data format (training pairs):**
        ```json
        {
            "prompt": "...",
            "chosen": "...",
            "rejected": "...",
            "model_name": "gpt-4o-mini",
            "timestamp": "...",
            "preference": "A"
        }
        ```
        """)


def _record_preference(preference: str):
    """Build the record and persist it to the database."""
    res_a = st.session_state.response_a
    res_b = st.session_state.response_b

    if preference == "A":
        chosen, rejected = res_a["text"], res_b["text"]
    elif preference == "B":
        chosen, rejected = res_b["text"], res_a["text"]
    else:  # tie
        chosen, rejected = res_a["text"], res_b["text"]

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": st.session_state.current_prompt,
        "response_a": res_a["text"],
        "response_b": res_b["text"],
        "preference": preference,
        "chosen": chosen,
        "rejected": rejected,
        "model_name": st.session_state.meta["model"],
        "temperature": st.session_state.meta["temperature"],
        "max_tokens": st.session_state.meta["max_tokens"],
        "response_a_tokens": res_a["tokens"],
        "response_b_tokens": res_b["tokens"],
        "generation_time_a": res_a["time"],
        "generation_time_b": res_b["time"],
        "prompt_category": getattr(st.session_state, "prompt_category", None),
        "session_id": st.session_state.session_id,
    }

    save_preference(record)
    st.session_state.preference_submitted = True
    st.rerun()


if __name__ == "__main__":
    main()
