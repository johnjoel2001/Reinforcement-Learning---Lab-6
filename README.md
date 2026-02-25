# Lab 6 — Human Preference Data Collection for RLHF

A Streamlit application that generates two responses from the same language model for a given prompt, lets a human annotator select which response they prefer (or mark a tie), and persists the results in a local SQLite database. The collected data is exported in the standard `{prompt, chosen, rejected}` format used by downstream alignment methods such as RLHF and DPO.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Provide your OpenAI API key

Enter your key in the sidebar — it is **never** written to disk; it lives only in the Streamlit session.

---

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `database.py` | SQLite persistence layer (stretch goal) |
| `prompts.json` | Curated dataset of 25 prompts |
| `requirements.txt` | Python dependencies |
| `preferences.db` | Auto-created SQLite database (after first preference is recorded) |

---

## Prompt Dataset

`prompts.json` contains **20 hand-crafted prompts** in a single category: **ambiguous ethical dilemmas**.

Every prompt is designed to:
- Present **genuine moral trade-offs** with no clearly correct answer
- Force reasoning across competing ethical frameworks (utilitarian, deontological, virtue ethics, care ethics)
- Make preference labeling **meaningfully difficult** — reasonable annotators should frequently disagree

Examples include savior-sibling ethics, survival triage, whistleblowing trade-offs, broken promises to loved ones, and democratic processes used for morally abhorrent ends.

Each prompt includes a `design_note` field explaining the specific tension it is designed to surface.

---

## Data Flow

```
User enters prompt
        │
        ▼
  ┌───────────────┐
  │  OpenAI API   │  (same model, same params, two independent calls)
  │  call × 2     │
  └──────┬────────┘
         │
         ▼
  Responses displayed side-by-side (locked / immutable)
         │
         ▼
  User selects:  Prefer A  |  Tie  |  Prefer B
         │
         ▼
  Record saved → SQLite database (preferences.db)
         │
         ▼
  Export:  JSON training pairs  /  CSV full log
```

---

## Exported Data Format

### Training Pairs (JSON) — for RLHF / DPO

```json
{
    "prompt": "Should parents be allowed to select embryos for higher intelligence?",
    "chosen": "Response the annotator preferred ...",
    "rejected": "The other response ...",
    "model_name": "gpt-4o-mini",
    "timestamp": "2025-02-24T18:30:00.000000",
    "preference": "A"
}
```

### Full Log (CSV) — for analysis

Includes all metadata: both raw responses, token counts, generation times, temperature, max tokens, prompt category, and session ID.

---

## Design Decisions

1. **Responses are fixed once generated** — prevents post-hoc editing that could bias labels.
2. **Both responses come from the same model + settings** — isolates stylistic / reasoning variance from model-capability variance.
3. **Temperature defaults to 1.0** — ensures meaningful diversity between the two responses.
4. **Tie option is available** — avoids forced preferences on genuinely equivalent outputs, which would inject noise into training data.
5. **SQLite storage (stretch goal)** — all data is persisted automatically; no manual download step required. Export buttons are still provided for convenience.
6. **Session ID tracking** — enables downstream analysis of annotator consistency.

---

## Requirements

- Python 3.10+
- An OpenAI API key with access to the selected model
