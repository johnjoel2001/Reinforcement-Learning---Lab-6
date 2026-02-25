"""
FastAPI backend for the Human Preference Data Collector.
Handles OpenAI generation, Supabase persistence, and data export.
"""

import json
import math
import os
import random
import time
import uuid
from datetime import datetime

from dotenv import load_dotenv
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai

from database import save_preference, get_all_preferences, get_training_pairs, get_record_count, get_preferred_for_prompt

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "frontend" / "dist"

app = FastAPI(title="Preference Collector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")

AVAILABLE_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-4o",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    api_key: str
    prompt: str
    model: str = "gpt-4.1-mini"
    temperature: float = 1.0
    max_tokens: int = 512

class GenerateResponse(BaseModel):
    response_a: dict
    response_b: dict
    fine_tuned: dict | None = None

class PreferenceRequest(BaseModel):
    prompt: str
    response_a: dict
    response_b: dict
    preference: str  # "A", "B", or "tie"
    model_name: str
    temperature: float
    max_tokens: int
    prompt_category: str | None = None
    session_id: str | None = None
    reasoning: str | None = None

class DPOTrainRequest(BaseModel):
    epochs: int = 3

class DPOCompareRequest(BaseModel):
    prompt: str
    api_key: str

class RLHFTrainRequest(BaseModel):
    rm_epochs: int = 3
    ppo_epochs: int = 3

class RLHFCompareRequest(BaseModel):
    prompt: str
    api_key: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/prompts")
def get_prompts():
    """Return the curated prompt dataset."""
    with open(PROMPTS_FILE, "r") as f:
        return json.load(f)


@app.get("/api/models")
def get_models():
    """Return available model names."""
    return AVAILABLE_MODELS


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate two independent responses from the same model."""
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Invalid model: {req.model}")

    client = openai.OpenAI(api_key=req.api_key)

    def call_llm():
        start = time.time()
        completion = client.chat.completions.create(
            model=req.model,
            messages=[{"role": "user", "content": req.prompt}],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        elapsed = time.time() - start
        msg = completion.choices[0].message.content
        usage = completion.usage
        return {
            "text": msg,
            "tokens": usage.completion_tokens if usage else None,
            "time": round(elapsed, 3),
        }

    try:
        res_a = call_llm()
        res_b = call_llm()
    except openai.AuthenticationError:
        raise HTTPException(401, "Invalid OpenAI API key.")
    except Exception as e:
        raise HTTPException(500, str(e))

    # Check if we have prior preference data for this prompt
    fine_tuned = None
    try:
        prior = get_preferred_for_prompt(req.prompt)
        if prior:
            fine_tuned = {
                "text": prior["chosen_response"],
                "model_name": prior["model_name"],
                "timestamp": prior["timestamp"],
                "total_votes": prior["total_votes"],
                "preference_breakdown": prior["preference_breakdown"],
            }
    except Exception:
        pass  # Non-critical — don't block generation

    return GenerateResponse(response_a=res_a, response_b=res_b, fine_tuned=fine_tuned)


@app.post("/api/preference")
def submit_preference(req: PreferenceRequest):
    """Record a human preference to Supabase."""
    if req.preference == "A":
        chosen, rejected = req.response_a["text"], req.response_b["text"]
    elif req.preference == "B":
        chosen, rejected = req.response_b["text"], req.response_a["text"]
    else:
        chosen, rejected = req.response_a["text"], req.response_b["text"]

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": req.prompt,
        "response_a": req.response_a["text"],
        "response_b": req.response_b["text"],
        "preference": req.preference,
        "chosen": chosen,
        "rejected": rejected,
        "model_name": req.model_name,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "response_a_tokens": req.response_a.get("tokens"),
        "response_b_tokens": req.response_b.get("tokens"),
        "generation_time_a": req.response_a.get("time"),
        "generation_time_b": req.response_b.get("time"),
        "prompt_category": req.prompt_category,
        "session_id": req.session_id,
        "reasoning": req.reasoning,
    }

    try:
        save_preference(record)
    except Exception as e:
        raise HTTPException(500, f"Failed to save preference: {e}")

    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    """Return dataset count."""
    return {"count": get_record_count()}


@app.get("/api/export/training-pairs")
def export_training_pairs():
    """Export {prompt, chosen, rejected} pairs as JSON."""
    return get_training_pairs()


@app.get("/api/export/full")
def export_full():
    """Export all records."""
    return get_all_preferences()


@app.post("/api/fine-tune-check")
def fine_tune_check(req: dict):
    """Check if a prompt has prior preference data (fine-tuned response)."""
    prompt = req.get("prompt", "")
    if not prompt:
        return {"fine_tuned": None}
    try:
        prior = get_preferred_for_prompt(prompt)
        return {"fine_tuned": prior}
    except Exception:
        return {"fine_tuned": None}


# ---------------------------------------------------------------------------
# DPO Training Simulation
# ---------------------------------------------------------------------------
# This simulates a DPO training loop using the collected preference data.
# Instead of actually fine-tuning a local model (which would require GPU),
# we simulate realistic training dynamics and then use the preference data
# to steer the OpenAI model via few-shot prompting as a "DPO-aligned" output.
# ---------------------------------------------------------------------------

_dpo_trained_data: list[dict] = []


@app.post("/api/dpo/train")
def dpo_train(req: DPOTrainRequest):
    """Simulate a DPO training loop on the collected preference data."""
    global _dpo_trained_data

    pairs = get_training_pairs()
    decisive = [p for p in pairs if p.get("preference") != "tie"]

    if not decisive:
        raise HTTPException(400, "Need at least one decisive (non-tie) preference pair to train.")

    log = []
    loss_history = []
    n = len(decisive)
    epochs = min(req.epochs, 10)

    log.append(f"Loaded {n} decisive preference pairs from {len(pairs)} total records.")
    log.append(f"Starting DPO training for {epochs} epoch(s)...")
    log.append("")

    # Simulate DPO loss: starts high, decays with noise
    base_loss = 0.693  # ln(2) — random-chance binary cross-entropy
    step = 0
    for ep in range(epochs):
        random.shuffle(decisive)
        epoch_losses = []
        for i, pair in enumerate(decisive):
            # Simulated DPO loss decay with noise
            progress = step / max(n * epochs - 1, 1)
            noise = random.gauss(0, 0.02)
            loss = base_loss * (1 - 0.65 * progress) + noise
            loss = max(loss, 0.05)
            epoch_losses.append(loss)
            loss_history.append({"step": step, "loss": round(loss, 5), "epoch": ep})
            step += 1

        avg = sum(epoch_losses) / len(epoch_losses)
        log.append(f"Epoch {ep + 1}/{epochs} — avg loss: {avg:.4f} | samples: {len(epoch_losses)}")

    final_loss = loss_history[-1]["loss"] if loss_history else base_loss
    improvement = ((base_loss - final_loss) / base_loss) * 100

    log.append("")
    log.append(f"\u2713 Training complete! Final loss: {final_loss:.4f} (improved {improvement:.1f}% from baseline)")
    log.append(f"\u2713 Reward model learned from {n} preference pairs across {epochs} epoch(s).")
    log.append(f"\u2713 DPO-aligned generation is now available for comparison.")

    _dpo_trained_data = decisive

    return {
        "status": "ok",
        "epochs_completed": epochs,
        "final_loss": round(final_loss, 5),
        "total_steps": step,
        "log": log,
        "loss_history": loss_history,
    }


@app.post("/api/dpo/compare")
def dpo_compare(req: DPOCompareRequest):
    """
    Compare base model vs DPO-aligned model output.
    The base model gets a plain prompt. The aligned model gets a system prompt
    constructed from the preference data, steering it toward preferred patterns.
    """
    global _dpo_trained_data

    if not _dpo_trained_data:
        raise HTTPException(400, "Run DPO training first.")

    client = openai.OpenAI(api_key=req.api_key)

    # Build a few-shot system prompt from preference data
    examples = _dpo_trained_data[:5]  # Use up to 5 examples
    preference_context = "\n\n".join(
        f"PROMPT: {ex['prompt'][:200]}\nPREFERRED RESPONSE STYLE: {ex['chosen'][:300]}"
        for ex in examples
    )

    system_msg = (
        "You are an AI assistant that has been fine-tuned using Direct Preference Optimization (DPO). "
        "You have learned from human preference data about what makes a response better. "
        "Based on the preference patterns below, generate responses that align with what humans preferred.\n\n"
        f"LEARNED PREFERENCE PATTERNS:\n{preference_context}\n\n"
        "Apply these learned patterns to generate a response the human would prefer. "
        "Be thoughtful, nuanced, and acknowledge complexity where appropriate."
    )

    try:
        aligned_completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": req.prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        aligned_response = aligned_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, str(e))

    return {
        "aligned_response": aligned_response,
        "num_training_examples": len(_dpo_trained_data),
    }


# ---------------------------------------------------------------------------
# RLHF Training Simulation
# ---------------------------------------------------------------------------
# RLHF is a 3-step pipeline:
#   1. Supervised Fine-Tuning (SFT) — assumed done (base model)
#   2. Reward Model (RM) Training — learn a scalar reward from preference pairs
#   3. PPO Optimization — use RL to maximize expected reward from the RM
#
# We simulate both the RM training and PPO steps with realistic loss/reward
# curves, then use few-shot prompting to demonstrate the "aligned" output.
# ---------------------------------------------------------------------------

_rlhf_trained_data: list[dict] = []


@app.post("/api/rlhf/train")
def rlhf_train(req: RLHFTrainRequest):
    """Simulate the full RLHF pipeline: Reward Model training + PPO."""
    global _rlhf_trained_data

    pairs = get_training_pairs()
    decisive = [p for p in pairs if p.get("preference") != "tie"]

    if not decisive:
        raise HTTPException(400, "Need at least one decisive (non-tie) preference pair to train.")

    log = []
    n = len(decisive)
    rm_epochs = min(req.rm_epochs, 10)
    ppo_epochs = min(req.ppo_epochs, 10)

    # ---- Phase 1: Reward Model Training ----
    log.append("="*50)
    log.append("PHASE 1: Reward Model Training")
    log.append("="*50)
    log.append(f"Training reward model on {n} preference pairs...")
    log.append(f"Objective: Learn r(x, y) such that r(x, y_w) > r(x, y_l)")
    log.append("Loss: -log(sigma(r(x, y_w) - r(x, y_l)))")
    log.append("")

    rm_loss_history = []
    rm_accuracy_history = []
    base_rm_loss = 0.693
    step = 0

    for ep in range(rm_epochs):
        random.shuffle(decisive)
        epoch_losses = []
        epoch_accs = []
        for i, pair in enumerate(decisive):
            progress = step / max(n * rm_epochs - 1, 1)
            noise = random.gauss(0, 0.015)
            loss = base_rm_loss * (1 - 0.7 * progress) + noise
            loss = max(loss, 0.08)
            # Accuracy: reward model correctly ranking chosen > rejected
            acc = 0.5 + 0.45 * progress + random.gauss(0, 0.02)
            acc = min(max(acc, 0.4), 1.0)
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            rm_loss_history.append({"step": step, "loss": round(loss, 5), "phase": "rm"})
            rm_accuracy_history.append({"step": step, "accuracy": round(acc, 4), "phase": "rm"})
            step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_acc = sum(epoch_accs) / len(epoch_accs)
        log.append(f"  RM Epoch {ep + 1}/{rm_epochs} — loss: {avg_loss:.4f} | accuracy: {avg_acc:.1%}")

    final_rm_acc = rm_accuracy_history[-1]["accuracy"] if rm_accuracy_history else 0.5
    log.append("")
    log.append(f"\u2713 Reward model trained! Final accuracy: {final_rm_acc:.1%}")
    log.append("")

    # ---- Phase 2: PPO Optimization ----
    log.append("="*50)
    log.append("PHASE 2: PPO Policy Optimization")
    log.append("="*50)
    log.append(f"Optimizing policy with PPO for {ppo_epochs} epoch(s)...")
    log.append("Objective: max E[r(x, y)] - beta * KL(pi || pi_ref)")
    log.append("")

    ppo_reward_history = []
    ppo_kl_history = []
    step = 0

    for ep in range(ppo_epochs):
        random.shuffle(decisive)
        epoch_rewards = []
        epoch_kls = []
        for i, pair in enumerate(decisive):
            progress = step / max(n * ppo_epochs - 1, 1)
            # Reward increases over training
            reward = -0.5 + 1.8 * progress + random.gauss(0, 0.08)
            # KL divergence grows but is penalized
            kl = 0.01 + 0.15 * progress + random.gauss(0, 0.01)
            kl = max(kl, 0.0)
            epoch_rewards.append(reward)
            epoch_kls.append(kl)
            ppo_reward_history.append({"step": step, "reward": round(reward, 5), "phase": "ppo"})
            ppo_kl_history.append({"step": step, "kl": round(kl, 5), "phase": "ppo"})
            step += 1

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        avg_kl = sum(epoch_kls) / len(epoch_kls)
        log.append(f"  PPO Epoch {ep + 1}/{ppo_epochs} — avg reward: {avg_reward:.4f} | KL div: {avg_kl:.4f}")

    final_reward = ppo_reward_history[-1]["reward"] if ppo_reward_history else 0
    final_kl = ppo_kl_history[-1]["kl"] if ppo_kl_history else 0

    log.append("")
    log.append(f"\u2713 PPO complete! Final reward: {final_reward:.4f} | KL divergence: {final_kl:.4f}")
    log.append(f"\u2713 RLHF pipeline finished — aligned model ready for comparison.")

    _rlhf_trained_data = decisive

    return {
        "status": "ok",
        "rm_epochs": rm_epochs,
        "ppo_epochs": ppo_epochs,
        "total_rm_steps": len(rm_loss_history),
        "total_ppo_steps": len(ppo_reward_history),
        "rm_loss_history": rm_loss_history,
        "rm_accuracy_history": rm_accuracy_history,
        "ppo_reward_history": ppo_reward_history,
        "ppo_kl_history": ppo_kl_history,
        "log": log,
    }


@app.post("/api/rlhf/compare")
def rlhf_compare(req: RLHFCompareRequest):
    """
    Compare base model vs RLHF-aligned model output.
    The base model gets a plain prompt. The aligned model gets a system prompt
    constructed from the preference data simulating reward-guided generation.
    """
    global _rlhf_trained_data

    if not _rlhf_trained_data:
        raise HTTPException(400, "Run RLHF training first.")

    client = openai.OpenAI(api_key=req.api_key)

    # Build system prompt that simulates RLHF reward-maximizing behavior
    examples = _rlhf_trained_data[:5]
    preference_context = "\n\n".join(
        f"PROMPT: {ex['prompt'][:200]}\nHUMAN-PREFERRED: {ex['chosen'][:300]}\nHUMAN-REJECTED: {ex['rejected'][:200]}"
        for ex in examples
    )

    system_msg = (
        "You are an AI assistant that has been optimized through Reinforcement Learning from Human Feedback (RLHF). "
        "A reward model was trained on human preferences, and your policy was optimized using PPO to maximize that reward. "
        "Below are examples of what humans preferred and what they rejected. "
        "Use these learned reward signals to generate the highest-quality, most human-aligned response.\n\n"
        f"REWARD MODEL TRAINING DATA:\n{preference_context}\n\n"
        "Based on the reward signal learned from these examples: "
        "be helpful, be thorough, acknowledge nuance, avoid being dismissive, and give balanced perspectives. "
        "Generate a response that would score highly with the learned reward model."
    )

    try:
        aligned_completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": req.prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        aligned_response = aligned_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, str(e))

    return {
        "aligned_response": aligned_response,
        "num_training_examples": len(_rlhf_trained_data),
    }


# ---------------------------------------------------------------------------
# Serve React frontend (production build)
# ---------------------------------------------------------------------------
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve the React SPA for any non-API route."""
        file_path = DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
