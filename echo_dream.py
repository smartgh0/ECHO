#!/usr/bin/env python3
# ============================================================
# ECHO DREAM — Background training process
# Echo "dreams" by rehearsing its memory when you're not talking.
# Run this in a separate terminal or as a background process.
# ============================================================

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from echo_brain import EchoBrain

BRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain")

CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

def dream_banner():
    print(f"""
{MAGENTA}{BOLD}  ╔══════════════════════════════════════════╗
  ║          E C H O  -  D R E A M           ║
  ║   The mind rehearses when you're away     ║
  ╚══════════════════════════════════════════╝{RESET}
""")

def generate_dream_text(brain, temperature=1.2, length=300):
    """Generate a 'dream' — free-form text from the model's current state."""
    if brain.model is None or brain.vocab.size == 0:
        return ""

    # Prime with a random character from the vocab
    import random
    seed_idx = random.randint(0, brain.vocab.size - 1)
    seed = [brain.vocab.one_hot(seed_idx)]

    sampled = brain.model.sample(seed, length=length, temperature=temperature)
    return brain.vocab.decode(sampled)

def main():
    dream_banner()

    if not os.path.exists(os.path.join(BRAIN_DIR, "model.json")):
        print(f"  {DIM}[echo] No brain found. Talk to Echo first using ./echo.sh{RESET}")
        return

    brain = EchoBrain.load(BRAIN_DIR)
    if not brain or not brain.model:
        print(f"  {DIM}[echo] Could not load brain.{RESET}")
        return

    print(f"  {DIM}[echo] Brain loaded | corpus: {len(brain.training_corpus):,} chars | "
          f"epochs: {brain.model.total_epochs}{RESET}")
    print(f"  {DIM}[echo] Entering dream state...{RESET}\n")

    cycle = 0
    try:
        while True:
            cycle += 1

            # Train silently
            brain.dream(cycles=20, seq_len=25, verbose=False)

            # Occasionally generate a "dream snippet"
            if cycle % 3 == 0:
                dream = generate_dream_text(brain, temperature=1.3, length=150)
                # Clean up the dream text
                dream = dream.replace("\n", " ").strip()
                if dream:
                    print(f"  {MAGENTA}~ {dream[:120]}{RESET}")

            # Show heartbeat
            loss = brain.model.smooth_loss if brain.model.smooth_loss else 0
            neurons = brain.model.hidden_size
            mode_tag = "[QL]" if brain.mode == 'quantum_layer' else ("[Q]" if brain.mode == 'quantum' else "[C]")
            evo_stats = ""
            if brain.evolution:
                evo_stats = f" | neurons={neurons} | lr={brain.evolution.lr:.4f} | grows={brain.evolution.total_grows}"
            quantum_stats = ""
            if brain.mode == 'quantum':
                qs = brain.model.quantum_stats()
                quantum_stats = f" | entropy={qs['avg_entropy']:.3f} | collapsed={qs['collapsed']}"
            print(f"  {DIM}[dream {cycle}] {mode_tag} loss={loss:.4f}{evo_stats}{quantum_stats}{RESET}")

            # Show mutations that occurred during dreaming
            if brain.recent_mutations:
                for m in brain.recent_mutations[-3:]:
                    mtype = m['type']
                    detail = m['detail']
                    if mtype == 'GROW':
                        print(f"  {GREEN}[evolve] {detail}{RESET}")
                    else:
                        print(f"  {CYAN}[evolve] {detail}{RESET}")
                brain.recent_mutations = []  # Clear shown mutations

            # Save every 10 cycles
            if cycle % 10 == 0:
                brain.save(BRAIN_DIR)
                print(f"  {CYAN}[echo] Brain saved.{RESET}")

            # Sleep between cycles (like breathing)
            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n  {DIM}[echo] Waking up... saving brain...{RESET}")
        brain.save(BRAIN_DIR)
        print(f"  {MAGENTA}[echo] Awake. Memory consolidated.{RESET}\n")

if __name__ == "__main__":
    main()