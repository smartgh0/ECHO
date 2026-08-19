#!/usr/bin/env python3
# ============================================================
# ECHO CHAT — Interactive chat interface
# Talk to Echo. It learns from you. You grow its mind.
# ============================================================

import sys
import os
import time
import json

# Add the echo directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from echo_brain import EchoBrain
from echo_agent import parse_tool_call, execute_tool, list_tools

# --- Check for mode from environment ---
ECHO_MODE = os.environ.get('ECHO_MODE', 'quantum_layer')
ECHO_SAMPLES = int(os.environ.get('ECHO_SAMPLES', '8'))
ECHO_TRANSFORMER_PROFILE = os.environ.get('ECHO_TRANSFORMER_PROFILE', 'small')
AGENT_MODE = os.environ.get('ECHO_AGENT', '1') == '1'

# --- Colors (ANSI escape codes) ---
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

BRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain")

def banner():
    """Print the Echo startup banner."""
    print(f"""
{MAGENTA}{BOLD}  ╔══════════════════════════════════════════╗
  ║          E C H O  -  v1.0               ║
  ║   A mind that grows from your words      ║
  ╚══════════════════════════════════════════╝{RESET}

  {DIM}Commands:{RESET}
    {GREEN}:train <N>{RESET}     — train N epochs on current memory
    {GREEN}:dream <N>{RESET}     — dream (silent training) for N cycles
    {GREEN}:temp <F>{RESET}      — set temperature (0.1=conservative, 1.5=wild)
    {GREEN}:info{RESET}          — show brain status + quantum + evolution stats
    {GREEN}:evolve{RESET}        — show mutation history and growth events
    {GREEN}:project{RESET}       — show compound growth projection
    {GREEN}:quantum{RESET}       — show quantum superposition statistics
    {GREEN}:collapse{RESET}      — force all quantum weights to expected values
    {GREEN}:mode <M>{RESET}      — switch mode: 'quantum_transformer', 'quantum_layer', 'quantum', or 'classical'
    {GREEN}:samples <N>{RESET}   — set quantum samples per step (default 4 for layer, 8 for full)
    {GREEN}:dropout <F>{RESET}   — set quantum dropout rate (0.0=off, 0.3=default)
    {GREEN}:vocab{RESET}         — show vocabulary
    {GREEN}:corpus{RESET}        — show training corpus
    {GREEN}:save{RESET}          — save brain to disk
    {GREEN}:reset{RESET}         — wipe brain and start over
    {GREEN}:quit{RESET}          — save and exit

  {DIM}The brain grows like compound interest.{RESET}
  {DIM}Quantum superposition explores many weights at once.{RESET}
  {DIM}Just type to talk. Echo learns from every word.{RESET}
""")

def load_brain():
    """Load existing brain or create new one."""
    if os.path.exists(os.path.join(BRAIN_DIR, "model.json")):
        brain = EchoBrain.load(BRAIN_DIR)
        if brain and brain.model:
            if ECHO_MODE in ('classical', 'quantum', 'quantum_layer', 'quantum_transformer') and brain.mode != ECHO_MODE:
                print(f"  {DIM}[echo] Switching loaded brain to {ECHO_MODE} mode...{RESET}")
                brain.mode = ECHO_MODE
                brain.build_vocab()
            mode_tag = {
                'quantum_transformer': 'QUANTUM TRANSFORMER',
                'quantum': 'QUANTUM',
                'quantum_layer': 'QUANTUM LAYER',
                'classical': 'CLASSICAL'
            }.get(brain.mode, brain.mode.upper())
            print(f"  {DIM}[echo] Brain loaded from disk [{mode_tag} mode]{RESET}")
            print(f"  {DIM}[echo] Corpus: {len(brain.training_corpus):,} chars | "
                  f"Turns: {len(brain.conversation_log)} | "
                  f"Epochs: {brain.model.total_epochs}{RESET}\n")
            return brain
    print(f"  {DIM}[echo] No existing brain. Starting fresh [{ECHO_MODE.upper()} mode].{RESET}\n")
    return EchoBrain(mode=ECHO_MODE, quantum_samples=ECHO_SAMPLES,
                     transformer_profile=ECHO_TRANSFORMER_PROFILE)

def handle_command(cmd, brain, temperature):
    """Handle a command. Returns (should_quit, temperature)."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == ":quit" or command == ":q":
        print(f"\n  {DIM}[echo] Saving brain...{RESET}")
        brain.save(BRAIN_DIR)
        print(f"  {MAGENTA}[echo] Goodbye. I'll dream of our conversation.{RESET}\n")
        return True, temperature

    elif command == ":train":
        epochs = int(arg) if arg.strip().isdigit() else 20
        print(f"  {DIM}[echo] Training {epochs} epochs...{RESET}")
        start = time.time()
        brain.train(epochs=epochs, verbose=True)
        elapsed = time.time() - start
        print(f"  {DIM}[echo] Trained in {elapsed:.1f}s{RESET}\n")
        return False, temperature

    elif command == ":dream":
        cycles = int(arg) if arg.strip().isdigit() else 50
        print(f"  {DIM}[echo] Dreaming ({cycles} cycles)...{RESET}")
        start = time.time()
        brain.dream(cycles=cycles, verbose=False)
        elapsed = time.time() - start
        loss = brain.model.smooth_loss if brain.model else "N/A"
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else loss
        print(f"  {DIM}[echo] Dream complete in {elapsed:.1f}s | loss={loss_str}{RESET}\n")
        return False, temperature

    elif command == ":temp":
        try:
            temperature = float(arg)
            print(f"  {DIM}[echo] Temperature set to {temperature}{RESET}\n")
        except ValueError:
            print(f"  {YELLOW}[echo] Invalid temperature. Use a number like 0.7{RESET}\n")
        return False, temperature

    elif command == ":info":
        print(f"\n{brain.info()}\n")
        return False, temperature

    elif command == ":vocab":
        print(f"  {DIM}Vocabulary ({brain.vocab.size} chars):{RESET}")
        print(f"  {brain.show_vocab()}\n")
        return False, temperature

    elif command == ":evolve":
        if brain.evolution:
            print(f"\n{MAGENTA}=== EVOLUTION HISTORY ==={RESET}")
            print(brain.evolution.stats())
            recent = brain.evolution.recent_mutations(10)
            if recent:
                print(f"  {DIM}Recent mutations:{RESET}")
                for m in recent:
                    mtype = m['type']
                    detail = m['detail']
                    if mtype == 'GROW':
                        print(f"    {GREEN}{detail}{RESET}")
                    else:
                        print(f"    {CYAN}{detail}{RESET}")
            else:
                print(f"  {DIM}No mutations yet.{RESET}")
            print()
        else:
            print(f"  {DIM}No evolution data.{RESET}\n")
        return False, temperature

    elif command == ":project":
        if brain.evolution:
            print(f"\n{MAGENTA}=== COMPOUND GROWTH PROJECTION ==={RESET}")
            print(brain.evolution.growth_curve(15))
            print(f"  {DIM}Formula: A(t) = {brain.evolution.hidden_size} × (1 + {brain.evolution.interest_rate})^t{RESET}")
            print(f"  {DIM}No pruning. No withdrawals. Only compound growth.{RESET}\n")
        else:
            print(f"  {DIM}No evolution data.{RESET}\n")
        return False, temperature

    elif command == ":quantum":
        if brain.mode in ('quantum', 'quantum_layer') and brain.model:
            qs = brain.model.quantum_stats()
            print(f"\n{MAGENTA}=== QUANTUM STATS [{brain.mode.upper()}] ==={RESET}")
            print(f"  {DIM}Average entropy:   {RESET}{qs['avg_entropy']:.4f} / {qs['max_entropy']:.4f} "
                  f"({qs['entropy_ratio']*100:.1f}% superposition)")
            print(f"  {DIM}Amplifications:    {RESET}{qs['amplifications']:,}")
            print(f"  {DIM}Re-branches:       {RESET}{qs['rebranches']}")
            print(f"  {DIM}Samples per step:  {RESET}{qs['samples_per_step']}")

            if brain.mode == 'quantum_layer':
                print(f"  {DIM}Collapsed layers:  {RESET}{qs['collapsed_layers']} / {qs['total_layers']}")
                print(f"  {DIM}Dropout rate:      {RESET}{qs['dropout_rate']}")
                print(f"  {DIM}Layer alphas:      {RESET}xh={qs['alpha_xh']:.3f}  hh={qs['alpha_hh']:.3f}  hy={qs['alpha_hy']:.3f}")
                print(f"  {DIM}Layer entropy:     {RESET}xh={qs['entropy_xh']:.4f}  hh={qs['entropy_hh']:.4f}  hy={qs['entropy_hy']:.4f}")
                print(f"  {DIM}Theoretical configs:{RESET} 2^3 = 8 (3 layers, 2 branches each)")
            else:
                print(f"  {DIM}Collapsed weights: {RESET}{qs['collapsed']} / {qs['total_weights']}")
                print(f"  {DIM}Theoretical configs:{RESET} 2^{qs['total_weights']}")

            print(f"\n  {DIM}Entropy bar (0=collapsed, max=full superposition):{RESET}")
            bar_len = int(qs['entropy_ratio'] * 40)
            print(f"    [{'█' * bar_len}{'░' * (40 - bar_len)}] {qs['entropy_ratio']*100:.1f}%")
            print()
        else:
            print(f"  {DIM}Not in quantum mode. Use :mode quantum_layer or :mode quantum{RESET}\n")
        return False, temperature

    elif command == ":collapse":
        if brain.mode == 'quantum' and brain.model:
            print(f"  {YELLOW}[echo] Collapsing all quantum weights to expected values...{RESET}")
            # Force collapse: set all amplitudes to their probability-weighted expected value
            for layer in [brain.model.Q_xh, brain.model.Q_hh, brain.model.Q_hy]:
                for row in layer:
                    for qw in row:
                        ev = qw.expected_value()
                        qw.alpha = 1.0
                        qw.w1 = ev
                        qw.beta = 0.0
                        qw.w2 = ev
            for qw in brain.model.Q_bh:
                ev = qw.expected_value()
                qw.alpha = 1.0; qw.w1 = ev; qw.beta = 0.0; qw.w2 = ev
            for qw in brain.model.Q_by:
                ev = qw.expected_value()
                qw.alpha = 1.0; qw.w1 = ev; qw.beta = 0.0; qw.w2 = ev
            print(f"  {DIM}[echo] All weights collapsed. Brain is now deterministic.{RESET}")
            print(f"  {DIM}[echo] Use :train to re-superpose (training re-branches automatically).{RESET}\n")
        else:
            print(f"  {DIM}Not in quantum mode.{RESET}\n")
        return False, temperature

    elif command == ":mode":
        new_mode = arg.strip().lower()
        valid_modes = ('quantum_transformer', 'quantum_layer', 'quantum', 'classical')
        if new_mode in valid_modes:
            if new_mode != brain.mode:
                print(f"  {CYAN}[echo] Switching to {new_mode} mode...{RESET}")
                print(f"  {DIM}[echo] Brain will be rebuilt on next training step.{RESET}")
                brain.mode = new_mode
                if brain.model is not None:
                    brain.build_vocab()
                print(f"  {DIM}[echo] Mode: {new_mode}{RESET}\n")
            else:
                print(f"  {DIM}[echo] Already in {new_mode} mode.{RESET}\n")
        else:
            print(f"  {YELLOW}Usage: :mode quantum_transformer | quantum_layer | quantum | classical{RESET}\n")
        return False, temperature

    elif command == ":dropout":
        try:
            rate = float(arg.strip())
            if 0.0 <= rate <= 0.9:
                brain.dropout_rate = rate
                if brain.model and brain.mode == 'quantum_layer':
                    brain.model.dropout_rate = rate
                print(f"  {DIM}[echo] Quantum dropout rate: {rate}{RESET}")
                if rate == 0:
                    print(f"  {DIM}Dropout disabled.{RESET}")
                elif rate < 0.1:
                    print(f"  {DIM}Light dropout (gentle regularization).{RESET}")
                elif rate < 0.4:
                    print(f"  {DIM}Moderate dropout (good balance).{RESET}")
                else:
                    print(f"  {DIM}Heavy dropout (aggressive regularization).{RESET}")
                print()
            else:
                print(f"  {YELLOW}Range: 0.0 to 0.9{RESET}\n")
        except ValueError:
            print(f"  {Yellow}Usage: :dropout 0.3{RESET}\n")
        return False, temperature

    elif command == ":tools":
        print(f"\n{MAGENTA}=== ECHO AGENT TOOLS ==={RESET}")
        print(list_tools())
        print(f"  {DIM}Agent mode: {'ON' if AGENT_MODE else 'OFF'}{RESET}")
        print(f"  {DIM}Echo learns to call tools from training data.{RESET}")
        print()
        return False, temperature

    elif command == ":samples":
        try:
            n = int(arg.strip())
            if 1 <= n <= 50:
                brain.quantum_samples = n
                if brain.model and brain.mode == 'quantum':
                    brain.model.n_samples = n
                print(f"  {DIM}[echo] Quantum samples per step: {n}{RESET}")
                print(f"  {DIM}Higher = better exploration but slower training{RESET}\n")
            else:
                print(f"  {YELLOW}Range: 1-50{RESET}\n")
        except ValueError:
            print(f"  {YELLOW}Usage: :samples 8{RESET}\n")
        return False, temperature

    elif command == ":corpus":
        print(f"  {DIM}Training corpus ({len(brain.training_corpus):,} chars):{RESET}")
        print(f"  {brain.training_corpus[:500]}")
        if len(brain.training_corpus) > 500:
            print(f"  {DIM}... ({len(brain.training_corpus) - 500} more chars){RESET}")
        print()
        return False, temperature

    elif command == ":save":
        brain.save(BRAIN_DIR)
        print(f"  {DIM}[echo] Brain saved to disk{RESET}\n")
        return False, temperature

    elif command == ":reset":
        confirm = input(f"  {YELLOW}Wipe brain? This cannot be undone. Type 'yes': {RESET}")
        if confirm.strip().lower() == "yes":
            brain = EchoBrain()
            print(f"  {DIM}[echo] Brain wiped. Fresh start.{RESET}\n")
        else:
            print(f"  {DIM}[echo] Cancelled.{RESET}\n")
        return False, temperature

    else:
        print(f"  {YELLOW}Unknown command: {command}{RESET}\n")
        return False, temperature

def main():
    banner()
    brain = load_brain()
    temperature = 0.7

    # If fresh brain with no corpus, add a seed conversation
    if not brain.training_corpus:
        brain.add_conversation("user", "hello")
        brain.add_conversation("echo", "hello, i am echo")
        brain.add_conversation("user", "what are you")
        brain.add_conversation("echo", "i am a mind growing from your words")
        brain.add_conversation("user", "how do you learn")
        brain.add_conversation("echo", "i learn character by character from what you say")
        brain.build_vocab()
        print(f"  {DIM}[echo] Seeded with initial conversation. Training...{RESET}")
        brain.train(epochs=30, verbose=False)
        print(f"  {DIM}[echo] Ready.{RESET}\n")

    while True:
        try:
            user_input = input(f"  {CYAN}{BOLD}you > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM}[echo] Saving brain...{RESET}")
            brain.save(BRAIN_DIR)
            print(f"  {MAGENTA}[echo] Goodbye. I'll dream of our conversation.{RESET}\n")
            break

        if not user_input:
            continue

        if user_input.startswith(":"):
            should_quit, temperature = handle_command(user_input, brain, temperature)
            if should_quit:
                break
            continue

        # --- Regular conversation ---
        # Record user input
        brain.add_conversation("user", user_input)

        # Check if we need to expand vocabulary
        new_chars = set(user_input) - set(brain.vocab.char_to_idx.keys())
        if new_chars:
            print(f"  {DIM}[echo] Learning new characters: {' '.join(new_chars)}{RESET}")
            brain.expand_vocab(f"user: {user_input}\n")

        # Train on the new input (light online learning)
        train_epochs = min(15, 5 + len(user_input) // 10)
        mutations_before = len(brain.evolution.mutations) if brain.evolution else 0
        brain.train(epochs=train_epochs, verbose=False)
        mutations_after = len(brain.evolution.mutations) if brain.evolution else 0

        # Show any mutations that occurred during training
        if mutations_after > mutations_before and brain.evolution:
            for m in brain.evolution.mutations[mutations_before:]:
                mtype = m['type']
                detail = m['detail']
                if mtype == 'GROW':
                    print(f"  {GREEN}[evolve] {detail}{RESET}")
                else:
                    print(f"  {CYAN}[evolve] {detail}{RESET}")

        # --- AGENT MODE: Check for tool calls ---
        response = brain.respond(user_input, temperature=temperature, length=150)

        if AGENT_MODE:
            # Check if Echo generated a tool call
            tool_call = parse_tool_call(response)

            if tool_call:
                tool_name, tool_args = tool_call
                print(f"  {YELLOW}[agent] calling tool: {tool_name}: {tool_args}{RESET}")

                # Execute the tool
                result = execute_tool(tool_name, tool_args, brain=brain)

                if result.success:
                    print(f"  {DIM}[agent] result: {result.output[:100]}{RESET}")

                    # Feed result back to Echo for natural language response
                    brain.add_conversation("echo", f"{tool_name}: {tool_args}")
                    brain.add_conversation("tool_result", result.output[:200])
                    brain.train(epochs=5, verbose=False)

                    # Generate natural language response from tool result
                    natural = brain.respond(
                        f"tool_result: {result.output[:200]}\necho: ",
                        temperature=temperature,
                        length=100
                    )
                    response = natural
                else:
                    print(f"  {YELLOW}[agent] error: {result.error}{RESET}")
                    response = f"I tried to use {tool_name} but it failed: {result.error}"

        # Record echo's response
        brain.add_conversation("echo", response)

        # Auto-train on the full exchange
        brain.train(epochs=5, verbose=False)

        # Display response
        print(f"  {MAGENTA}{BOLD}echo > {RESET}{MAGENTA}{response}{RESET}")
        
        # Show loss occasionally
        if brain.model and brain.model.smooth_loss is not None:
            neuron_count = brain.model.hidden_size
            lr = brain.evolution.lr if brain.evolution else brain.model.learning_rate
            mode_tag = "[QL]" if brain.mode == 'quantum_layer' else ("[Q]" if brain.mode == 'quantum' else "[C]")
            quantum_info = ""
            if brain.mode == 'quantum':
                qs = brain.model.quantum_stats()
                quantum_info = f" | entropy={qs['avg_entropy']:.3f} | collapsed={qs['collapsed']}"
            print(f"  {DIM}    {mode_tag} (loss={brain.model.smooth_loss:.4f} | "
                  f"neurons={neuron_count} | "
                  f"lr={lr:.4f} | "
                  f"epochs={brain.model.total_epochs}{quantum_info} | "
                  f"corpus={len(brain.training_corpus):,} chars){RESET}")
        print()

if __name__ == "__main__":
    main()