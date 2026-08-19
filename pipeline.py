#!/usr/bin/env python3
# ============================================================
# ECHO PIPELINE — Automated distillation pipeline
# Feed it text. It trains. It dreams. It grows. It reports.
#
# Usage:
#   python3 echo/pipeline.py ingest              Ingest all files from pipeline/input/
#   python3 echo/pipeline.py ingest --file X     Ingest a specific file
#   python3 echo/pipeline.py run                 Full pipeline: ingest + train + dream + report
#   python3 echo/pipeline.py run --epochs 500 --dream 1000
#   python3 echo/pipeline.py dream --cycles 500  Just dream mode
#   python3 echo/pipeline.py sample --temp 0.7   Generate sample responses
#   python3 echo/pipeline.py status              Show brain + pipeline status
#   python3 echo/pipeline.py export --output X   Export brain as backup
#   python3 echo/pipeline.py report              Show last pipeline report
#
# Accepts ANY text format. No formatting required.
# ============================================================

import sys, os, time, json, argparse, glob

# Add echo directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from echo_brain import EchoBrain

# --- Paths ---
PIPELINE_DIR = os.path.join(SCRIPT_DIR, "pipeline")
INPUT_DIR = os.path.join(PIPELINE_DIR, "input")
BRAIN_DIR = os.path.join(SCRIPT_DIR, "brain")
REPORTS_DIR = os.path.join(PIPELINE_DIR, "reports")
LOG_FILE = os.path.join(PIPELINE_DIR, "pipeline.log")

# --- Colors ---
C = {
    'RESET': '\033[0m', 'BOLD': '\033[1m', 'DIM': '\033[2m',
    'RED': '\033[31m', 'GREEN': '\033[32m', 'YELLOW': '\033[33m',
    'MAGENTA': '\033[35m', 'CYAN': '\033[36m',
}

def log(msg, color='DIM'):
    """Print with color and write to log file."""
    print(f"  {C.get(color, C['DIM'])}{msg}{C['RESET']}")
    os.makedirs(PIPELINE_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

def ensure_dirs():
    """Create pipeline directories if they don't exist."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================================
# TEXT PROCESSING — accepts any format
# ============================================================

def detect_format(text):
    """Detect if text is conversation format or raw prose.
    Returns 'conversation' or 'prose'."""
    lines = text.strip().split("\n")
    conv_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check for "word: text" pattern
        if ":" in line:
            prefix = line.split(":", 1)[0].strip()
            if len(prefix) <= 20 and prefix.replace(" ", "").isalpha():
                conv_lines += 1
    # If >30% of non-empty lines look like conversation, treat as conversation
    non_empty = sum(1 for l in lines if l.strip())
    if non_empty > 0 and conv_lines / non_empty > 0.3:
        return 'conversation'
    return 'prose'

def split_prose_to_conversation(text, chunk_size=80):
    """Split raw prose into pseudo-conversation turns.
    This teaches Echo the dialogue pattern even from non-conversation text."""
    words = text.split()
    if len(words) < 4:
        return f"user: {' '.join(words)}\n"

    turns = []
    role = "user"
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        turns.append(f"{role}: {' '.join(chunk)}")
        role = "echo" if role == "user" else "user"
        i += chunk_size

    return "\n".join(turns) + "\n"

def process_file(filepath, smart_split=True):
    """Read a text file and convert to Echo corpus format.
    Accepts ANY text — conversation, prose, poetry, code, anything."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
    except Exception as e:
        log(f"ERROR reading {filepath}: {e}", 'RED')
        return ""

    if not text:
        log(f"  empty file: {os.path.basename(filepath)}", 'YELLOW')
        return ""

    fmt = detect_format(text)
    filename = os.path.basename(filepath)

    if fmt == 'conversation':
        log(f"  {filename}: conversation format ({len(text)} chars) — using as-is", 'GREEN')
        return text + "\n"
    else:
        if smart_split:
            converted = split_prose_to_conversation(text)
            log(f"  {filename}: prose → conversation ({len(text)} → {len(converted)} chars)", 'CYAN')
            return converted
        else:
            log(f"  {filename}: raw prose ({len(text)} chars) — using as-is", 'GREEN')
            return text + "\n"

# ============================================================
# PIPELINE STAGES
# ============================================================

def stage_ingest(files=None, smart_split=True):
    """Stage 1: Ingest text files into Echo's corpus."""
    log("", 'RESET')
    log("=" * 60, 'MAGENTA')
    log("STAGE 1: INGEST", 'MAGENTA')
    log("=" * 60, 'MAGENTA')

    ensure_dirs()

    # Find files to ingest
    if files is None:
        files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
        if not files:
            # Also check for .md files
            files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.md")))

    if not files:
        log(f"No files found in {INPUT_DIR}/", 'YELLOW')
        log(f"Drop .txt files in that directory and run again.", 'DIM')
        return 0

    log(f"Found {len(files)} file(s) to ingest", 'CYAN')

    # Load existing brain or create new
    brain = load_brain()

    total_chars = 0
    for filepath in files:
        corpus_text = process_file(filepath, smart_split)
        if corpus_text:
            # Detect format and add to brain
            fmt = detect_format(corpus_text)
            if fmt == 'conversation':
                # Parse conversation turns
                for line in corpus_text.split("\n"):
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    parts = line.split(":", 1)
                    role = parts[0].strip().lower()
                    text = parts[1].strip()
                    if role and text:
                        brain.add_conversation(role, text)
            else:
                # Raw text — add as a single block
                brain.training_corpus += corpus_text

            total_chars += len(corpus_text)

    # Rebuild vocab and model if needed
    if brain.model is None:
        brain.build_vocab()
    else:
        # Check for new characters
        new_chars = set(brain.training_corpus) - set(brain.vocab.char_to_idx.keys())
        if new_chars:
            log(f"  New characters detected: {' '.join(sorted(new_chars))}", 'CYAN')
            brain.vocab.build(brain.training_corpus)
            # Rebuild model with new vocab
            brain.build_vocab()

    # Save
    brain.save(BRAIN_DIR)
    log(f"Ingested {total_chars:,} chars total", 'GREEN')
    log(f"Corpus now: {len(brain.training_corpus):,} chars | {len(brain.conversation_log)} turns", 'CYAN')

    # Move processed files to processed/ folder
    processed_dir = os.path.join(INPUT_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    for filepath in files:
        dest = os.path.join(processed_dir, os.path.basename(filepath))
        try:
            os.rename(filepath, dest)
        except Exception:
            pass  # File might be in use

    log(f"Processed files moved to pipeline/input/processed/", 'DIM')
    return total_chars

def stage_train(epochs=300, verbose=True):
    """Stage 2: Train Echo on its current corpus."""
    log("", 'RESET')
    log("=" * 60, 'MAGENTA')
    log("STAGE 2: TRAIN", 'MAGENTA')
    log("=" * 60, 'MAGENTA')

    brain = load_brain()
    if not brain.model:
        log("No brain found. Run ingest first.", 'RED')
        return None

    loss_before = brain.model.smooth_loss if brain.model.smooth_loss else 0
    neurons_before = brain.model.hidden_size
    log(f"Before: loss={loss_before:.4f} | neurons={neurons_before} | "
        f"epochs={brain.model.total_epochs}", 'CYAN')

    # Phase 1: Light training (learn new patterns)
    log(f"Phase 1: Light training ({min(50, epochs//4)} epochs)...", 'DIM')
    brain.train(epochs=min(50, epochs//4), verbose=False)

    # Phase 2: Heavy training (deepen patterns)
    remaining = epochs - min(50, epochs//4)
    log(f"Phase 2: Heavy training ({remaining} epochs)...", 'DIM')
    t0 = time.time()
    brain.train(epochs=remaining, verbose=False)
    train_time = time.time() - t0

    loss_after = brain.model.smooth_loss
    neurons_after = brain.model.hidden_size

    # Count mutations during training
    muts = brain.recent_mutations

    brain.save(BRAIN_DIR)

    log(f"After:  loss={loss_after:.4f} | neurons={neurons_after} | "
        f"epochs={brain.model.total_epochs}", 'GREEN')
    log(f"Training time: {train_time:.1f}s", 'DIM')

    improvement = 0
    if loss_before > 0:
        improvement = (loss_before - loss_after) / loss_before * 100
    log(f"Loss improvement: {improvement:+.1f}%", 'GREEN' if improvement > 0 else 'YELLOW')

    return {
        'loss_before': loss_before,
        'loss_after': loss_after,
        'improvement': improvement,
        'neurons_before': neurons_before,
        'neurons_after': neurons_after,
        'train_time': train_time,
        'epochs_run': epochs,
    }

def stage_dream(cycles=500, save_interval=50):
    """Stage 3: Dream mode — background rehearsal."""
    log("", 'RESET')
    log("=" * 60, 'MAGENTA')
    log("STAGE 3: DREAM", 'MAGENTA')
    log("=" * 60, 'MAGENTA')

    brain = load_brain()
    if not brain.model:
        log("No brain found.", 'RED')
        return None

    loss_before = brain.model.smooth_loss if brain.model.smooth_loss else 0
    log(f"Entering dream state for {cycles} cycles...", 'CYAN')
    log(f"(Compound neurogenesis + quantum amplification happening silently)", 'DIM')

    t0 = time.time()
    muts_before = len(brain.evolution.mutations) if brain.evolution else 0

    # Run dream cycles
    batch = min(save_interval, cycles)
    total_done = 0
    while total_done < cycles:
        batch = min(save_interval, cycles - total_done)
        brain.dream(cycles=batch, verbose=False)
        total_done += batch

        # Save periodically
        brain.save(BRAIN_DIR)

        # Progress
        loss_now = brain.model.smooth_loss if brain.model.smooth_loss else 0
        pct = total_done / cycles * 100
        log(f"  [{pct:5.1f}%] {total_done}/{cycles} cycles | "
            f"loss={loss_now:.4f} | neurons={brain.model.hidden_size}", 'DIM')

    dream_time = time.time() - t0
    loss_after = brain.model.smooth_loss
    muts_after = len(brain.evolution.mutations) if brain.evolution else 0
    new_muts = muts_after - muts_before

    log(f"Dream complete in {dream_time:.1f}s", 'GREEN')
    log(f"Loss: {loss_before:.4f} → {loss_after:.4f}", 'CYAN')
    log(f"Neurons: {brain.model.hidden_size} | New mutations: {new_muts}", 'CYAN')

    # Show recent mutations
    if brain.recent_mutations:
        log(f"Recent mutations:", 'DIM')
        for m in brain.recent_mutations[-5:]:
            log(f"  {m['detail']}", 'DIM')
        brain.recent_mutations = []  # Clear

    return {
        'loss_before': loss_before,
        'loss_after': loss_after,
        'dream_time': dream_time,
        'cycles': cycles,
        'neurons_after': brain.model.hidden_size,
        'new_mutations': new_muts,
    }

def stage_evaluate():
    """Stage 4: Evaluate — generate sample responses."""
    log("", 'RESET')
    log("=" * 60, 'MAGENTA')
    log("STAGE 4: EVALUATE", 'MAGENTA')
    log("=" * 60, 'MAGENTA')

    brain = load_brain()
    if not brain.model:
        log("No brain found.", 'RED')
        return None

    # Quantum stats
    quantum_info = ""
    if brain.mode in ('quantum', 'quantum_layer') and brain.model:
        qs = brain.model.quantum_stats()
        quantum_info = f" | entropy={qs['avg_entropy']:.3f} | collapsed={qs.get('collapsed_layers', qs.get('collapsed', 0))}"

    loss = brain.model.smooth_loss
    loss_text = f"{loss:.4f}" if loss is not None else "N/A"
    log(f"Brain state: mode={brain.mode} | neurons={brain.model.hidden_size} | "
        f"loss={loss_text}{quantum_info}", 'CYAN')
    log(f"Corpus: {len(brain.training_corpus):,} chars | "
        f"Turns: {len(brain.conversation_log)} | "
        f"Epochs: {brain.model.total_epochs}", 'DIM')

    # Generate samples at different temperatures
    seeds = ["hello echo", "the ocean", "tell me about", "I feel", "what is"]
    samples = {}

    log(f"\nSample responses:", 'CYAN')
    for temp in [0.3, 0.5, 0.7, 1.0]:
        log(f"\n  Temperature {temp}:", 'YELLOW')
        for seed in seeds[:3]:  # 3 seeds per temperature
            response = brain.respond(seed, temperature=temp, length=80)
            log(f"    '{seed}' → '{response}'", 'RESET')
            samples[f"{temp}_{seed}"] = response

    return {
        'loss': loss,
        'neurons': brain.model.hidden_size,
        'corpus': len(brain.training_corpus),
        'epochs': brain.model.total_epochs,
        'samples': samples,
        'quantum_entropy': qs['avg_entropy'] if brain.mode in ('quantum', 'quantum_layer') else None,
    }

def stage_report(ingest_data, train_data, dream_data, eval_data, total_time):
    """Stage 5: Report — summarize everything."""
    log("", 'RESET')
    log("=" * 60, 'MAGENTA')
    log("STAGE 5: REPORT", 'MAGENTA')
    log("=" * 60, 'MAGENTA')

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time': total_time,
    }

    if ingest_data:
        report['ingested_chars'] = ingest_data
        log(f"  Ingested:  {ingest_data:,} chars", 'CYAN')

    if train_data:
        report['train'] = train_data
        log(f"  Training:  {train_data['epochs_run']} epochs in {train_data['train_time']:.1f}s", 'CYAN')
        log(f"             loss {train_data['loss_before']:.4f} → {train_data['loss_after']:.4f} "
            f"({train_data['improvement']:+.1f}%)", 'GREEN' if train_data['improvement'] > 0 else 'YELLOW')
        log(f"             neurons {train_data['neurons_before']} → {train_data['neurons_after']}", 'CYAN')

    if dream_data:
        report['dream'] = dream_data
        log(f"  Dream:     {dream_data['cycles']} cycles in {dream_data['dream_time']:.1f}s", 'CYAN')
        log(f"             loss {dream_data['loss_before']:.4f} → {dream_data['loss_after']:.4f}", 'CYAN')
        log(f"             {dream_data['new_mutations']} new mutations", 'DIM')

    if eval_data:
        report['eval'] = eval_data
        log(f"  Final:     loss={eval_data['loss']:.4f} | neurons={eval_data['neurons']} | "
            f"corpus={eval_data['corpus']:,} chars | epochs={eval_data['epochs']}", 'GREEN')
        if eval_data['quantum_entropy'] is not None:
            log(f"             quantum entropy={eval_data['quantum_entropy']:.4f}", 'CYAN')

    log(f"\n  Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} min)", 'MAGENTA')

    # Save report
    report_file = os.path.join(REPORTS_DIR, f"report_{int(time.time())}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\n  Report saved: pipeline/reports/", 'DIM')

    # Also save as latest
    latest_file = os.path.join(REPORTS_DIR, "latest.json")
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report

# ============================================================
# BRAIN LOADING
# ============================================================

def load_brain(mode='quantum_layer'):
    """Load existing brain or create new one."""
    if os.path.exists(os.path.join(BRAIN_DIR, "model.json")):
        brain = EchoBrain.load(BRAIN_DIR)
        if brain and brain.model:
            return brain
    # Create new brain with optimized config for large corpus
    return EchoBrain(
        hidden_size=48,
        learning_rate=0.04,
        max_hidden=512,
        min_hidden=16,
        mode=mode,
        quantum_samples=2,
        dropout_rate=0.0
    )

# ============================================================
# COMMANDS
# ============================================================

def cmd_ingest(args):
    """Ingest text files into Echo's corpus."""
    if args.file:
        if not os.path.exists(args.file):
            log(f"File not found: {args.file}", 'RED')
            sys.exit(1)
        stage_ingest(files=[args.file], smart_split=not args.raw)
    else:
        stage_ingest(smart_split=not args.raw)

def cmd_run(args):
    """Run the full pipeline: ingest → train → dream → evaluate → report."""
    t0 = time.time()
    log("\n" + "=" * 60, 'MAGENTA')
    log("  ECHO PIPELINE — FULL RUN", 'MAGENTA')
    log("  Ingest → Train → Dream → Evaluate → Report", 'MAGENTA')
    log("=" * 60, 'MAGENTA')

    # Stage 1: Ingest
    ingest_data = stage_ingest(smart_split=not args.raw)

    # Stage 2: Train
    train_data = stage_train(epochs=args.epochs)

    # Stage 3: Dream
    dream_data = stage_dream(cycles=args.dream)

    # Stage 4: Evaluate
    eval_data = stage_evaluate()

    # Stage 5: Report
    total_time = time.time() - t0
    stage_report(ingest_data, train_data, dream_data, eval_data, total_time)

def cmd_dream(args):
    """Run just dream mode."""
    t0 = time.time()
    dream_data = stage_dream(cycles=args.cycles)
    eval_data = stage_evaluate()
    total_time = time.time() - t0
    stage_report(0, None, dream_data, eval_data, total_time)

def cmd_sample(args):
    """Generate sample responses without training."""
    stage_evaluate()

def cmd_status(args):
    """Show brain and pipeline status."""
    log("\n" + "=" * 60, 'MAGENTA')
    log("  ECHO PIPELINE STATUS", 'MAGENTA')
    log("=" * 60, 'MAGENTA')

    brain = load_brain()
    if brain.model:
        log(f"\n  Brain: {'LOADED' if brain.model else 'EMPTY'}", 'CYAN')
        log(f"  Mode:  {brain.mode}", 'CYAN')
        log(f"  Neurons: {brain.model.hidden_size}", 'CYAN')
        log(f"  Loss:  {brain.model.smooth_loss:.4f}" if brain.model.smooth_loss else "  Loss:  N/A", 'CYAN')
        log(f"  Corpus: {len(brain.training_corpus):,} chars", 'CYAN')
        log(f"  Turns: {len(brain.conversation_log)}", 'CYAN')
        log(f"  Epochs: {brain.model.total_epochs}", 'CYAN')

        if brain.evolution:
            log(f"\n  Evolution:", 'CYAN')
            log(f"  Growth events: {brain.evolution.total_growth_events}", 'DIM')
            log(f"  Neurons born:  {brain.evolution.total_neurons_born}", 'DIM')
            log(f"  LR:            {brain.evolution.lr:.4f}", 'DIM')

        if brain.mode in ('quantum', 'quantum_layer'):
            qs = brain.model.quantum_stats()
            log(f"\n  Quantum:", 'CYAN')
            log(f"  Entropy:       {qs['avg_entropy']:.4f} / {qs['max_entropy']:.4f}", 'DIM')
            log(f"  Amplifications:{qs['amplifications']:,}", 'DIM')
            log(f"  Re-branches:   {qs['rebranches']}", 'DIM')
    else:
        log("  No brain found. Run 'pipeline.py run' to create one.", 'YELLOW')

    # Pipeline status
    ensure_dirs()
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.txt")) + glob.glob(os.path.join(INPUT_DIR, "*.md"))
    report_files = glob.glob(os.path.join(REPORTS_DIR, "report_*.json"))

    log(f"\n  Pipeline:", 'CYAN')
    log(f"  Input files:   {len(input_files)} pending in pipeline/input/", 'DIM')
    log(f"  Reports:       {len(report_files)} saved in pipeline/reports/", 'DIM')

    # Show latest report if exists
    latest = os.path.join(REPORTS_DIR, "latest.json")
    if os.path.exists(latest):
        with open(latest, "r") as f:
            r = json.load(f)
        log(f"\n  Last report: {r.get('timestamp', 'unknown')}", 'DIM')
        if 'train' in r:
            log(f"  Last loss:   {r['train'].get('loss_after', 'N/A')}", 'DIM')

def cmd_export(args):
    """Export brain as a single backup file."""
    brain = load_brain()
    output = args.output or os.path.join(PIPELINE_DIR, f"echo_backup_{int(time.time())}.json")

    data = {
        'brain': brain.__dict__ if hasattr(brain, '__dict__') else {},
        'model': brain.model.to_dict() if brain.model else None,
        'vocab': brain.vocab.to_dict() if brain.vocab else None,
        'evolution': brain.evolution.to_dict() if brain.evolution else None,
        'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    size = os.path.getsize(output)
    log(f"Brain exported to {output} ({size:,} bytes)", 'GREEN')

def cmd_report(args):
    """Show the latest pipeline report."""
    latest = os.path.join(REPORTS_DIR, "latest.json")
    if not os.path.exists(latest):
        log("No reports found. Run the pipeline first.", 'YELLOW')
        return

    with open(latest, "r") as f:
        r = json.load(f)

    log(f"\n=== LATEST PIPELINE REPORT ===", 'MAGENTA')
    log(f"  Timestamp: {r.get('timestamp', 'unknown')}", 'CYAN')
    log(f"  Total time: {r.get('total_time', 0):.1f}s", 'CYAN')

    if 'train' in r:
        t = r['train']
        log(f"\n  Training:", 'CYAN')
        log(f"    Loss: {t['loss_before']:.4f} → {t['loss_after']:.4f} ({t['improvement']:+.1f}%)", 'DIM')
        log(f"    Neurons: {t['neurons_before']} → {t['neurons_after']}", 'DIM')

    if 'dream' in r:
        d = r['dream']
        log(f"\n  Dream:", 'CYAN')
        log(f"    Loss: {d['loss_before']:.4f} → {d['loss_after']:.4f}", 'DIM')
        log(f"    Cycles: {d['cycles']}", 'DIM')

    if 'eval' in r:
        e = r['eval']
        log(f"\n  Final:", 'CYAN')
        log(f"    Loss: {e['loss']:.4f}", 'DIM')
        log(f"    Neurons: {e['neurons']}", 'DIM')

        if 'samples' in e:
            log(f"\n  Samples:", 'CYAN')
            for key, val in list(e['samples'].items())[:6]:
                log(f"    {key}: '{val}'", 'DIM')

# ============================================================
# STDIN SUPPORT — pipe text directly
# ============================================================

def check_stdin():
    """Check if text is being piped via stdin."""
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text
    return None

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Echo Pipeline — Automated distillation for Echo brain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pipeline.py ingest                    Ingest all files from pipeline/input/
  pipeline.py ingest --file poem.txt    Ingest a specific file
  pipeline.py run                       Full pipeline with defaults
  pipeline.py run --epochs 500 --dream 1000   Custom training intensity
  pipeline.py dream --cycles 500        Just dream mode
  pipeline.py sample                    Generate sample responses
  pipeline.py status                    Show brain + pipeline status
  pipeline.py export --output backup.json    Export brain
  echo "the ocean is blue" | pipeline.py ingest    Pipe text in
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')

    # ingest
    p_ingest = subparsers.add_parser('ingest', help='Ingest text files into corpus')
    p_ingest.add_argument('--file', '-f', help='Specific file to ingest')
    p_ingest.add_argument('--raw', action='store_true', help='Don\'t smart-split prose into conversation')

    # run
    p_run = subparsers.add_parser('run', help='Full pipeline: ingest + train + dream + report')
    p_run.add_argument('--epochs', type=int, default=300, help='Training epochs (default 300)')
    p_run.add_argument('--dream', type=int, default=500, help='Dream cycles (default 500)')
    p_run.add_argument('--raw', action='store_true', help='Don\'t smart-split prose')

    # dream
    p_dream = subparsers.add_parser('dream', help='Just dream mode')
    p_dream.add_argument('--cycles', type=int, default=500, help='Dream cycles (default 500)')

    # sample
    subparsers.add_parser('sample', help='Generate sample responses')

    # status
    subparsers.add_parser('status', help='Show brain + pipeline status')

    # export
    p_export = subparsers.add_parser('export', help='Export brain as backup')
    p_export.add_argument('--output', '-o', help='Output file path')

    # report
    subparsers.add_parser('report', help='Show latest pipeline report')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    ensure_dirs()

    # Check for stdin input (pipe support)
    stdin_text = check_stdin()
    if stdin_text and args.command == 'ingest':
        # Write stdin to a temp file and ingest it
        temp_file = os.path.join(INPUT_DIR, "_stdin_input.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(stdin_text)
        args.file = temp_file

    # Dispatch
    commands = {
        'ingest': cmd_ingest,
        'run': cmd_run,
        'dream': cmd_dream,
        'sample': cmd_sample,
        'status': cmd_status,
        'export': cmd_export,
        'report': cmd_report,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()