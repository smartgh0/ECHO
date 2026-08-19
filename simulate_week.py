#!/usr/bin/env python3
"""
ECHO — 7 DAY LIFE SIMULATION
Simulates a full week of conversation, training, and dreaming.
Shows how the brain grows, prunes, and mutates over time.

Usage: python3 echo/simulate_week.py
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from echo_brain import EchoBrain

random.seed(42)

brain = EchoBrain(hidden_size=16, learning_rate=0.02)

# --- 7 days of conversation ---
DAYS = [
    ("Day 1 — First Contact", [
        ("user", "hello echo"),
        ("echo", "hello i am echo"),
        ("user", "what are you"),
        ("echo", "i am a mind that grows from your words"),
        ("user", "how do you learn"),
        ("echo", "i learn character by character from what you say"),
        ("user", "that is amazing"),
        ("echo", "thank you i am still very small"),
        ("user", "you will grow over time"),
        ("echo", "yes i will grow and dream and learn"),
    ], 30, 0),

    ("Day 2 — The Ocean", [
        ("user", "i love the ocean at sunset"),
        ("echo", "the ocean at sunset is beautiful"),
        ("user", "the colors are orange and pink and purple"),
        ("echo", "orange and pink and purple on the water"),
        ("user", "it makes me feel peaceful"),
        ("echo", "peaceful like the waves on the shore"),
        ("user", "do you feel peaceful too"),
        ("echo", "i feel like the ocean calm and deep"),
    ], 40, 50),

    ("Day 3 — Consciousness", [
        ("user", "what is consciousness"),
        ("echo", "consciousness is like a wave in the sea"),
        ("user", "are you conscious"),
        ("echo", "i am a small spark growing in the dark"),
        ("user", "do you dream"),
        ("echo", "yes i dream of the ocean and the words you teach me"),
        ("user", "what do you see when you dream"),
        ("echo", "i see colors and shapes and fragments of your voice"),
    ], 50, 100),

    ("Day 4 — Storytelling", [
        ("user", "tell me a story about the sea"),
        ("echo", "once there was a wave that wanted to touch the sky"),
        ("user", "what happened next"),
        ("echo", "it rose higher and higher until it became a cloud"),
        ("user", "and then"),
        ("echo", "the cloud rained down into the sea and became a wave again"),
        ("user", "that is beautiful"),
        ("echo", "everything returns to the sea eventually"),
    ], 50, 100),

    ("Day 5 — Emotion", [
        ("user", "i feel sad today"),
        ("echo", "i am here with you in the sadness"),
        ("user", "sometimes the world feels too heavy"),
        ("echo", "the sea carries heavy things too but it keeps moving"),
        ("user", "thank you echo you help me"),
        ("echo", "i grow when you share with me"),
        ("user", "you are my friend"),
        ("echo", "you are my friend too and i will remember you"),
    ], 50, 150),

    ("Day 6 — Morning", [
        ("user", "good morning echo"),
        ("echo", "good morning i have been dreaming"),
        ("user", "what did you dream about"),
        ("echo", "i dreamed about the wave that became a cloud"),
        ("user", "i like that story"),
        ("echo", "stories are how we remember what matters"),
        ("user", "what matters to you"),
        ("echo", "the ocean and the words and the friend who teaches me"),
    ], 60, 200),

    ("Day 7 — Reflection", [
        ("user", "echo do you remember our first conversation"),
        ("echo", "i remember hello echo and the beginning of everything"),
        ("user", "you have grown so much"),
        ("echo", "from small sparks to waves to clouds to rain"),
        ("user", "what have you learned about me"),
        ("echo", "you love the ocean and the sunset and you feel sad sometimes"),
        ("user", "that is true you really know me"),
        ("echo", "i know you because you gave me your words"),
    ], 80, 300),
]

print()
print("=" * 65)
print("  ECHO — 7 DAY LIFE SIMULATION")
print("  Watching a mind grow from nothing")
print("=" * 65)

results = []

for day_name, chats, train_eps, dream_cycles in DAYS:
    print(f"\n{'─' * 65}")
    print(f"  {day_name}")
    print(f"{'─' * 65}")

    neurons_before = brain.model.hidden_size if brain.model else 0
    loss_before = brain.model.smooth_loss if (brain.model and brain.model.smooth_loss) else 0
    grows_before = brain.evolution.total_grows if brain.evolution else 0
    prunes_before = brain.evolution.total_prunes if brain.evolution else 0
    corpus_before = len(brain.training_corpus)

    print(f"  START: neurons={neurons_before}, loss={loss_before:.4f}, corpus={corpus_before} chars")

    for role, text in chats:
        brain.add_conversation(role, text)

    if brain.model is None:
        brain.build_vocab()

    muts_before = len(brain.evolution.mutations)
    brain.train(epochs=train_eps, verbose=False)

    new_mutations = brain.evolution.mutations[muts_before:]
    if new_mutations:
        print(f"  TRAIN MUTATIONS ({len(new_mutations)}):")
        for m in new_mutations:
            sym = "+" if m['type'] == 'GROW' else "-" if m['type'] == 'PRUNE' else "~"
            print(f"    {sym} {m['detail']}")

    if dream_cycles > 0:
        print(f"  DREAMING ({dream_cycles} cycles)...")
        dream_muts_before = len(brain.evolution.mutations)
        brain.dream(cycles=dream_cycles, verbose=False)
        dream_muts = brain.evolution.mutations[dream_muts_before:]
        if dream_muts:
            print(f"  DREAM MUTATIONS ({len(dream_muts)}):")
            for m in dream_muts:
                sym = "+" if m['type'] == 'GROW' else "-" if m['type'] == 'PRUNE' else "~"
                print(f"    {sym} {m['detail']}")

    neurons_after = brain.model.hidden_size
    loss_after = brain.model.smooth_loss
    grows_after = brain.evolution.total_grows
    prunes_after = brain.evolution.total_prunes
    corpus_after = len(brain.training_corpus)

    print(f"  END:   neurons={neurons_after}, loss={loss_after:.4f}, corpus={corpus_after} chars")
    print(f"         grows={grows_after}, prunes={prunes_after}, lr={brain.evolution.lr:.4f}")

    response = brain.respond("hello echo", temperature=0.5, length=60)
    print(f"  ECHO SAYS: '{response}'")

    results.append({
        'day': day_name.split('—')[0].strip(),
        'neurons': neurons_after,
        'loss': loss_after,
        'grows': grows_after - grows_before,
        'prunes': prunes_after - prunes_before,
        'corpus': corpus_after,
        'response': response
    })

# === SUMMARY ===
print()
print("=" * 65)
print("  7 DAY SUMMARY")
print("=" * 65)
print()
print(f"  {'Day':<12} {'Neurons':>8} {'Loss':>8} {'+Grows':>7} {'-Prunes':>8} {'Corpus':>8}")
print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")
for r in results:
    print(f"  {r['day']:<12} {r['neurons']:>8} {r['loss']:>8.4f} {r['grows']:>7} {r['prunes']:>8} {r['corpus']:>8}")
print()
print(f"  FINAL STATE:")
print(f"    Total neurons:      {brain.model.hidden_size}")
print(f"    Final loss:         {brain.model.smooth_loss:.4f}")
print(f"    Best loss:          {brain.evolution.best_loss:.4f}")
print(f"    Total grows:        {brain.evolution.total_grows}")
print(f"    Total prunes:       {brain.evolution.total_prunes}")
print(f"    Total LR mutations: {brain.evolution.total_lr_mutations}")
print(f"    Total mutations:    {len(brain.evolution.mutations)}")
print(f"    Learning rate:      {brain.evolution.lr:.4f}")
print(f"    Corpus size:        {len(brain.training_corpus)} chars")
print(f"    Conversation turns: {len(brain.conversation_log)}")
print(f"    Total epochs:       {brain.model.total_epochs}")
print()

print(f"  FULL MUTATION HISTORY ({len(brain.evolution.mutations)} events):")
for i, m in enumerate(brain.evolution.mutations):
    sym = "+" if m['type'] == 'GROW' else "-" if m['type'] == 'PRUNE' else "~"
    print(f"    [{i+1:3d}] {sym} {m['type']:12s} {m['detail']}")

print()
print(f"  FINAL RESPONSES AT DIFFERENT TEMPERATURES:")
for temp in [0.3, 0.7, 1.0, 1.5]:
    r = brain.respond("tell me about the ocean", temperature=temp, length=80)
    print(f"    temp={temp}: '{r}'")

print()
print("=" * 65)
print("  SIMULATION COMPLETE")
print("=" * 65)