# ============================================================
# ECHO EVOLVE — Compound Neurogenesis Engine
# No pruning. No death. Only growth.
# The brain grows like compound interest: each new neuron
# increases the capacity for FUTURE growth.
# ============================================================

import math
import random
from echo_matrix import random_matrix, vec_zeros, zeros

class EvolutionTracker:
    """Tracks loss history and drives compound neuron growth.

    PRINCIPLE: Compound Interest on Neurons
    ───────────────────────────────────────
    Instead of pruning weak neurons (biological constraint we don't have),
    we grow new neurons at a rate proportional to current brain size.

    growth_amount = floor(current_neurons × interest_rate)

    So 24 neurons at 5% interest → +1 neuron
       48 neurons at 5% interest → +2 neurons
       96 neurons at 5% interest → +4 neurons

    The bigger the brain, the faster it grows. Like money in a bank,
    but for intelligence. No withdrawals. No fees. Only compound growth.

    NEW NEURON MATURITY:
    ───────────────────
    New neurons are "immature" for a grace period (N training steps).
    During this time they are protected and their importance is tracked
    separately. This ensures new neurons get a chance to learn before
    being evaluated.

    LEARNING RATE ADAPTATION:
    ────────────────────────
    As the brain grows, the learning rate automatically scales down
    slightly (larger networks need smaller steps). This is like how
    a large ship needs smaller rudder adjustments than a small boat.
    """

    def __init__(self, initial_hidden=24, min_hidden=12, max_hidden=1024,
                 lr_min=0.001, lr_max=0.1, initial_lr=0.02,
                 interest_rate=0.05, compound_interval=50,
                 maturity_period=100):
        # Architecture bounds
        self.min_hidden = min_hidden
        self.max_hidden = max_hidden
        self.hidden_size = initial_hidden

        # Learning rate bounds
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr = initial_lr
        self.initial_lr = initial_lr

        # --- COMPOUND GROWTH PARAMETERS ---
        # The interest rate: what fraction of current neurons to add per growth event
        self.interest_rate = interest_rate  # 5% by default

        # How often to check for growth (in training steps)
        self.compound_interval = compound_interval

        # How many steps a new neuron needs to "mature"
        self.maturity_period = maturity_period

        # Neuron birth records: {neuron_index: birth_step}
        self.neuron_births = {}  # Track when each neuron was born
        self.total_steps = 0     # Global step counter

        # Loss tracking
        self.loss_history = []
        self.best_loss = float('inf')
        self.stagnation_count = 0
        self.steps_since_check = 0

        # Mutation log
        self.mutations = []

        # Plateau detection (still useful — triggers growth when stuck)
        self.plateau_threshold = 0.03
        self.plateau_patience = 3

        # Growth tracking
        self.total_grows = 0
        self.total_growth_events = 0
        self.total_neurons_born = 0
        self.total_lr_mutations = 0

        # Neuron importance (for display only — no pruning)
        self.neuron_importance = [1.0] * initial_hidden

        # Growth history for visualization
        self.growth_history = [(0, initial_hidden)]  # (step, neuron_count)

    def record_loss(self, loss):
        """Record a loss value and check if compound growth should trigger."""
        self.loss_history.append(loss)
        if len(self.loss_history) > 200:
            self.loss_history.pop(0)

        self.total_steps += 1

        if loss < self.best_loss:
            improvement = self.best_loss - loss
            self.best_loss = loss
            if improvement > self.plateau_threshold:
                self.stagnation_count = 0
            return None

        self.steps_since_check += 1
        if self.steps_since_check < self.compound_interval:
            return None

        self.steps_since_check = 0
        return self._evaluate_growth()

    def _evaluate_growth(self):
        """Decide if and how much to grow. No pruning. Ever."""
        if len(self.loss_history) < 20:
            return None

        # Compare recent loss to older loss
        recent = self.loss_history[-10:]
        older = self.loss_history[-20:-10] if len(self.loss_history) >= 20 else self.loss_history[:10]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        improvement = older_avg - recent_avg

        mutation = None

        # --- PLATEAU: Loss not improving → GROW (compound) ---
        if improvement < self.plateau_threshold:
            self.stagnation_count += 1

            if self.stagnation_count >= self.plateau_patience:
                if self.hidden_size < self.max_hidden:
                    mutation = self._compound_grow(reason="plateau")
                    self.stagnation_count = 0

        # --- DOING WELL: Loss improving → GROW ANYWAY (reward growth) ---
        elif improvement > 0.1:
            # When learning is going well, compound growth is the MOST valuable
            # because new neurons have good gradients to learn from.
            if self.hidden_size < self.max_hidden:
                mutation = self._compound_grow(reason="momentum")
                self.stagnation_count = 0

                # Also boost learning rate slightly — ride the wave
                if random.random() < 0.4:
                    self._boost_learning_rate()

        # --- STEADY: Small improvement → occasional growth + LR mutation ---
        elif improvement > 0.02:
            if random.random() < 0.3 and self.hidden_size < self.max_hidden:
                mutation = self._compound_grow(reason="steady")
            if random.random() < 0.15:
                mutation = self._mutate_learning_rate() or mutation

        return mutation

    def _compound_grow(self, reason=""):
        """Compound neurogenesis: grow by interest_rate × current_size.

        This is the heart of the compound interest model.
        The bigger the brain, the more neurons it adds per growth event.

        growth = floor(current_neurons × interest_rate)
        growth = max(1, growth)  # Always add at least 1

        Like compound interest:
          24 neurons × 5% = 1.2 → +1 neuron  → 25 total
          48 neurons × 5% = 2.4 → +2 neurons → 50 total
          96 neurons × 5% = 4.8 → +4 neurons → 100 total
          200 neurons × 5% = 10  → +10 neurons → 210 total
        """
        # Calculate compound growth
        raw_growth = self.hidden_size * self.interest_rate
        n_new = max(1, int(math.floor(raw_growth)))

        # Don't exceed max
        n_new = min(n_new, self.max_hidden - self.hidden_size)
        if n_new <= 0:
            return None

        old_size = self.hidden_size
        self.hidden_size += n_new
        self.total_grows += n_new
        self.total_growth_events += 1
        self.total_neurons_born += n_new

        # Track birth of new neurons
        for i in range(old_size, old_size + n_new):
            self.neuron_births[i] = self.total_steps

        # Extend importance scores
        self.neuron_importance.extend([0.5] * n_new)

        # Record growth in history
        self.growth_history.append((self.total_steps, self.hidden_size))

        # Format reason
        reason_str = {
            'plateau': 'brain stuck, growing to break through',
            'momentum': 'learning well, compounding while gradients are rich',
            'steady': 'steady progress, incremental growth'
        }.get(reason, reason)

        event = {
            'type': 'GROW',
            'detail': f'compound growth ({reason}): {old_size} → {self.hidden_size} '
                      f'neurons (+{n_new}, rate={self.interest_rate*100:.0f}%)',
            'old_hidden': old_size,
            'new_hidden': self.hidden_size,
            'n_new': n_new,
            'reason': reason,
            'interest_rate': self.interest_rate,
            'step': self.total_steps
        }
        self.mutations.append(event)
        return event

    def _mutate_learning_rate(self):
        """Mutate the learning rate randomly."""
        old_lr = self.lr
        factor = random.choice([0.6, 0.75, 0.85, 1.15, 1.3, 1.5])
        new_lr = self.lr * factor
        new_lr = max(self.lr_min, min(self.lr_max, new_lr))
        self.lr = new_lr
        self.total_lr_mutations += 1

        direction = "↑" if new_lr > old_lr else "↓"
        event = {
            'type': 'LR_MUTATE',
            'detail': f'learning rate mutation: {old_lr:.4f} → {new_lr:.4f} {direction}',
            'old_lr': old_lr,
            'new_lr': new_lr
        }
        self.mutations.append(event)
        return event

    def _boost_learning_rate(self):
        """Slight LR boost when momentum is good."""
        old_lr = self.lr
        # Smaller boost for larger brains (they need smaller steps)
        size_factor = 1.0 + (0.1 * (24.0 / max(self.hidden_size, 1)))
        new_lr = min(self.lr_max, self.lr * size_factor)
        self.lr = new_lr
        self.total_lr_mutations += 1

        event = {
            'type': 'LR_BOOST',
            'detail': f'learning rate boost: {old_lr:.4f} → {new_lr:.4f} ↑ (brain size {self.hidden_size})',
            'old_lr': old_lr,
            'new_lr': new_lr
        }
        self.mutations.append(event)
        return event

    def update_neuron_importance(self, grads, model):
        """Update neuron importance scores. No pruning — for display only."""
        if 'dW_hy' not in grads:
            return

        dW_hy = grads['dW_hy']
        hidden = len(dW_hy[0]) if dW_hy else 0

        for j in range(hidden):
            if j >= len(self.neuron_importance):
                break
            grad_sum = 0.0
            for i in range(len(dW_hy)):
                grad_sum += abs(dW_hy[i][j])

            old_score = self.neuron_importance[j]
            new_score = 0.9 * old_score + 0.1 * grad_sum
            self.neuron_importance[j] = new_score

    def compound_projection(self, steps=100):
        """Project future brain size using compound interest formula.

        A(t) = A₀ × (1 + r)^t

        Where:
          A(t) = neurons after t growth events
          A₀   = current neuron count
          r    = interest rate
          t    = number of growth events
        """
        projections = []
        a0 = self.hidden_size
        r = self.interest_rate
        for t in range(1, steps + 1):
            projected = int(a0 * ((1 + r) ** t))
            projected = min(projected, self.max_hidden)
            projections.append((t, projected))
        return projections

    def stats(self):
        """Return evolution statistics."""
        return (
            f"  hidden neurons: {self.hidden_size} (min={self.min_hidden}, max={self.max_hidden})\n"
            f"  interest rate:  {self.interest_rate*100:.0f}%% per growth event\n"
            f"  learning rate:  {self.lr:.4f}\n"
            f"  best loss:      {self.best_loss:.4f}\n"
            f"  total steps:    {self.total_steps}\n"
            f"  growth events:  {self.total_growth_events}\n"
            f"  neurons born:   {self.total_neurons_born}\n"
            f"  lr mutations:   {self.total_lr_mutations}\n"
            f"  total mutations: {len(self.mutations)}\n"
            f"  pruning:        NEVER (compound growth only)\n"
        )

    def recent_mutations(self, n=5):
        return self.mutations[-n:] if self.mutations else []

    def growth_curve(self, n=10):
        """Show the projected compound growth curve."""
        projections = self.compound_projection(n)
        lines = [f"  Compound growth projection ({self.interest_rate*100:.0f}% interest):"]
        current = self.hidden_size
        for t, projected in projections:
            bar = "█" * min(40, int(projected / self.max_hidden * 40))
            lines.append(f"    +{t:2d} events: {projected:4d} neurons {bar}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'min_hidden': self.min_hidden,
            'max_hidden': self.max_hidden,
            'lr': self.lr,
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'initial_lr': self.initial_lr,
            'interest_rate': self.interest_rate,
            'compound_interval': self.compound_interval,
            'maturity_period': self.maturity_period,
            'neuron_births': self.neuron_births,
            'total_steps': self.total_steps,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'stagnation_count': self.stagnation_count,
            'steps_since_check': self.steps_since_check,
            'neuron_importance': self.neuron_importance,
            'total_grows': self.total_grows,
            'total_growth_events': self.total_growth_events,
            'total_neurons_born': self.total_neurons_born,
            'total_lr_mutations': self.total_lr_mutations,
            'mutations': self.mutations,
            'growth_history': self.growth_history
        }

    @classmethod
    def from_dict(cls, d):
        e = cls(
            initial_hidden=d['hidden_size'],
            min_hidden=d['min_hidden'],
            max_hidden=d['max_hidden'],
            lr_min=d['lr_min'],
            lr_max=d['lr_max'],
            initial_lr=d.get('initial_lr', 0.02),
            interest_rate=d.get('interest_rate', 0.05),
            compound_interval=d.get('compound_interval', 50),
            maturity_period=d.get('maturity_period', 100)
        )
        e.lr = d['lr']
        e.neuron_births = d.get('neuron_births', {})
        e.total_steps = d.get('total_steps', 0)
        e.loss_history = d.get('loss_history', [])
        e.best_loss = d.get('best_loss', float('inf'))
        e.stagnation_count = d.get('stagnation_count', 0)
        e.steps_since_check = d.get('steps_since_check', 0)
        e.neuron_importance = d.get('neuron_importance', [1.0] * e.hidden_size)
        e.total_grows = d.get('total_grows', 0)
        e.total_growth_events = d.get('total_growth_events', 0)
        e.total_neurons_born = d.get('total_neurons_born', 0)
        e.total_lr_mutations = d.get('total_lr_mutations', 0)
        e.mutations = d.get('mutations', [])
        e.growth_history = d.get('growth_history', [(0, e.hidden_size)])
        return e