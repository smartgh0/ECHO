"""Quantum-inspired decoder transformer for Echo.

Architecture upgrades matching modern LLMs (Qwen2.5-style):
- RMSNorm instead of LayerNorm
- SwiGLU feed-forward instead of GELU
- RoPE (Rotary Position Embedding) instead of learned positions
- GQA (Grouped-Query Attention) with configurable KV head ratio

Quantum feature: each Q/K/V/O projection has two weight branches mixed by an
input-conditional sigmoid gate. Branch A specializes toward coding/tools;
branch B toward general/identity (via supervised route loss when labels exist).
"""

import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    """RMSNorm — simpler than LayerNorm, no mean centering or bias."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class QuantumLinear(nn.Module):
    """Two-branch linear with input-conditional mix and sharpness temperature.

    Forward mix:
        p = σ((gate_proj(x) + branch_logit) * temp)
        y = p * (x @ W_a) + (1 - p) * (x @ W_b)

    gate_proj is zero-initialized so resumed checkpoints start at the old
    static 50/50 behaviour and then learn routing from data + aux losses.
    """

    def __init__(self, input_size, output_size, lora_rank=0, bias=True):
        super().__init__()
        scale = 1.0 / math.sqrt(input_size)
        self.weight_a = nn.Parameter(torch.randn(output_size, input_size) * scale)
        self.weight_b = nn.Parameter(torch.randn(output_size, input_size) * scale)
        self.bias = nn.Parameter(torch.zeros(output_size)) if bias else None
        self.branch_logit = nn.Parameter(torch.tensor(0.0))
        # Start warmer (sharper sigmoid) so small logit moves commit faster.
        self.branch_temp = nn.Parameter(torch.tensor(2.0))
        # Per-token router; near-zero init keeps resume stable, tiny noise breaks 50/50 ties.
        self.gate_proj = nn.Linear(input_size, 1, bias=True)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gate_proj.bias)
        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora_a = nn.Parameter(torch.randn(lora_rank, input_size) * 0.01)
            self.lora_b = nn.Parameter(torch.zeros(output_size, lora_rank))
        self._last_mix = None

    def gate_temperature(self):
        return torch.sigmoid(self.branch_temp) * 4.0 + 0.5  # [0.5, 4.5]

    def effective_mix_from_logit(self):
        """Static mix ignoring input (for stats when no forward has run)."""
        return torch.sigmoid(self.branch_logit * self.gate_temperature())

    def mixed_weight(self):
        mix = self.effective_mix_from_logit()
        return mix * self.weight_a + (1.0 - mix) * self.weight_b

    def forward(self, values):
        out_a = F.linear(values, self.weight_a, self.bias)
        out_b = F.linear(values, self.weight_b, self.bias)
        cond = self.gate_proj(values).squeeze(-1)
        mix = torch.sigmoid((cond + self.branch_logit) * self.gate_temperature())
        self._last_mix = mix
        mix_u = mix.unsqueeze(-1)
        result = mix_u * out_a + (1.0 - mix_u) * out_b
        if self.lora_rank > 0:
            result = result + F.linear(F.linear(values, self.lora_a), self.lora_b) / self.lora_rank
        return result

    def branch_probability(self):
        if self._last_mix is not None:
            return float(self._last_mix.detach().float().mean().cpu())
        return float(self.effective_mix_from_logit().detach().cpu())

    def branch_diversity_loss(self):
        """Penalize aligned branches so W_a and W_b stay distinguishable."""
        flat_a = self.weight_a.reshape(-1).float()
        flat_b = self.weight_b.reshape(-1).float()
        cosine = F.cosine_similarity(flat_a, flat_b, dim=0)
        return cosine.square()


def rotate_half(x):
    """Rotate half the hidden dims for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class QuantumTransformerBlock(nn.Module):
    """Transformer block with RMSNorm, GQA, RoPE, SwiGLU, and quantum attention."""

    def __init__(self, d_model, n_heads, n_kv_heads=None, ff_multiplier=4, lora_rank=0,
                 rope_theta=10000.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        n_kv_heads = n_kv_heads or n_heads
        if n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope_theta = rope_theta

        # Quantum attention projections — Q and O use n_heads, K and V use n_kv_heads (GQA)
        self.q_proj = QuantumLinear(d_model, n_heads * self.head_dim, lora_rank)
        self.k_proj = QuantumLinear(d_model, n_kv_heads * self.head_dim, lora_rank)
        self.v_proj = QuantumLinear(d_model, n_kv_heads * self.head_dim, lora_rank)
        self.o_proj = QuantumLinear(n_heads * self.head_dim, d_model, lora_rank)

        # RMSNorm instead of LayerNorm
        self.norm_attn = RMSNorm(d_model)
        self.norm_ff = RMSNorm(d_model)

        # SwiGLU feed-forward: gate × up → down
        # SwiGLU uses 3 matrices (gate, up, down) but we use 2 with interleaving
        # FF hidden = ff_multiplier * d_model, but SwiGLU needs 2x that for gate+up
        self.ff_hidden = ff_multiplier * d_model
        self.ff_gate = nn.Linear(d_model, self.ff_hidden, bias=False)
        self.ff_up = nn.Linear(d_model, self.ff_hidden, bias=False)
        self.ff_down = nn.Linear(self.ff_hidden, d_model, bias=False)

    def _rope_freqs(self, seq_len, device, dtype):
        """Compute RoPE frequency table."""
        half = self.head_dim // 2
        freqs = 1.0 / (self.rope_theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * freqs[None, :]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1).to(dtype)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1).to(dtype)
        return cos, sin

    def _attention(self, values):
        batch_size, sequence_length, _ = values.shape
        q = self.q_proj(values).view(batch_size, sequence_length, self.n_heads, self.head_dim)
        k = self.k_proj(values).view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(values).view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self._rope_freqs(sequence_length, values.device, values.dtype)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q, k = apply_rope(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        causal = torch.triu(
            torch.ones(sequence_length, sequence_length, device=values.device, dtype=torch.bool),
            diagonal=1,
        )
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=~causal)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.n_heads * self.head_dim)
        return self.o_proj(attended)

    def forward(self, values):
        values = values + self._attention(self.norm_attn(values))
        normed = self.norm_ff(values)
        # SwiGLU: down(silu(gate(x)) * up(x))
        feedforward = self.ff_down(F.silu(self.ff_gate(normed)) * self.ff_up(normed))
        return values + feedforward

    def quantum_stats(self):
        projections = (self.q_proj, self.k_proj, self.v_proj, self.o_proj)
        probabilities = [projection.branch_probability() for projection in projections]
        entropy = 0.0
        committed = 0
        for probability in probabilities:
            probability = min(max(probability, 1e-10), 1.0 - 1e-10)
            entropy -= probability * math.log(probability) + (1.0 - probability) * math.log(1.0 - probability)
            if probability < 0.2 or probability > 0.8:
                committed += 1
        return (
            entropy / len(probabilities),
            sum(probabilities) / max(len(probabilities), 1),
            committed,
        )


class QuantumTransformerLM(nn.Module):
    """Causal language model with quantum attention, RMSNorm, SwiGLU, RoPE, and GQA."""

    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4,
                 n_kv_heads=None, ff_multiplier=4, max_context=256, learning_rate=0.0003,
                 batch_size=1, gradient_accumulation_steps=1,
                 lora_rank=0, freeze_base=False, profile_name='small',
                 optimizer='adamw', gradient_checkpointing=False,
                 device=None, seed=42):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        n_kv_heads = n_kv_heads or n_heads
        if n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        random.seed(seed)
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ff_multiplier = ff_multiplier
        self.max_context = max_context
        self.learning_rate = learning_rate
        self.profile_name = profile_name
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lora_rank = lora_rank
        self.freeze_base = freeze_base
        self.optimizer_name = optimizer
        self.gradient_checkpointing = gradient_checkpointing
        self.device = torch.device(
            device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.seed = seed
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # No position embedding table — RoPE handles positions
        self.blocks = nn.ModuleList(
            [QuantumTransformerBlock(d_model, n_heads, n_kv_heads, ff_multiplier, lora_rank)
             for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self._initialize_weights()
        if freeze_base and lora_rank > 0:
            for name, parameter in self.named_parameters():
                parameter.requires_grad = name.startswith('blocks.') and '.lora_' in name
        self.to(self.device)
        # Stronger pressure: CE alone leaves p≈0.5 and W_a≈W_b (branch_cos→1).
        self.quantum_route_weight = 0.12
        self.quantum_commit_weight = 0.10
        self.quantum_diversity_weight = 0.02
        self.quantum_gate_lr_scale = 10.0
        self._build_optimizer()
        self.use_amp = self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.total_epochs = 0
        self.total_chars_seen = 0
        self.smooth_loss = None
        self.last_quantum_aux = 0.0

    def _build_optimizer(self):
        """AdamW/SGD with higher LR and no weight decay on gate parameters."""
        gate_parameters = []
        other_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                "gate_proj" in name
                or name.endswith("branch_logit")
                or name.endswith("branch_temp")
            ):
                gate_parameters.append(parameter)
            else:
                other_parameters.append(parameter)
        groups = []
        if other_parameters:
            groups.append({
                "params": other_parameters,
                "lr": self.learning_rate,
                "weight_decay": 0.01 if self.optimizer_name != "sgd" else 0.0,
                "lr_scale": 1.0,
            })
        if gate_parameters:
            groups.append({
                "params": gate_parameters,
                "lr": self.learning_rate * self.quantum_gate_lr_scale,
                "weight_decay": 0.0,
                "lr_scale": self.quantum_gate_lr_scale,
            })
        if not groups:
            raise RuntimeError("No trainable parameters for optimizer")
        if self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(groups, momentum=0.9)
        else:
            self.optimizer = torch.optim.AdamW(groups)

    def _initialize_weights(self):
        """Use language-model scale initialization for tied embeddings."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                # Keep quantum gate routers at zero (set in QuantumLinear).
                if any(module is block.q_proj.gate_proj
                       or module is block.k_proj.gate_proj
                       or module is block.v_proj.gate_proj
                       or module is block.o_proj.gate_proj
                       for block in self.blocks):
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, token_indices):
        if token_indices.dim() == 1:
            token_indices = token_indices.unsqueeze(0)
        token_indices = token_indices[:, -self.max_context:]
        values = self.token_embedding(token_indices)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and values.requires_grad:
                values = checkpoint(block, values, use_reentrant=False)
            else:
                values = block(values)
        return self.lm_head(self.final_norm(values))

    def iter_quantum_linears(self):
        for block in self.blocks:
            yield block.q_proj
            yield block.k_proj
            yield block.v_proj
            yield block.o_proj

    def quantum_aux_loss(self, branch_targets=None):
        """Route + commitment + branch-diversity regularizers.

        branch_targets: optional float tensor [batch] with 1 = code/tools (branch A),
        0 = general/identity (branch B).
        """
        mixes = []
        diversity = []
        for projection in self.iter_quantum_linears():
            if projection._last_mix is not None:
                mixes.append(projection._last_mix.float())
            diversity.append(projection.branch_diversity_loss())
        if not mixes:
            return torch.zeros((), device=self.device)

        # Mean gate probability per sequence item, averaged across all projections.
        per_proj = []
        for mix in mixes:
            if mix.dim() == 2:
                per_proj.append(mix.mean(dim=1))
            elif mix.dim() == 1:
                per_proj.append(mix)
            else:
                per_proj.append(mix.reshape(-1))
        mean_p = torch.stack(per_proj, dim=0).mean(dim=0)  # [batch]

        # 4p(1-p) peaks at 0.5 — minimize to force commitment.
        commit = (4.0 * mean_p * (1.0 - mean_p)).mean()
        loss = self.quantum_commit_weight * commit
        loss = loss + self.quantum_diversity_weight * torch.stack(diversity).mean()

        if branch_targets is not None:
            target = branch_targets.float().to(device=mean_p.device, dtype=mean_p.dtype)
            if target.shape != mean_p.shape:
                target = target.reshape(-1)[: mean_p.numel()]
                mean_p = mean_p[: target.numel()]
            route = F.binary_cross_entropy(mean_p.clamp(1e-4, 1.0 - 1e-4), target)
            loss = loss + self.quantum_route_weight * route
        return loss

    def train_step(self, inputs, targets, h_prev=None, target_mask=None, branch_target=None):
        masks = [target_mask] if target_mask is not None else None
        targets_branch = [branch_target] if branch_target is not None else None
        return self.train_step_batch(
            [inputs], [targets], h_prev=h_prev, mask_batch=masks, branch_targets=targets_branch,
        )

    def train_step_batch(self, input_batch, target_batch, h_prev=None, mask_batch=None,
                         branch_targets=None):
        del h_prev
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        aux_values = []
        if mask_batch is None:
            mask_batch = [None] * len(input_batch)
        if branch_targets is None:
            branch_targets = [None] * len(input_batch)
        # group same-length sequences so they can be stacked into one real
        # batched tensor per micro-step instead of looping sample-by-sample
        groups = {}
        for inputs, targets, mask, branch in zip(
            input_batch, target_batch, mask_batch, branch_targets
        ):
            groups.setdefault(len(inputs), []).append((inputs, targets, mask, branch))
        micro_batch = max(1, self.batch_size)
        chunks = []
        for pairs in groups.values():
            for i in range(0, len(pairs), micro_batch):
                chunks.append(pairs[i:i + micro_batch])
        n_chunks = len(chunks) or 1
        for pairs in chunks:
            inputs = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=self.device)
            targets = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.forward(inputs)
                flat_logits = logits.reshape(-1, self.vocab_size)
                flat_targets = targets.reshape(-1)
                if pairs[0][2] is None:
                    ce_loss = F.cross_entropy(flat_logits, flat_targets)
                else:
                    masks = torch.tensor([p[2] for p in pairs], dtype=torch.float32, device=self.device)
                    flat_masks = masks.reshape(-1)
                    token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                    denom = flat_masks.sum().clamp(min=1.0)
                    ce_loss = (token_loss * flat_masks).sum() / denom
            branch_vals = [p[3] for p in pairs]
            if any(value is not None for value in branch_vals):
                branch_tensor = torch.tensor(
                    [0.0 if value is None else float(value) for value in branch_vals],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                branch_tensor = None
            # Aux uses BCE on probabilities — keep it in float32 outside autocast.
            aux = self.quantum_aux_loss(branch_tensor)
            loss = ce_loss.float() + aux
            loss_value = float(loss.detach().cpu())
            aux_values.append(float(aux.detach().cpu()))
            if not (loss_value == loss_value) or loss_value > 1e6:
                continue
            losses.append(loss_value)
            (loss / n_chunks).backward()
        if losses:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
        loss_value = sum(losses) / max(len(losses), 1)
        self.last_quantum_aux = sum(aux_values) / max(len(aux_values), 1)
        self.total_chars_seen += sum(len(targets) for targets in target_batch)
        if self.smooth_loss is None:
            self.smooth_loss = loss_value
        else:
            self.smooth_loss = 0.99 * self.smooth_loss + 0.01 * loss_value
        return loss_value, None
    @torch.no_grad()
    def sample(self, seed_input, length=100, temperature=0.7, h_prev=None):
        del h_prev
        if seed_input:
            indices = [max(range(len(vector)), key=vector.__getitem__) for vector in seed_input]
        else:
            indices = [0]
        generated = []
        self.eval()
        for _ in range(length):
            context = torch.tensor(indices[-self.max_context:], dtype=torch.long, device=self.device)
            logits = self.forward(context)[0, -1] / max(temperature, 0.01)
            probabilities = torch.softmax(logits, dim=-1)
            next_index = int(torch.multinomial(probabilities, 1).item())
            generated.append(next_index)
            indices.append(next_index)
        return generated

    def quantum_stats(self):
        entropies = []
        probabilities = []
        committed = 0
        total_gates = 0
        for block in self.blocks:
            entropy, probability, block_committed = block.quantum_stats()
            entropies.append(entropy)
            probabilities.append(probability)
            committed += block_committed
            total_gates += 4
        average_entropy = sum(entropies) / max(len(entropies), 1)
        return {
            'avg_entropy': average_entropy,
            'max_entropy': math.log(2.0),
            'entropy_ratio': average_entropy / math.log(2.0),
            'branch_probability': sum(probabilities) / max(len(probabilities), 1),
            'committed_gates': committed,
            'total_gates': total_gates,
            'layers': len(self.blocks),
            'samples_per_step': 1,
        }

    def info(self):
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        loss = f"{self.smooth_loss:.4f}" if self.smooth_loss is not None else "N/A"
        kv_info = f"  kv_heads:       {self.n_kv_heads}\n" if self.n_kv_heads != self.n_heads else ""
        return (f"  [QUANTUM TRANSFORMER]\n"
                f"  vocab_size:     {self.vocab_size}\n"
                f"  d_model:        {self.d_model}\n"
                f"  layers:         {self.n_layers}\n"
                f"  heads:          {self.n_heads}\n"
                f"{kv_info}"
                f"  parameters:     {parameter_count:,}\n"
                f"  norm:           RMSNorm\n"
                f"  activation:     SwiGLU\n"
                f"  position:       RoPE\n"
                f"  attention:      {'GQA' if self.n_kv_heads != self.n_heads else 'MHA'}\n"
                f"  loss:           {loss}")

    def to_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'n_kv_heads': self.n_kv_heads,
            'ff_multiplier': self.ff_multiplier,
            'max_context': self.max_context,
            'learning_rate': self.learning_rate,
            'profile_name': self.profile_name,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'lora_rank': self.lora_rank,
            'freeze_base': self.freeze_base,
            'optimizer': self.optimizer_name,
            'gradient_checkpointing': self.gradient_checkpointing,
            'seed': self.seed,
            'total_epochs': self.total_epochs,
            'total_chars_seen': self.total_chars_seen,
            'smooth_loss': self.smooth_loss,
            'state_dict': {key: value.detach().cpu().tolist() for key, value in self.state_dict().items()},
        }

    @classmethod
    def from_dict(cls, data):
        model = cls(
            vocab_size=data['vocab_size'],
            d_model=data['d_model'],
            n_layers=data['n_layers'],
            n_heads=data['n_heads'],
            n_kv_heads=data.get('n_kv_heads', data['n_heads']),
            ff_multiplier=data.get('ff_multiplier', 4),
            max_context=data['max_context'],
            learning_rate=data['learning_rate'],
            batch_size=data.get('batch_size', 1),
            gradient_accumulation_steps=data.get('gradient_accumulation_steps', 1),
            lora_rank=data.get('lora_rank', 0),
            freeze_base=data.get('freeze_base', False),
            profile_name=data.get('profile_name', 'small'),
            optimizer=data.get('optimizer', 'adamw'),
            gradient_checkpointing=data.get('gradient_checkpointing', False),
            seed=data.get('seed', 42),
        )
        state = {
            key: value.detach().clone() if torch.is_tensor(value) else torch.tensor(value)
            for key, value in data['state_dict'].items()
        }
        incompatible = model.load_state_dict(state, strict=False)
        missing_gates = [key for key in incompatible.missing_keys if "gate_proj" in key]
        if missing_gates:
            # Old static-gate checkpoints: keep W_a/W_b, add fresh routers with
            # symmetry-breaking noise so commit/route losses can move off 0.5.
            with torch.no_grad():
                for projection in model.iter_quantum_linears():
                    nn.init.normal_(projection.gate_proj.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(projection.gate_proj.bias)
                    projection.branch_logit.add_(torch.randn((), device=projection.branch_logit.device) * 0.15)
                    projection.branch_temp.fill_(2.0)
            model._build_optimizer()
        model.total_epochs = data.get('total_epochs', 0)
        model.total_chars_seen = data.get('total_chars_seen', 0)
        model.smooth_loss = data.get('smooth_loss')
        return model