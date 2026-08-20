"""Named transformer profiles for Echo's scalable decoder architecture."""

TRANSFORMER_PROFILES = {
    "small": {
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "ff_multiplier": 4,
        "max_context": 256,
        "learning_rate": 0.0003,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
    },
    "local-8gb": {
        "d_model": 1024,
        "n_layers": 12,
        "n_heads": 8,
        "ff_multiplier": 4,
        "max_context": 512,
        "learning_rate": 0.0003,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
    },
    "local-8gb-lora": {
        "d_model": 1024,
        "n_layers": 12,
        "n_heads": 8,
        "ff_multiplier": 4,
        "max_context": 512,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_rank": 8,
        "freeze_base": True,
    },
    "from-scratch-0.5b": {
        "d_model": 1536,
        "n_layers": 12,
        "n_heads": 12,
        "ff_multiplier": 5,
        "max_context": 256,
        "learning_rate": 0.01,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "optimizer": "sgd",
        "gradient_checkpointing": True,
    },
    "coherent-150m": {
        "d_model": 1024,
        "n_layers": 16,
        "n_heads": 8,
        "n_kv_heads": 2,
        "ff_multiplier": 4,
        "max_context": 1024,
        "learning_rate": 0.0003,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "optimizer": "adamw",
        "gradient_checkpointing": True,
    },
    # A Gemma-2B-shaped target profile. It is intentionally opt-in: training
    # it requires substantially more memory and data than the local profile.
    # A Gemma-2B-shaped target profile for 96GB GPU (e.g. RTX 6000 Pro / H100)
    "echo-2b": {
        "d_model": 2048,
        "n_layers": 18,
        "n_heads": 8,
        "n_kv_heads": 2,
        "ff_multiplier": 4,
        "max_context": 2048,
        "learning_rate": 0.0003,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "optimizer": "adamw",
        "gradient_checkpointing": True,
    },
}


def get_transformer_profile(name="small"):
    """Return a copy so callers can tune a profile without global mutation."""
    if name not in TRANSFORMER_PROFILES:
        valid = ", ".join(sorted(TRANSFORMER_PROFILES))
        raise ValueError(f"unknown transformer profile {name!r}; choose {valid}")
    return dict(TRANSFORMER_PROFILES[name])
