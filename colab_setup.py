#!/usr/bin/env python3
"""Google Colab setup script for Echo training.

Run this in a Colab notebook cell:
    !pip install sentencepiece torch numpy
    !git clone <your-repo> /content/ECHO
    %cd /content/ECHO
    !python3 colab_setup.py --prepare

Then upload your data and train:
    !python3 colab_setup.py --train --profile coherent-150m --steps 20000

Or train the full 0.5B:
    !python3 colab_setup.py --train --profile from-scratch-0.5b --steps 20000
"""

import argparse
import os
import subprocess
import sys


def check_gpu():
    """Check if a GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"GPU: {name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print("WARNING: No GPU detected. Go to Runtime > Change runtime type > GPU")
            return False
    except ImportError:
        print("PyTorch not installed. Run: !pip install torch")
        return False


def install_deps():
    """Install required packages."""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "sentencepiece", "torch", "numpy"])


def prepare():
    """Prepare the Colab environment."""
    install_deps()
    check_gpu()

    # Create directories
    os.makedirs("pipeline/input/distill_only", exist_ok=True)
    os.makedirs("domain_brain", exist_ok=True)

    print("\n=== SETUP COMPLETE ===")
    print("\nNext steps:")
    print("1. Upload your training data to pipeline/input/distill_only/")
    print("   - Upload AllCombined.txt (Wikipedia data)")
    print("   - Upload distilled_data.txt (Q&A pairs)")
    print("   - Or upload any .txt files you want to train on")
    print()
    print("2. Start training:")
    print("   !python3 colab_setup.py --train --profile coherent-150m --steps 20000 --seq-len 512")
    print()
    print("3. For the full 0.5B model (fits in 16GB):")
    print("   !python3 colab_setup.py --train --profile from-scratch-0.5b --steps 20000 --seq-len 256")
    print()
    print("4. Download the checkpoint when done:")
    print("   !python3 colab_setup.py --download")
    print()
    print("Profiles available:")
    from echo_model_config import TRANSFORMER_PROFILES
    for name, cfg in TRANSFORMER_PROFILES.items():
        print(f"  {name}: d_model={cfg['d_model']}, layers={cfg['n_layers']}, ctx={cfg['max_context']}")


def train(profile, steps, seq_len, input_dir, output_dir, resume=False, lr=None):
    """Run training."""
    cmd = [
        sys.executable, "train_domain.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--profile", profile,
        "--steps", str(steps),
        "--seq-len", str(seq_len),
    ]
    if resume:
        cmd.append("--resume")
    if lr:
        cmd.extend(["--lr", str(lr)])

    print(f"Starting training: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def download():
    """List checkpoint files for download."""
    print("Checkpoint files to download:")
    for f in ["domain_brain/model.pt", "domain_brain/echo_domain.model",
              "domain_brain/tokens.u32", "domain_brain/training.json"]:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024**2
            print(f"  {f} ({size:.1f} MB)")
        else:
            print(f"  {f} (not found)")
    print("\nDownload these files from the Colab file browser (left sidebar > Files icon)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="setup environment")
    group.add_argument("--train", action="store_true", help="run training")
    group.add_argument("--download", action="store_true", help="list checkpoint files")
    parser.add_argument("--profile", default="coherent-150m")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--input-dir", default="pipeline/input/distill_only")
    parser.add_argument("--output-dir", default="domain_brain")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.train:
        train(args.profile, args.steps, args.seq_len, args.input_dir,
              args.output_dir, args.resume, args.lr)
    elif args.download:
        download()


if __name__ == "__main__":
    main()