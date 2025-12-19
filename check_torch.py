import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint (.pt)")
    parser.add_argument("ckpt_path", help="Path to checkpoint .pt/.pth")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    print("Keys:", list(ckpt.keys()))

    # Tenta descobrir um state_dict de ator ou genérico
    state = ckpt.get("actor") or ckpt.get("state_dict") or ckpt.get("model")
    if state:
        print("\nState dict tensors (name -> shape):")
        for k, v in state.items():
            if hasattr(v, "shape"):
                print(f"  {k}: {tuple(v.shape)}")

    # Mostra metadados comuns, se existirem
    for meta_key in ["hparams", "state_mask", "step", "best_avg", "bad_epochs"]:
        if meta_key in ckpt:
            print(f"{meta_key}: {ckpt[meta_key]}")

if __name__ == "__main__":
    main()