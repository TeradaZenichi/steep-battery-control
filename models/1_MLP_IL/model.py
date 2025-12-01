
try:
    from .hp import HyperParameters
except ImportError:  
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from hp import HyperParameters

from torch import nn
import torch
import os



_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu":  nn.ELU,
}


def _get_activation_cls(name: str) -> type[nn.Module]:
    key = name.lower()
    return _ACTIVATIONS[key]


class ImitationMLP(nn.Module):
    def __init__(self, hparams: HyperParameters):
        super().__init__()
        self.hparams = hparams
        layers = []
        in_dim = hparams.input_dim
        dropout = float(hparams.dropout)
        activation_cls = _get_activation_cls(hparams.activation)
        hidden = list(hparams.hidden_sizes)
        if not hidden:
            hidden = [max(2 * hparams.output_dim, 32)]
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, hparams.output_dim)

    @classmethod
    def from_json(cls, json_path: str | os.PathLike[str]) -> "ImitationMLP":
        hparams = HyperParameters.from_json(json_path)
        return cls(hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.backbone(x)
        return self.head(features)


if __name__ == "__main__":
    from pathlib import Path

    config_path = Path(__file__).with_name("model.json")
    hparams = HyperParameters.from_json(config_path)
    model = ImitationMLP(hparams)
    dummy_input = torch.randn(1, hparams.input_dim)
    output = model(dummy_input)
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    

    print(dummy_input)
    print(output)
