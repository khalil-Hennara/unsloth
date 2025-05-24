import torch
import torch.nn.functional as F
from transformers.models.siglip2.modeling_siglip2 import Siglip2MLP, ACT2FN

# --- 1) Bring in your Triton patch function ----
from unsloth.models.siglibvision import fast_siglip2_mlp  # adjust import

# --- 2) Monkey-patch Siglip2MLP.forward ----
# Siglip2MLP.forward = fast_siglip2_mlp_inference


# --- 3) Build a dummy config ----
class DummyConfig:
    hidden_act = "gelu"
    hidden_size = 16
    intermediate_size = 64


class FastSiglip2MLP(Siglip2MLP):
    def forward(self, hidden_states):
        # dispatch directly to your Triton function
        return fast_siglip2_mlp(self, hidden_states)


def test_fast_mlp_matches_reference():
    cfg = DummyConfig()

    # --- 4) Instantiate two MLPs and copy weights/biases ----
    mlp_ref = Siglip2MLP(cfg)
    # use the fast-forward pass
    mlp_fast = FastSiglip2MLP(cfg)

    # Copy exactly so they’re identical
    mlp_fast.fc1.weight.data.copy_(mlp_ref.fc1.weight.data)
    mlp_fast.fc1.bias.data.copy_(mlp_ref.fc1.bias.data)
    mlp_fast.fc2.weight.data.copy_(mlp_ref.fc2.weight.data)
    mlp_fast.fc2.bias.data.copy_(mlp_ref.fc2.bias.data)

    # --- 5) Run a forward on random input and compare ----
    device = "cuda" # we can only run on GPU as the kernel only work with GPU.
    mlp_ref.to(device)
    mlp_fast.to(device)

    x = torch.randn(2, 10, cfg.hidden_size, device=device)  # e.g. batch=2, seq_len=10

    with torch.no_grad():
        out_ref = mlp_ref(x)  # uses original Python forward
        out_fast = mlp_fast(x)  # uses your Triton-accelerated forward

    # This will raise an error if they differ outside the tolerance
    torch.testing.assert_close(out_ref,
                               out_fast,
                               rtol=1e-3,
                               atol=1e-4,
                               msg="Triton MLP output does not match reference!"
    )

    print("✅ fast_siglip2_mlp_inference matches the original Siglip2MLP!")
