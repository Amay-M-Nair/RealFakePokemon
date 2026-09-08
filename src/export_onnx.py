"""Export the EMA generator to ONNX for CPU serving.

    python -m src.export_onnx checkpoints/fastgan/final.pt --out server/models/generator.onnx

Why ONNX rather than shipping PyTorch: torch on CPU is ~800MB installed and
~300MB resident, which does not fit a 512MB free tier. onnxruntime is ~50MB and
the fp16 graph is ~60MB, so the whole service sits around 250-300MB. That single
choice is what makes free hosting possible.

Also writes z_mean.npy -- the empirical latent mean, which the truncation slider
interpolates toward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import get_device
from src.models.blocks import set_deterministic_noise
from src.sample import load_generator


def fold_spectral_norm(model: nn.Module) -> int:
    """Remove spectral-norm reparametrisation in place; returns how many were folded.

    Spectral norm is a training-time Lipschitz constraint. It keeps both
    `weight_orig` and the normalised `weight` alive, so exporting with it
    attached roughly doubles the graph -- 222MB instead of ~111MB here. Folding
    it collapses them to the single effective weight, which is numerically what
    inference already uses.
    """
    from torch.nn.utils import remove_spectral_norm

    folded = 0
    for module in model.modules():
        try:
            remove_spectral_norm(module)
            folded += 1
        except (ValueError, RuntimeError):
            continue  # module had no spectral norm attached
    return folded


class GeneratorWrapper(nn.Module):
    """Expose a single-output graph.

    The training generator returns [full_res, 128px]; the 128px branch exists
    only to give the discriminator a second scale and is dead weight at serving
    time. Exporting just the full-resolution head keeps the graph smaller.
    """

    def __init__(self, netG):
        super().__init__()
        self.netG = netG

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.netG(z)[0]


@torch.no_grad()
def compute_z_mean(netG, samples: int = 10_000, batch: int = 256, device=None) -> np.ndarray:
    """Empirical mean of the latent prior.

    For a standard normal this is ~0, but measuring it keeps truncation correct
    if the prior ever changes, and costs one cheap pass.
    """
    device = device or next(netG.parameters()).device
    total = torch.zeros(netG.z_dim, device=device)
    for i in range(0, samples, batch):
        n = min(batch, samples - i)
        total += torch.randn(n, netG.z_dim, device=device).sum(0)
    return (total / samples).cpu().numpy().astype(np.float32)


def export(args) -> None:
    device = torch.device("cpu") if args.cpu else get_device()
    netG = load_generator(args.checkpoint, device, use_ema=not args.raw)
    netG.eval()
    # Must happen before tracing: it removes the RandomNormalLike nodes the
    # fp16 converter chokes on, and is what makes (seed -> image) a function.
    print(f"deterministic noise on {set_deterministic_noise(netG)} NoiseInjection modules")
    if not args.keep_spectral_norm:
        print(f"folded spectral norm on {fold_spectral_norm(netG)} modules")
    wrapper = GeneratorWrapper(netG).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, netG.z_dim, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        str(args.out),
        input_names=["z"],
        output_names=["image"],
        # Batch is dynamic so one session serves both single renders and grids.
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"exported fp32 -> {args.out}  ({args.out.stat().st_size/2**20:.1f} MB)")

    if args.fp16:
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(str(args.out))
        onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), str(args.out))
        print(f"converted to fp16 -> {args.out}  ({args.out.stat().st_size/2**20:.1f} MB)")

    z_mean = compute_z_mean(netG, device=device)
    mean_path = args.out.with_name("z_mean.npy")
    np.save(mean_path, z_mean)
    print(f"z_mean (|mean|={np.abs(z_mean).mean():.4f}) -> {mean_path}")

    verify(args.out, netG, device, args.tolerance)


@torch.no_grad()
def verify(onnx_path: Path, netG, device, tolerance: float) -> None:
    """Assert the exported graph matches the torch model on the same latents."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    z = torch.randn(2, netG.z_dim)
    torch_out = netG(z.to(device))[0].cpu().numpy()
    onnx_out = sess.run(["image"], {"z": z.numpy().astype(np.float32)})[0]

    diff = np.abs(torch_out - onnx_out)
    print(f"parity: max |diff| = {diff.max():.5f}, mean = {diff.mean():.5f} (tolerance {tolerance})")
    if diff.max() > tolerance:
        raise SystemExit(
            f"ONNX output diverges from torch by {diff.max():.5f} > {tolerance}; refusing to ship"
        )
    print("parity OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("server/models/generator.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="halve the file size (~120MB -> ~60MB)")
    parser.add_argument("--raw", action="store_true", help="export live G weights, not the EMA copy")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--keep-spectral-norm",
        action="store_true",
        help="skip folding spectral norm (roughly doubles the exported size)",
    )
    export(parser.parse_args())


if __name__ == "__main__":
    main()
