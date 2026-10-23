#!/usr/bin/env python3
"""Check the fp16 positional encoding against the benchmark's sequence lengths.

megatron/model/vision/vit_backbone.py:217 hardcodes

    dtype = torch.float16
    position = torch.arange(0, length, dtype=dtype, device=dev)

fp16 represents consecutive integers exactly only up to 2048; above that the
spacing doubles at every power of two, and above 65504 there is no finite
value at all. The ALCF-4 sweep runs 4K to 1.73M tokens, so distinct tokens
collapse onto the same position -- and the two longest points overflow to
infinity, making every position vector NaN.

Run it anywhere torch is importable; it touches no devices.

    python3 -I ViT/check_posenc_fp16.py
"""
import math

import torch

HIDDEN = 1536          # VIT="315M"
FOM_POINTS = [         # factor, tokens
    (16, 16 ** 3),
    (24, 24 ** 3),
    (32, 32 ** 3),
    (64, 64 ** 3),
    (120, 120 ** 3),
]


def main():
    print(f"fp16 finite range: max {torch.finfo(torch.float16).max:,.0f}, "
          f"integers exact to 2,048")
    print()
    hdr = (f"{'factor':>7} {'tokens':>10} {'distinct':>10} {'collapse':>9} "
           f"{'overflowed':>11} {'verdict':>9}")
    print(hdr)
    print("-" * len(hdr))

    for factor, length in FOM_POINTS:
        pos = torch.arange(0, length, dtype=torch.float16)
        n_inf = int(torch.isinf(pos).sum())
        finite = pos[torch.isfinite(pos)]
        distinct = int(torch.unique(finite).numel())
        verdict = "NaN" if n_inf else ("degraded" if distinct < length else "exact")
        print(f"{factor:>7} {length:>10,} {distinct:>10,} "
              f"{length / max(distinct, 1):>8.0f}x {n_inf:>11,} {verdict:>9}")

    # What the collapse does to the encoding actually handed to the model:
    # tokens sharing an fp16 index get bit-identical position vectors, so the
    # transformer cannot tell them apart.
    print()
    for factor, length in FOM_POINTS[:3]:          # the three that stay finite
        pos = torch.arange(0, length, dtype=torch.float16).unsqueeze(1)
        div = torch.exp(torch.arange(0, HIDDEN, 2, dtype=torch.float16)
                        * -(math.log(10000.0) / HIDDEN))
        pe = torch.zeros((length, HIDDEN), dtype=torch.float16)
        pe[:, 0::2] = torch.sin(pos.float() * div)
        pe[:, 1::2] = torch.cos(pos.float() * div)
        dup = int((pe[1:] == pe[:-1]).all(dim=1).sum())
        print(f"factor={factor:>3} ({length:>9,} tokens): {dup:>9,} of "
              f"{length - 1:>9,} adjacent token pairs are positionally "
              f"indistinguishable ({100 * dup / (length - 1):>5.1f}%)")

    # And in fp32/bf16, for comparison. bf16 has an 8-bit exponent so it never
    # overflows here, but only 8 bits of mantissa -- worse integer spacing.
    print()
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        line = [f"{name}:"]
        for factor, length in FOM_POINTS:
            pos = torch.arange(0, length, dtype=dt)
            distinct = int(torch.unique(pos[torch.isfinite(pos)]).numel())
            line.append(f"f{factor}={100 * distinct / length:.0f}%")
        print("  ".join(line))
    print("(percent of token positions that remain distinguishable)")


if __name__ == "__main__":
    main()
