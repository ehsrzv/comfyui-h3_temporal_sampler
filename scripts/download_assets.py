"""Fetch the Sharp curve asset (assets/lms_curve.bin) for H3 IC Loader.

Only the adaln_t_table tensor of the pruned Comfy-Org checkpoint is fetched,
by byte range — the full checkpoint is never downloaded. Pure stdlib, no
huggingface_hub needed. The file is SHA256-verified on every run.
"""
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

CURVE = dict(repo="Comfy-Org/MiniMax-H3", revision="e5eb578a89295337b8ff433a035929ce0279e0b6",
             file="diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
             tensor="adaln_t_table",
             sha256="c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7")
OUT = Path(__file__).resolve().parents[1] / "assets" / "lms_curve.bin"


def read_range(url, start, length):
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{start + length - 1}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise RuntimeError("server ignored the Range request; refusing a full checkpoint download")
        data = r.read()
    if len(data) != length:
        raise RuntimeError("truncated range response")
    return data


def main():
    url = (f"https://huggingface.co/{CURVE['repo']}/resolve/"
           f"{CURVE['revision']}/{CURVE['file']}")
    header_length, = struct.unpack("<Q", read_range(url, 0, 8))
    tensor = json.loads(read_range(url, 8, header_length))[CURVE["tensor"]]
    assert tensor["dtype"] == "F32" and tensor["shape"] == [1025, 8], tensor
    start, end = tensor["data_offsets"]
    data = read_range(url, 8 + header_length + start, end - start)
    if hashlib.sha256(data).hexdigest() != CURVE["sha256"]:
        raise ValueError("adaln_t_table SHA256 mismatch")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"Sharp curve ready: {OUT} ({len(data)} bytes, SHA256 verified)")


if __name__ == "__main__":
    main()
