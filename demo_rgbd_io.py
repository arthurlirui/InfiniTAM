"""Demo: read & write the dataset01-0922-2058 RGBD+IMU recording using the
eskf_vgm submodule's :class:`eskf.recording.Recording` container.

The dataset on disk follows the EuRoC-ASL-style layout that ``Recording``
expects::

    dataset01-0922-2058/
        device0/ meta.json rgbd.csv imu.csv color/*.jpg depth/*.npy
        device1/ ...
        device2/ ...
        device3/ ...

This script
  1. loads every device via ``Recording.load`` (metadata + images + depth),
  2. prints a per-device summary (intrinsics, frame/IMU counts, depth stats),
  3. dumps one frame per device to ``demo_out/preview/`` as PNG (color) and a
     normalized depth PNG, so you can eyeball the data without a viewer,
  4. round-trips device0 through ``Recording.save`` into ``demo_out/roundtrip/``
     to prove the writer reproduces the same on-disk format, then reloads it
     and compares the depth arrays for equality.

Run::

    py -3 demo_rgbd_io.py
"""

from __future__ import annotations

import os
import sys
import shutil

import numpy as np
import cv2

# Make the submodule importable without installing it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ESKF_ROOT = os.path.join(_HERE, "third_party", "eskf_vgm")
if _ESKF_ROOT not in sys.path:
    sys.path.insert(0, _ESKF_ROOT)

from eskf.recording import Recording  # noqa: E402

DATA_DIR = os.path.join(_HERE, "data", "dataset01-0922-2058")
OUT_DIR = os.path.join(_HERE, "demo_out")
PREVIEW_DIR = os.path.join(OUT_DIR, "preview")
ROUNDTRIP_DIR = os.path.join(OUT_DIR, "roundtrip")


def _depth_to_visu(depth_m: np.ndarray) -> np.ndarray:
    """Map a float depth map (meters) to an 8-bit grayscale PNG for display.

    Pixels at 0 m are treated as invalid (rendered black); valid depths are
    normalized by the 5th–95th percentile range so a few far outliers don't
    wash out the image.
    """
    d = depth_m.copy()
    valid = d > 0
    if not np.any(valid):
        return np.zeros_like(d, dtype=np.uint8)
    lo, hi = np.percentile(d[valid], [5, 95])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    out = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def main() -> None:
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"dataset not found: {DATA_DIR}")

    devices = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("device")
    )
    print(f"found {len(devices)} devices under {DATA_DIR}")
    print("=" * 72)

    os.makedirs(PREVIEW_DIR, exist_ok=True)

    recordings = {}
    for dev in devices:
        dev_path = os.path.join(DATA_DIR, dev)
        print(f"loading {dev} ...")
        rec = Recording.load(dev_path, load_images=True)
        recordings[dev] = rec

        # --- summary ---
        print(f"  device_serial : {rec.device_serial}")
        print(f"  n_frames      : {len(rec.frames)}")
        print(f"  n_imu         : {len(rec.imu)}")
        if rec.color_intrinsic is not None:
            print("  color_intrinsic (fx fy cx cy): "
                  f"{rec.color_intrinsic[0,0]:.2f} {rec.color_intrinsic[1,1]:.2f} "
                  f"{rec.color_intrinsic[0,2]:.2f} {rec.color_intrinsic[1,2]:.2f}")
        if rec.depth_intrinsic is not None:
            print("  depth_intrinsic (fx fy cx cy): "
                  f"{rec.depth_intrinsic[0,0]:.2f} {rec.depth_intrinsic[1,1]:.2f} "
                  f"{rec.depth_intrinsic[0,2]:.2f} {rec.depth_intrinsic[1,2]:.2f}")

        f0 = rec.frames[0]
        if f0.color_bgr is not None:
            print(f"  color[0]      : {f0.color_bgr.shape} {f0.color_bgr.dtype}")
        if f0.depth_meters is not None:
            d = f0.depth_meters
            valid = d > 0
            dmin = d[valid].min() if np.any(valid) else float("nan")
            dmax = d[valid].max() if np.any(valid) else float("nan")
            dmean = d[valid].mean() if np.any(valid) else float("nan")
            print(f"  depth[0]      : {d.shape} {d.dtype} "
                  f"min={dmin:.3f} max={dmax:.3f} mean={dmean:.3f} (m)")

        # --- preview dump (frame 0) ---
        if f0.color_bgr is not None:
            cv2.imwrite(os.path.join(PREVIEW_DIR, f"{dev}_color0.jpg"), f0.color_bgr)
        if f0.depth_meters is not None:
            cv2.imwrite(os.path.join(PREVIEW_DIR, f"{dev}_depth0.png"),
                        _depth_to_visu(f0.depth_meters))
        print(f"  -> preview written to {PREVIEW_DIR}")
        print("-" * 72)

    # --- round-trip write test on device0 ---
    src = recordings[devices[0]]
    if os.path.isdir(ROUNDTRIP_DIR):
        shutil.rmtree(ROUNDTRIP_DIR)
    print(f"round-trip: re-saving {devices[0]} -> {ROUNDTRIP_DIR}")
    src.save(ROUNDTRIP_DIR)

    print("round-trip: reloading and comparing depth ...")
    rt = Recording.load(ROUNDTRIP_DIR, load_images=True)
    assert len(rt.frames) == len(src.frames), "frame count mismatch after round-trip"
    assert len(rt.imu) == len(src.imu), "imu count mismatch after round-trip"

    max_abs_err = 0.0
    for a, b in zip(src.frames, rt.frames):
        if a.depth_meters is None or b.depth_meters is None:
            continue
        err = float(np.max(np.abs(a.depth_meters - b.depth_meters)))
        if err > max_abs_err:
            max_abs_err = err
    print(f"  frames: {len(rt.frames)}  imu: {len(rt.imu)}")
    print(f"  max per-pixel depth error after round-trip: {max_abs_err:.2e} (m)")
    print(f"  depth dtype on reload: {rt.frames[0].depth_meters.dtype}")

    # Note: saved depth is float32, so reload gives float32; original was
    # promoted to float64 in load(). The ~1e-7 error is just the f32 rounding.
    assert max_abs_err < 1e-5, "round-trip depth error too large"
    print("  round-trip OK")

    print("=" * 72)
    print("done. previews in demo_out/preview/, round-trip in demo_out/roundtrip/")


if __name__ == "__main__":
    main()
