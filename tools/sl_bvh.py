"""Make an RGB MoCap BVH ready to upload to Second Life or OpenSim.

Takes a BVH saved with the "Generic / Blender" BVH profile and writes a copy
that the SL viewer's BVH uploader (also used by Firestorm and other OpenSim
viewers) accepts:

* Bones get SL skeleton names. The viewer only accepts a root called "hip" or
  "mPelvis", and it silently drops bones it does not know.
* Positions go from metres (or centimetres) to inches, the unit the viewer
  reads. Without this, hip movement plays about 39 times too small.
* The first frame is a T-pose. The viewer skips frame 0 and measures the hips
  from it, so a missing T-pose frame is added at the first frame's hip
  position.
* The result must play for 60 seconds at most; --start and --length cut a
  section out of a longer take.

Rotations are copied unchanged. RGB MoCap's rest pose (arms straight out,
legs straight down, Y up, facing +Z) is the pose the viewer expects, and the
viewer reads each bone's own rotation order.

RGB MoCap has three spine bones and SL's main chain has two (mTorso, mChest).
The middle one goes to mSpine3, a spare SL spine bone that sits at chest
height, so the three bends add up the same way.

Only the Python standard library is used:

    python tools\\sl_bvh.py take.bvh                      (writes take_sl.bvh)
    python tools\\sl_bvh.py take.bvh dance.bvh --start 12 --length 30
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

INCHES_PER_UNIT = {"m": 1 / 0.0254, "cm": 1 / 2.54}
MAX_SECONDS = 60.0  # viewer limit (MAX_ANIM_DURATION)
UNPLAYED_FRAMES = 2  # viewer plays (frames - 2) * frame time
MAX_HIP_METRES = 5.0  # viewer clamps hip offsets to 5 m
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
BIPED_NAMES = {"LeftHip", "RightHip", "Chest", "Chest2", "LeftKnee", "LeftAnkle", "LeftCollar"}


def bone_map() -> dict[str, str]:
    """RGB MoCap (Mixamo-style) bone name -> SL skeleton joint name."""
    names = {"Hips": "mPelvis", "Spine": "mTorso", "Spine1": "mSpine3", "Spine2": "mChest",
             "Neck": "mNeck", "Head": "mHead"}
    for side in ("Left", "Right"):
        names.update({
            side + "Shoulder": "mCollar" + side, side + "Arm": "mShoulder" + side,
            side + "ForeArm": "mElbow" + side, side + "Hand": "mWrist" + side,
            side + "UpLeg": "mHip" + side, side + "Leg": "mKnee" + side,
            side + "Foot": "mAnkle" + side, side + "ToeBase": "mFoot" + side,
        })
        for finger in FINGERS:
            for k in (1, 2, 3):
                names[f"{side}Hand{finger}{k}"] = f"mHand{finger}{k}{side}"
    return names


class BvhError(ValueError):
    """The file can't be converted; the message says why in plain words."""


@dataclass
class Result:
    text: str
    frames: int
    seconds: float
    warnings: list[str] = field(default_factory=list)


def _number(token: str) -> float:
    try:
        value = float(token)
    except ValueError:
        raise BvhError(f"'{token}' is not a number") from None
    if not math.isfinite(value):
        raise BvhError("the file contains an infinite or missing number")
    return value


def _fmt(value: float) -> str:
    return f"{value:.9f}"


def convert(text: str, units: str = "m", hip_scale: float = 1.0,
            start: float = 0.0, length: float | None = None) -> Result:
    if units not in INCHES_PER_UNIT:
        raise BvhError(f"unknown unit '{units}'")
    if not hip_scale > 0:
        raise BvhError("--hip-scale must be above 0")
    to_inches = INCHES_PER_UNIT[units]
    to_metres = {"m": 1.0, "cm": 0.01}[units]
    mapping = bone_map()
    warnings: list[str] = []

    lines = text.splitlines()
    try:
        motion_at = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration:
        raise BvhError("no MOTION section; this is not a complete BVH file") from None
    if not lines or lines[0].strip() != "HIERARCHY":
        raise BvhError("the file does not start with HIERARCHY")

    # --- hierarchy: rename bones, scale offsets, record channel layout ---
    header: list[str] = []
    names: list[str] = []
    unknown: list[str] = []
    channels: list[tuple[int, str]] = []  # (joint index, channel name) per column
    for line in lines[:motion_at]:
        indent = line[: len(line) - len(line.lstrip())]
        tokens = line.split()
        if tokens[:1] in (["ROOT"], ["JOINT"]):
            if len(tokens) != 2:
                raise BvhError(f"bad joint line: {line.strip()}")
            kind, name = tokens
            if (kind == "ROOT") != (not names):
                raise BvhError("the file must have exactly one ROOT, and it must come first")
            if name in BIPED_NAMES:
                raise BvhError("this BVH uses the 3ds Max Biped profile; export it again "
                               "with the 'Generic / Blender' BVH profile")
            if kind == "ROOT" and name != "Hips":
                raise BvhError(f"expected the root bone 'Hips' from an RGB MoCap export, found '{name}'")
            names.append(name)
            if name not in mapping:
                unknown.append(name)
            line = f"{indent}{kind} {mapping.get(name, name)}"
        elif tokens[:1] == ["OFFSET"]:
            if len(tokens) != 4:
                raise BvhError(f"bad OFFSET line: {line.strip()}")
            line = indent + "OFFSET " + " ".join(_fmt(_number(t) * to_inches) for t in tokens[1:])
        elif tokens[:1] == ["CHANNELS"]:
            if not names or len(tokens) < 2 or len(tokens) != int(tokens[1]) + 2:
                raise BvhError(f"bad CHANNELS line: {line.strip()}")
            channels.extend((len(names) - 1, c) for c in tokens[2:])
        header.append(line)
    if not names:
        raise BvhError("the file has no bones")
    if unknown:
        warnings.append(f"{len(unknown)} bone(s) have no SL match and the viewer will ignore them: "
                        + ", ".join(unknown))

    positions = [i for i, (_, c) in enumerate(channels) if c.endswith("position")]
    root_positions = [i for i in positions if channels[i][0] == 0]
    root_rotations = [i for i, (j, c) in enumerate(channels) if j == 0 and c.endswith("rotation")]
    other_rotations = [i for i, (j, c) in enumerate(channels) if j != 0 and c.endswith("rotation")]
    if len(root_positions) != 3:
        raise BvhError("the root bone needs X, Y and Z position channels")

    # --- motion ---
    body = [line for line in lines[motion_at + 1:] if line.strip()]
    if len(body) < 2 or not body[0].startswith("Frames:") or not body[1].startswith("Frame Time:"):
        raise BvhError("the MOTION section needs 'Frames:' and 'Frame Time:' lines")
    count = int(_number(body[0].split(":", 1)[1]))
    frame_time = _number(body[1].split(":", 1)[1])
    if frame_time <= 0:
        raise BvhError("the frame time must be above 0")
    rows = [line.split() for line in body[2:]]
    if len(rows) != count:
        raise BvhError(f"the file says {count} frames but has {len(rows)}")
    for row in rows:
        if len(row) != len(channels):
            raise BvhError(f"a frame has {len(row)} values; the bones need {len(channels)}")
        for token in row:
            _number(token)
    if not rows:
        raise BvhError("the file has no frames")

    has_reference = all(abs(float(rows[0][i])) <= 1e-6 for i in other_rotations)
    reference_root_rotation = [rows[0][i] for i in root_rotations] if has_reference else ["0"] * 3
    motion = rows[1:] if has_reference else rows
    if not has_reference:
        warnings.append("the first frame was not a T-pose, so one was added "
                        "(tick 'Add T-pose before motion' in RGB MoCap to get it there)")

    fps = 1 / frame_time
    first = round(start * fps)
    if start < 0 or first >= len(motion):
        raise BvhError(f"--start {start:g} s is past the end of the motion "
                       f"({len(motion) * frame_time:.2f} s)")
    if length is not None:
        if length <= 0:
            raise BvhError("--length must be above 0")
        motion = motion[first:first + max(1, round(length * fps))]
    else:
        motion = motion[first:]
    if not motion:
        raise BvhError("no frames left to export")

    frames = len(motion) + 1
    seconds = max(frames - UNPLAYED_FRAMES, 1) * frame_time
    if seconds > MAX_SECONDS + 1e-6:
        raise BvhError(f"the motion plays for {seconds:.2f} s; the viewer accepts {MAX_SECONDS:g} s at most. "
                       "Cut a section with --start and --length (for example --length 60)")

    reference = ["0"] * len(channels)
    for i in positions:
        reference[i] = motion[0][i]
    for i, value in zip(root_rotations, reference_root_rotation):
        reference[i] = value
    out_rows = [reference, *motion]

    start_hips = [float(reference[i]) * to_metres * hip_scale for i in root_positions]
    farthest = 0.0
    scaled_rows = []
    for row in out_rows:
        row = list(row)
        hips = [float(row[i]) * to_metres * hip_scale for i in root_positions]
        farthest = max(farthest, max(abs(a - b) for a, b in zip(hips, start_hips)))
        for i in positions:
            row[i] = _fmt(float(row[i]) * to_inches * hip_scale)
        scaled_rows.append(" ".join(row))
    if farthest > MAX_HIP_METRES:
        warnings.append(f"the hips move up to {farthest:.1f} m from where they start; "
                        f"the viewer stops them at {MAX_HIP_METRES:g} m")

    text_out = "\n".join([*header, "MOTION", f"Frames: {frames}", body[1].strip(), *scaled_rows]) + "\n"
    return Result(text=text_out, frames=frames, seconds=seconds, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Make an RGB MoCap 'Generic / Blender' BVH ready to upload to Second Life or OpenSim.")
    parser.add_argument("input", type=Path, help="BVH file exported by RGB MoCap")
    parser.add_argument("output", type=Path, nargs="?", help="where to write it (default: INPUT_sl.bvh)")
    parser.add_argument("--units", choices=sorted(INCHES_PER_UNIT), default="m",
                        help="units of the input file (RGB MoCap writes metres)")
    parser.add_argument("--hip-scale", type=float, default=1.0,
                        help="scale hip movement, e.g. 0.8 for a shorter avatar (default 1)")
    parser.add_argument("--start", type=float, default=0.0, help="start this many seconds into the motion")
    parser.add_argument("--length", type=float, help="keep this many seconds (60 at most)")
    parser.add_argument("--force", action="store_true", help="replace the output file if it exists")
    args = parser.parse_args(argv)

    output = args.output or args.input.with_name(args.input.stem + "_sl.bvh")
    try:
        if output.resolve() == args.input.resolve():
            raise BvhError("the output would replace the input file; give a different name")
        text = args.input.read_text(encoding="ascii")
        result = convert(text, args.units, args.hip_scale, args.start, args.length)
        with open(output, "w" if args.force else "x", encoding="ascii", newline="\n") as f:
            f.write(result.text)
    except FileExistsError:
        print(f"error: {output} already exists; add --force to replace it", file=sys.stderr)
        return 1
    except (BvhError, OSError, UnicodeDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"Wrote {output}: {result.frames} frames, {result.seconds:.2f} s of motion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
