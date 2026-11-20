"""Tests for the Second Life / OpenSim BVH converter (tools/sl_bvh.py).

The sample files are written by RGB MoCap's own BVH writer
(mocap.source_bvh.write_channels), so they have the exact layout of a
"Generic / Blender" export: 52 Mixamo-named bones, metres, Y up.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sl_bvh  # noqa: E402
from mocap.mixamo_rig import BODY_NAMES, BODY_PARENTS  # noqa: E402
from mocap.source_bvh import write_channels  # noqa: E402

INCH = 0.0254
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def skeleton():
    names = list(BODY_NAMES)
    parents = list(BODY_PARENTS)
    rest = np.zeros((52, 3))
    rest[0] = [0, 0.95, 0]
    for j in range(1, 22):
        side = -1 if "Right" in names[j] else 1
        rest[j] = rest[parents[j]] + [0.05 * side, 0.1, 0]
    for side, hand in (("Left", 20), ("Right", 21)):
        for finger in FINGERS:
            parent = hand
            for k in (1, 2, 3):
                names.append(f"{side}Hand{finger}{k}")
                parents.append(parent)
                rest[len(names) - 1] = rest[parent] + [0.02, 0, 0]
                parent = len(names) - 1
    ends = np.tile([0.0, 0.05, 0.0], (52, 1))
    return names, np.array(parents), rest, ends


def write_sample(path, frames=5, fps=30.0, tpose=False, names=None):
    default_names, parents, rest, ends = skeleton()
    names = names or default_names
    rng = np.random.default_rng(7)
    roots = np.cumsum(np.full((frames, 3), 0.01), axis=0) + [0.2, 0.9, -0.3]
    angles = rng.uniform(-40, 40, (frames, 52, 3))
    if tpose:
        roots = np.vstack([roots[:1], roots])
        angles = np.vstack([np.zeros((1, 52, 3)), angles])
    orders = ["ZXY" if j % 2 else "XYZ" for j in range(52)]
    write_channels(path, names, parents, rest, ends, roots, angles, orders, fps)
    return roots, angles


def read(path):
    text = Path(path).read_text(encoding="ascii")
    head, motion = text.split("MOTION\n")
    lines = motion.strip().splitlines()
    frames = int(lines[0].split(":")[1])
    rows = [list(map(float, line.split())) for line in lines[2:]]
    joints = [line.split()[1] for line in head.splitlines() if line.split()[:1] in (["ROOT"], ["JOINT"])]
    offsets = [list(map(float, line.split()[1:])) for line in head.splitlines() if line.split()[:1] == ["OFFSET"]]
    return dict(head=head, frames=frames, rows=np.array(rows), joints=joints, offsets=np.array(offsets))


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "take.bvh"
        self.out = self.dir / "take_sl.bvh"

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self, *args):
        return sl_bvh.main([str(self.src), str(self.out), *args])

    def test_renames_every_bone_to_an_sl_joint(self):
        write_sample(self.src)
        self.assertEqual(self.convert(), 0)
        joints = read(self.out)["joints"]
        self.assertEqual(joints[0], "mPelvis")
        self.assertEqual(len(joints), 52)
        self.assertTrue(all(name.startswith("m") for name in joints), joints)
        for expected in ("mTorso", "mSpine3", "mChest", "mNeck", "mHead",
                         "mCollarLeft", "mShoulderLeft", "mElbowLeft", "mWristLeft",
                         "mHipRight", "mKneeRight", "mAnkleRight", "mFootRight",
                         "mHandThumb1Left", "mHandPinky3Right"):
            self.assertIn(expected, joints)

    def test_rotations_are_copied_unchanged(self):
        _, angles = write_sample(self.src, tpose=True)
        self.convert()
        source, result = read(self.src), read(self.out)
        np.testing.assert_array_equal(source["rows"][:, 3:], result["rows"][:, 3:])
        self.assertEqual(source["head"].count("rotation"), result["head"].count("rotation"))

    def test_metres_become_inches(self):
        write_sample(self.src, tpose=True)
        self.convert()
        source, result = read(self.src), read(self.out)
        np.testing.assert_allclose(result["rows"][:, :3], source["rows"][:, :3] / INCH, rtol=1e-6)
        np.testing.assert_allclose(result["offsets"], source["offsets"] / INCH, rtol=1e-6, atol=1e-6)

    def test_centimetres_become_inches(self):
        write_sample(self.src, tpose=True)
        self.convert("--units", "cm")
        source, result = read(self.src), read(self.out)
        np.testing.assert_allclose(result["rows"][:, :3], source["rows"][:, :3] / 2.54, rtol=1e-6)

    def test_hip_scale_changes_positions_only(self):
        write_sample(self.src, tpose=True)
        self.convert("--hip-scale", "0.5")
        source, result = read(self.src), read(self.out)
        np.testing.assert_allclose(result["rows"][:, :3], source["rows"][:, :3] / INCH * 0.5, rtol=1e-6)
        np.testing.assert_allclose(result["offsets"], source["offsets"] / INCH, rtol=1e-6, atol=1e-6)

    def test_adds_a_tpose_reference_frame_when_missing(self):
        roots, _ = write_sample(self.src, frames=5)
        self.convert()
        result = read(self.out)
        self.assertEqual(result["frames"], 6)
        self.assertEqual(len(result["rows"]), 6)
        np.testing.assert_array_equal(result["rows"][0, 3:], 0)
        np.testing.assert_allclose(result["rows"][0, :3], roots[0] / INCH, rtol=1e-6)

    def test_keeps_an_existing_tpose_frame(self):
        write_sample(self.src, frames=5, tpose=True)
        self.convert()
        self.assertEqual(read(self.out)["frames"], 6)

    def test_too_long_is_refused(self):
        # The viewer plays (frames - 2) * frame_time seconds; 60 s at most.
        write_sample(self.src, frames=62, fps=1.0, tpose=True)
        self.assertNotEqual(self.convert(), 0)
        self.assertFalse(self.out.exists())

    def test_exactly_sixty_seconds_is_accepted(self):
        write_sample(self.src, frames=61, fps=1.0, tpose=True)
        self.assertEqual(self.convert(), 0)

    def test_start_and_length_cut_a_section(self):
        roots, angles = write_sample(self.src, frames=100, fps=10.0, tpose=True)
        self.assertEqual(self.convert("--start", "2", "--length", "3"), 0)
        result = read(self.out)
        # Reference frame plus 3 s * 10 fps of motion starting at motion frame 20.
        self.assertEqual(result["frames"], 31)
        source = read(self.src)["rows"]
        np.testing.assert_array_equal(result["rows"][1:, 3:], source[21:51, 3:])
        np.testing.assert_array_equal(result["rows"][0, 3:], 0)
        # The reference frame stands where the cut section starts.
        np.testing.assert_allclose(result["rows"][0, :3], source[21, :3] / INCH, rtol=1e-6)

    def test_biped_profile_is_refused(self):
        names = skeleton()[0]
        biped = {"Hips": "Hips", "LeftUpLeg": "LeftHip", "RightUpLeg": "RightHip", "Spine": "Chest"}
        write_sample(self.src, names=[biped.get(n, n) for n in names])
        self.assertNotEqual(self.convert(), 0)
        self.assertFalse(self.out.exists())

    def test_existing_output_is_not_overwritten(self):
        write_sample(self.src)
        self.out.write_text("keep me", encoding="ascii")
        self.assertNotEqual(self.convert(), 0)
        self.assertEqual(self.out.read_text(encoding="ascii"), "keep me")
        self.assertEqual(self.convert("--force"), 0)
        self.assertEqual(read(self.out)["joints"][0], "mPelvis")

    def test_default_output_name(self):
        write_sample(self.src)
        self.assertEqual(sl_bvh.main([str(self.src)]), 0)
        self.assertTrue((self.dir / "take_sl.bvh").exists())

    def test_hips_far_from_start_warn(self):
        write_sample(self.src, frames=3, tpose=True)
        text = self.src.read_text(encoding="ascii").splitlines()
        last = text[-1].split()
        last[0] = "7.0"  # 7 m away, beyond the viewer's 5 m clamp
        text[-1] = " ".join(last)
        self.src.write_text("\n".join(text) + "\n", encoding="ascii")
        report = sl_bvh.convert(self.src.read_text(encoding="ascii"))
        self.assertTrue(any("5 m" in w for w in report.warnings), report.warnings)


if __name__ == "__main__":
    unittest.main()
