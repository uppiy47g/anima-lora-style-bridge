import hashlib
import json
from pathlib import Path
import re
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "anima-xdna-validated.json"
SCHEMA = ROOT / "docs" / "evidence" / "anima-xdna-evidence.schema.json"


def read_png(path):
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG: {path}")
    offset = 8
    chunks = []
    width = height = color_type = None
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        chunks.append(kind)
        if kind == b"IHDR":
            width, height, _depth, color_type = struct.unpack(">IIBB", data[:10])
        offset += 12 + length
    metadata = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}
    return payload, width, height, color_type, sum(kind in metadata for kind in chunks)


class PublicReleaseTests(unittest.TestCase):
    def test_evidence_manifest_and_assets_are_consistent(self):
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(evidence["schema_version"], schema["properties"]["schema_version"]["const"])
        self.assertFalse(evidence["boundary"]["onnx"])
        self.assertFalse(evidence["boundary"]["vitis_ai_ep"])
        self.assertFalse(evidence["boundary"]["silent_fallback"])
        self.assertEqual(
            evidence["measurements"]["user_observation_20_step"]["status"],
            "observational_non_controlled",
        )
        turbo = evidence["measurements"]["turbo_v1_1"]
        self.assertNotEqual(
            turbo["source_block_fingerprint"],
            turbo["base_block_fingerprint"],
        )
        self.assertLessEqual(
            turbo["chain_28"]["worst_block_normalized_rms_error"],
            turbo["chain_28"]["max_block_gate"],
        )
        self.assertLessEqual(
            turbo["chain_28"]["final_normalized_rms_error"],
            turbo["chain_28"]["final_gate"],
        )
        for artifact in evidence["artifacts"]:
            path = ROOT / artifact["path"]
            payload, width, height, color_type, metadata_entries = read_png(path)
            self.assertEqual(len(payload), artifact["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), artifact["sha256"])
            self.assertEqual((width, height), (artifact["width"], artifact["height"]))
            self.assertEqual(color_type, 2)  # RGB
            self.assertEqual(artifact["mode"], "RGB")
            self.assertEqual(metadata_entries, artifact["metadata_entries"])

    def test_public_examples_parse_and_reference_registered_nodes(self):
        workflow = json.loads(
            (ROOT / "examples" / "anima_xdna_512_api.json").read_text(encoding="utf-8")
        )
        self.assertEqual(workflow["2"]["class_type"], "LoadAttachAnimaXDNAModel")
        self.assertEqual(workflow["3"]["class_type"], "AnimaXDNARuntimeStatus")
        self.assertEqual(
            workflow["2"]["inputs"]["checkpoint"],
            "<MATCHING_BASE_OR_TURBO_SAFETENSORS>",
        )
        launcher = (ROOT / "examples" / "run_comfyui_xdna.ps1").read_text(encoding="utf-8")
        self.assertIn("--cpu", launcher)
        self.assertIn("--extra-model-paths-config", launcher)
        self.assertIn("$env:XRT_DEV_DIR", launcher)

    def test_tracked_public_text_has_no_known_private_paths(self):
        forbidden = (
            re.compile("rgm" + "47", re.IGNORECASE),
            re.compile("session" + "-state", re.IGNORECASE),
            re.compile("Amuse" + "_v3", re.IGNORECASE),
            re.compile("Comfy" + " Desktop", re.IGNORECASE),
            re.compile("Comfy" + "-Desktop", re.IGNORECASE),
        )
        suffixes = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".ps1", ".txt"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part in {".git", ".venv"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertIsNone(pattern.search(text), f"{pattern.pattern} in {path}")


if __name__ == "__main__":
    unittest.main()
