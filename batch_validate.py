#!/usr/bin/env python3
"""Run the complete Podcast-ready pipeline against archived episode pairs."""

import argparse
import gc
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ducking_core import engine as ducking


DEFAULT_EPISODES = [
    "Agte",
    "Allende",
    "Delecourt",
    "Ferraz",
    "GRao",
    "Iyoha",
    "Lowe",
    "MRao",
    "Weigel",
]

PODCAST_SETTINGS = {
    "gain_enabled": False,
    "gain_db": 0,
    "comp_enabled": True,
    "comp_threshold": -24,
    "comp_ratio": 2.0,
    "comp_attack": 10,
    "comp_release": 150,
    "lufs_enabled": True,
    "lufs_target": -18,
    "limiter_enabled": True,
    "limiter_ceiling": -1,
    "master_enabled": True,
}


def find_source(folder, role):
    """Find one raw host or guest WAV while ignoring prior app outputs."""
    matches = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
        and role in path.stem.lower()
        and "processed" not in path.stem.lower()
        and "master" not in path.stem.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one raw {role} WAV in {folder}, found {len(matches)}."
        )
    return matches[0]


def plain_values(value):
    """Convert NumPy values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {key: plain_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain_values(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def limiter_diagnostics(gain):
    """Summarize how often a stem limiter reduces level audibly."""
    reduction_db = -20 * np.log10(np.maximum(gain, 1e-12))
    return {
        "limited_over_1_pct": float(np.mean(reduction_db > 1.0) * 100),
        "limited_over_3_pct": float(np.mean(reduction_db > 3.0) * 100),
        "max_reduction_db": float(np.max(reduction_db)),
    }


def verify_with_ffmpeg(path):
    """Measure the rendered master independently with FFmpeg EBU R128."""
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-filter_complex",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summaries = completed.stderr.split("Summary:")
    summary = summaries[-1]
    loudness = re.search(r"I:\s+(-?\d+(?:\.\d+)?) LUFS", summary)
    loudness_range = re.search(r"LRA:\s+(-?\d+(?:\.\d+)?) LU", summary)
    true_peak = re.search(r"Peak:\s+(-?\d+(?:\.\d+)?) dBFS", summary)
    if not loudness or not loudness_range or not true_peak:
        raise RuntimeError("FFmpeg did not return its expected EBU R128 summary.")
    lufs = float(loudness.group(1))
    lra = float(loudness_range.group(1))
    peak_db = float(true_peak.group(1))
    return {
        "integrated_lufs": lufs,
        "loudness_range_lu": lra,
        "true_peak_db": peak_db,
        "passed": abs(lufs - PODCAST_SETTINGS["lufs_target"]) <= 1.0
        and peak_db <= PODCAST_SETTINGS["limiter_ceiling"] + 0.1,
    }


def validate_episode(root, episode, model, vad_helper):
    """Process one episode in a temporary directory and return all diagnostics."""
    started = time.monotonic()
    folder = root / episode
    host_path = find_source(folder, "host")
    guest_path = find_source(folder, "guest")
    sr, host, host_dtype = ducking.load_wav(host_path)
    guest_sr, guest, guest_dtype = ducking.load_wav(guest_path)
    if sr != guest_sr or len(host) != len(guest):
        raise ValueError("The raw host and guest tracks are not synchronized.")
    if sr != 48000:
        raise ValueError(f"Expected 48 kHz sources, found {sr} Hz.")
    if not np.all(np.isfinite(host)) or not np.all(np.isfinite(guest)):
        raise ValueError("A source contains non-finite audio samples.")

    host_mono = ducking.get_mono(host)
    guest_mono = ducking.get_mono(guest)
    host_regions = ducking.get_speech_regions(
        model,
        vad_helper,
        ducking.resample_to_16k(host_mono, sr),
        threshold=0.5,
    )
    guest_regions = ducking.get_speech_regions(
        model,
        vad_helper,
        ducking.resample_to_16k(guest_mono, sr),
        threshold=0.5,
    )
    env_host, env_guest, detection = ducking.build_validated_ducking_envelopes(
        host_mono,
        guest_mono,
        sr,
        host_regions,
        guest_regions,
        fade_ms=150,
        duck_db=-15,
        dominance_db=3,
        require_two_speakers=True,
    )
    ducking_check = ducking.validate_ducking_stage(
        host, guest, env_host, env_guest, sr, duck_db=-15
    )
    if not ducking_check["passed"]:
        raise RuntimeError(f"Ducking gate failed: {ducking_check['checks']}")

    with tempfile.TemporaryDirectory(prefix=f"ducking-{episode.lower()}-") as output:
        _, host_data = ducking.process_track_audio(
            host,
            sr,
            host_dtype,
            env_host,
            host_path,
            output,
            PODCAST_SETTINGS,
            lambda _value: None,
            lambda _text: None,
        )
        _, guest_data = ducking.process_track_audio(
            guest,
            sr,
            guest_dtype,
            env_guest,
            guest_path,
            output,
            PODCAST_SETTINGS,
            lambda _value: None,
            lambda _text: None,
        )
        mix = ducking.validate_mix_stage(
            host_data["output_audio"],
            guest_data["output_audio"],
            env_host,
            env_guest,
            sr,
        )
        if not mix["passed"]:
            raise RuntimeError(f"Mix gate failed: {mix['checks']}")

        stem_limiting = {
            "host": limiter_diagnostics(host_data["limiter_gain"]),
            "guest": limiter_diagnostics(guest_data["limiter_gain"]),
        }
        dense_stems = [
            role
            for role, metrics in stem_limiting.items()
            if metrics["limited_over_1_pct"] > 3.0
        ]
        if dense_stems:
            raise RuntimeError(
                "Stem limiter exceeded the 3% activity gate: " + ", ".join(dense_stems)
            )

        master, mastering = ducking.build_podcast_master(
            host_data["output_audio"],
            guest_data["output_audio"],
            sr,
            target_lufs=PODCAST_SETTINGS["lufs_target"],
            true_peak_ceiling_db=-1,
            speaker_a_mask=mix["speaker_a_mask"],
            speaker_b_mask=mix["speaker_b_mask"],
        )
        master_path = Path(output) / f"{episode.lower()}_mastered.wav"
        ducking.save_wav(master_path, sr, master, np.dtype("float32"))
        independent = verify_with_ffmpeg(master_path)
        if not independent["passed"]:
            raise RuntimeError(f"Independent master check failed: {independent}")

        host_peak_db = 20 * np.log10(np.max(np.abs(host_data["output_audio"])) + 1e-12)
        guest_peak_db = 20 * np.log10(
            np.max(np.abs(guest_data["output_audio"])) + 1e-12
        )

    mix_summary = {
        key: value
        for key, value in mix.items()
        if key not in ("speaker_a_mask", "speaker_b_mask")
    }
    return plain_values(
        {
            "episode": episode,
            "passed": True,
            "duration_seconds": len(host) / sr,
            "elapsed_seconds": time.monotonic() - started,
            "detection": detection,
            "ducking": ducking_check,
            "stem_peak_db": {"host": host_peak_db, "guest": guest_peak_db},
            "stem_limiting": stem_limiting,
            "mix": mix_summary,
            "master": mastering,
            "ffmpeg": independent,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "episodes", nargs="*", default=DEFAULT_EPISODES, help="Episode folder names"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/mikey/Documents/Academic/UCSD/Backstory"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model, vad_helper = ducking.load_vad_model()
    results = []
    for episode in args.episodes:
        print(f"START {episode}", flush=True)
        try:
            result = validate_episode(args.root, episode, model, vad_helper)
        except Exception as error:
            result = {
                "episode": episode,
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
        gc.collect()

        if args.report:
            args.report.write_text(json.dumps(results, indent=2) + "\n")

    failures = [result for result in results if not result["passed"]]
    print(f"SUMMARY {len(results) - len(failures)}/{len(results)} passed", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
