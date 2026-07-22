#!/usr/bin/env python3
"""Desktop interface for the shared Ducking audio engine."""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ducking_core.engine import (
    PODCAST_MASTER_TARGET_LUFS,
    apply_gain_db,
    build_podcast_master,
    build_validated_ducking_envelopes,
    format_quality_report,
    get_mono,
    get_speech_regions,
    load_vad_model,
    load_wav,
    master_output_path,
    mix_to_mono,
    process_track_audio,
    resample_to_16k,
    save_wav,
    validate_ducking_stage,
    validate_mix_stage,
    validate_track,
)

class DuckingApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Podcast Mic Ducking")

        # Processing state
        self.processing = False
        self.vad_model = None
        self.vad_utils = None
        self._progress_value = 0
        self._status_text = "Ready"
        self._done = False
        self._error = None
        self._result = None
        self._report = ""

        self._build_gui()

        # Force a layout pass so the window sizes itself to fit all widgets
        # before we lock the size — otherwise resizable(False, False) freezes
        # the window at its initial near-zero geometry, and clicks below the
        # visible-but-not-claimed area are lost.
        self.update_idletasks()
        self.resizable(False, False)

        # Force window to become the frontmost foreground app on first Map event.
        # macOS sometimes fails to activate shell-wrapped Python GUIs properly,
        # which breaks click-to-focus on widgets.
        self.bind("<Map>", self._on_first_map, add="+")
        self._mapped_once = False

    def _on_first_map(self, event):
        if self._mapped_once:
            return
        self._mapped_once = True
        try:
            from AppKit import NSApp, NSApplicationActivationPolicyRegular

            NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            NSApp.activateIgnoringOtherApps_(True)
        except Exception:
            pass
        self.lift()
        self.focus_force()

    def _build_gui(self):
        """Build all the GUI widgets."""
        main = ttk.Frame(self, padding=15)
        main.grid(row=0, column=0, sticky="nsew")

        row = 0

        # --- File Selection ---
        ttk.Label(main, text="Audio Files", font=("", 13, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        # Speaker A file picker
        ttk.Label(main, text="Speaker A:").grid(row=row, column=0, sticky="w")
        self.file_a_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.file_a_var, width=45).grid(
            row=row, column=1, padx=5
        )
        ttk.Button(main, text="Browse...", command=lambda: self._browse_file("a")).grid(
            row=row, column=2
        )
        row += 1

        # Speaker B file picker
        ttk.Label(main, text="Speaker B:").grid(row=row, column=0, sticky="w")
        self.file_b_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.file_b_var, width=45).grid(
            row=row, column=1, padx=5
        )
        ttk.Button(main, text="Browse...", command=lambda: self._browse_file("b")).grid(
            row=row, column=2
        )
        row += 1

        # Output directory picker
        ttk.Label(main, text="Output dir:").grid(
            row=row, column=0, sticky="w", pady=(5, 0)
        )
        self.output_dir_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.output_dir_var, width=45).grid(
            row=row, column=1, padx=5, pady=(5, 0)
        )
        ttk.Button(main, text="Browse...", command=self._browse_output).grid(
            row=row, column=2, pady=(5, 0)
        )
        row += 1

        # --- Ducking Settings ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(main, text="Ducking", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        duck_frame = ttk.Frame(main)
        duck_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=3)

        ttk.Label(duck_frame, text="VAD threshold:").grid(row=0, column=0, sticky="w")
        self.vad_thresh_var = tk.StringVar(value="0.5")
        ttk.Entry(duck_frame, textvariable=self.vad_thresh_var, width=6).grid(
            row=0, column=1, padx=(5, 15)
        )

        ttk.Label(duck_frame, text="Fade:").grid(row=0, column=2, sticky="w")
        self.fade_var = tk.StringVar(value="150")
        ttk.Entry(duck_frame, textvariable=self.fade_var, width=6).grid(
            row=0, column=3, padx=(5, 0)
        )
        ttk.Label(duck_frame, text="ms").grid(row=0, column=4, padx=(2, 15))

        ttk.Label(duck_frame, text="Duck:").grid(row=0, column=5, sticky="w")
        self.duck_db_var = tk.StringVar(value="-12")
        ttk.Entry(duck_frame, textvariable=self.duck_db_var, width=6).grid(
            row=0, column=6, padx=(5, 0)
        )
        ttk.Label(duck_frame, text="dB").grid(row=0, column=7, padx=(2, 15))

        ttk.Label(duck_frame, text="Dominance:").grid(row=0, column=8, sticky="w")
        self.dominance_db_var = tk.StringVar(value="3")
        ttk.Entry(duck_frame, textvariable=self.dominance_db_var, width=6).grid(
            row=0, column=9, padx=(5, 0)
        )
        ttk.Label(duck_frame, text="dB").grid(row=0, column=10, padx=(2, 0))
        row += 1

        # --- Processing Chain ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(main, text="Processing Chain", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        preset_frame = ttk.Frame(main)
        preset_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(3, 5))
        ttk.Label(preset_frame, text="Preset:").grid(row=0, column=0, sticky="w")
        self.preset_var = tk.StringVar(value="Natural cleanup (recommended)")
        preset = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=("Natural cleanup (recommended)", "Podcast-ready", "Custom"),
            width=29,
            state="readonly",
        )
        preset.grid(row=0, column=1, padx=5)
        preset.bind("<<ComboboxSelected>>", self._apply_preset)
        row += 1

        self.preset_help = ttk.Label(
            main,
            text="Turns down mic bleed while preserving the recorded voice.",
            foreground="#555555",
        )
        self.preset_help.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 5))
        row += 1

        # Gain
        gain_frame = ttk.Frame(main)
        gain_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.gain_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(gain_frame, text="Gain:", variable=self.gain_enabled).grid(
            row=0, column=0, sticky="w"
        )
        self.gain_db_var = tk.StringVar(value="0")
        ttk.Entry(gain_frame, textvariable=self.gain_db_var, width=6).grid(
            row=0, column=1, padx=5
        )
        ttk.Label(gain_frame, text="dB").grid(row=0, column=2)
        row += 1

        # Compressor
        comp_frame = ttk.Frame(main)
        comp_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.comp_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            comp_frame, text="Compressor:", variable=self.comp_enabled
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(comp_frame, text="Thresh:").grid(row=0, column=1, padx=(10, 0))
        self.comp_thresh_var = tk.StringVar(value="-24")
        ttk.Entry(comp_frame, textvariable=self.comp_thresh_var, width=5).grid(
            row=0, column=2, padx=2
        )
        ttk.Label(comp_frame, text="dB").grid(row=0, column=3)

        ttk.Label(comp_frame, text="Ratio:").grid(row=0, column=4, padx=(10, 0))
        self.comp_ratio_var = tk.StringVar(value="2.0")
        ttk.Entry(comp_frame, textvariable=self.comp_ratio_var, width=4).grid(
            row=0, column=5, padx=2
        )
        ttk.Label(comp_frame, text=":1").grid(row=0, column=6)
        row += 1

        # Compressor attack/release (indented under compressor)
        comp_frame2 = ttk.Frame(main)
        comp_frame2.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 2))
        ttk.Label(comp_frame2, text="").grid(row=0, column=0, padx=55)  # indent

        ttk.Label(comp_frame2, text="Attack:").grid(row=0, column=1)
        self.comp_attack_var = tk.StringVar(value="10")
        ttk.Entry(comp_frame2, textvariable=self.comp_attack_var, width=5).grid(
            row=0, column=2, padx=2
        )
        ttk.Label(comp_frame2, text="ms").grid(row=0, column=3)

        ttk.Label(comp_frame2, text="Release:").grid(row=0, column=4, padx=(10, 0))
        self.comp_release_var = tk.StringVar(value="150")
        ttk.Entry(comp_frame2, textvariable=self.comp_release_var, width=5).grid(
            row=0, column=5, padx=2
        )
        ttk.Label(comp_frame2, text="ms").grid(row=0, column=6)
        row += 1

        # Limiter
        lim_frame = ttk.Frame(main)
        lim_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.limiter_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(lim_frame, text="Limiter:", variable=self.limiter_enabled).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(lim_frame, text="Ceiling:").grid(row=0, column=1, padx=(10, 0))
        self.limiter_ceil_var = tk.StringVar(value="-1.0")
        ttk.Entry(lim_frame, textvariable=self.limiter_ceil_var, width=6).grid(
            row=0, column=2, padx=2
        )
        ttk.Label(lim_frame, text="dBFS").grid(row=0, column=3)
        row += 1

        # LUFS normalization
        lufs_frame = ttk.Frame(main)
        lufs_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.lufs_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(lufs_frame, text="LUFS norm:", variable=self.lufs_enabled).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(lufs_frame, text="Target:").grid(row=0, column=1, padx=(10, 0))
        self.lufs_target_var = tk.StringVar(value="-18")
        ttk.Entry(lufs_frame, textvariable=self.lufs_target_var, width=6).grid(
            row=0, column=2, padx=2
        )
        ttk.Label(lufs_frame, text="LUFS").grid(row=0, column=3)
        row += 1

        # --- Process Button ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10
        )
        row += 1

        self.process_btn = ttk.Button(
            main, text="Process", command=self._start_processing
        )
        self.process_btn.grid(row=row, column=0, columnspan=3, sticky="ew", ipady=5)
        row += 1

        # Status label
        self.status_label = ttk.Label(main, text="Ready")
        self.status_label.grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
        row += 1

        # Progress bar
        self.progress_bar = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress_bar.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(5, 0)
        )

    def _apply_preset(self, _event=None):
        """Fill the controls with safe settings for the selected workflow."""
        preset = self.preset_var.get()
        if preset == "Natural cleanup (recommended)":
            self.fade_var.set("150")
            self.duck_db_var.set("-12")
            self.gain_enabled.set(False)
            self.gain_db_var.set("0")
            self.comp_enabled.set(False)
            self.limiter_enabled.set(False)
            self.lufs_enabled.set(False)
            self.lufs_target_var.set("-18")
            self.preset_help.config(
                text="Turns down mic bleed while preserving the recorded voice."
            )
        elif preset == "Podcast-ready":
            self.fade_var.set("150")
            self.duck_db_var.set("-15")
            self.gain_enabled.set(False)
            self.gain_db_var.set("0")
            self.comp_enabled.set(True)
            self.comp_thresh_var.set("-24")
            self.comp_ratio_var.set("2.0")
            self.comp_attack_var.set("10")
            self.comp_release_var.set("150")
            self.limiter_enabled.set(True)
            self.limiter_ceil_var.set("-1.0")
            self.lufs_enabled.set(True)
            self.lufs_target_var.set("-18")
            self.preset_help.config(
                text="Creates cleaned stems and a mastered mono mix ready for editing."
            )
        else:
            self.preset_help.config(text="Uses the settings shown below.")

    # --- File browsing callbacks ---

    def _browse_file(self, which):
        """Open file picker for speaker A or B."""
        path = filedialog.askopenfilename(
            title=f"Select Speaker {'A' if which == 'a' else 'B'} audio",
            filetypes=[("WAV files", "*.wav *.WAV"), ("All files", "*.*")],
        )
        if path:
            if which == "a":
                self.file_a_var.set(path)
            else:
                self.file_b_var.set(path)

            # Auto-fill output directory from first file selected
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(path))

    def _browse_output(self):
        """Open folder picker for output directory."""
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    # --- Settings ---

    def _get_settings(self):
        """Read all GUI settings into a dictionary."""
        return {
            "vad_threshold": float(self.vad_thresh_var.get()),
            "fade_ms": float(self.fade_var.get()),
            "duck_db": float(self.duck_db_var.get()),
            "dominance_db": float(self.dominance_db_var.get()),
            "gain_enabled": self.gain_enabled.get(),
            "gain_db": float(self.gain_db_var.get()),
            "comp_enabled": self.comp_enabled.get(),
            "comp_threshold": float(self.comp_thresh_var.get()),
            "comp_ratio": float(self.comp_ratio_var.get()),
            "comp_attack": float(self.comp_attack_var.get()),
            "comp_release": float(self.comp_release_var.get()),
            "limiter_enabled": self.limiter_enabled.get(),
            "limiter_ceiling": float(self.limiter_ceil_var.get()),
            "lufs_enabled": self.lufs_enabled.get(),
            "lufs_target": float(self.lufs_target_var.get()),
            "master_enabled": self.preset_var.get() == "Podcast-ready",
        }

    # --- Validation ---

    def _validate(self):
        """Check that inputs are valid before processing."""
        if not self.file_a_var.get():
            messagebox.showerror("Error", "Please select Speaker A audio file.")
            return False
        if not self.file_b_var.get():
            messagebox.showerror("Error", "Please select Speaker B audio file.")
            return False
        if not os.path.isfile(self.file_a_var.get()):
            messagebox.showerror("Error", f"File not found: {self.file_a_var.get()}")
            return False
        if not os.path.isfile(self.file_b_var.get()):
            messagebox.showerror("Error", f"File not found: {self.file_b_var.get()}")
            return False

        try:
            settings = self._get_settings()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid setting value: {e}")
            return False

        valid_ranges = [
            ("VAD threshold", settings["vad_threshold"], 0.1, 0.9),
            ("Fade", settings["fade_ms"], 10, 1000),
            ("Duck level", settings["duck_db"], -40, 0),
            ("Dominance", settings["dominance_db"], 0, 20),
            ("Compressor ratio", settings["comp_ratio"], 1, 20),
            ("Compressor attack", settings["comp_attack"], 1, 200),
            ("Compressor release", settings["comp_release"], 10, 2000),
            ("Limiter ceiling", settings["limiter_ceiling"], -12, 0),
            ("Loudness target", settings["lufs_target"], -30, -10),
        ]
        for label, value, minimum, maximum in valid_ranges:
            if not minimum <= value <= maximum:
                messagebox.showerror(
                    "Error", f"{label} must be between {minimum} and {maximum}."
                )
                return False

        # Default output dir to same folder as Speaker A
        if not self.output_dir_var.get():
            self.output_dir_var.set(os.path.dirname(self.file_a_var.get()))

        return True

    # --- Processing ---

    def _start_processing(self):
        """Validate inputs and launch processing in a background thread."""
        if not self._validate():
            return

        self.processing = True
        self._done = False
        self._error = None
        self.process_btn.config(state="disabled")
        self.progress_bar["value"] = 0

        settings = self._get_settings()

        thread = threading.Thread(
            target=self._run_processing,
            args=(settings,),
            daemon=True,  # Thread dies when app closes
        )
        thread.start()

        # Start polling the thread's progress every 100 ms
        self.after(100, self._check_progress)

    def _run_processing(self, settings):
        """Process both tracks with cross-track ducking. Runs in a background thread."""
        try:
            # Load the VAD model bundled with the installed silero-vad package.
            self._update_status("Loading VAD model...")
            if self.vad_model is None:
                self.vad_model, self.vad_utils = load_vad_model()
            self._update_progress(3)

            output_dir = self.output_dir_var.get()
            file_a = self.file_a_var.get()
            file_b = self.file_b_var.get()

            # Step 1: Load both tracks
            self._update_status(f"Loading {os.path.basename(file_a)}...")
            sr_a, audio_a, dtype_a = load_wav(file_a)
            self._update_progress(6)

            self._update_status(f"Loading {os.path.basename(file_b)}...")
            sr_b, audio_b, dtype_b = load_wav(file_b)
            self._update_progress(9)

            if sr_a != sr_b:
                raise ValueError(
                    f"The files use different sample rates ({sr_a} and {sr_b} Hz). "
                    "Export both tracks with the same sample rate and try again."
                )
            if len(audio_a) != len(audio_b):
                difference = abs(len(audio_a) - len(audio_b)) / sr_a
                raise ValueError(
                    f"The files have different durations by {difference:.2f} seconds. "
                    "Export synchronized tracks with the same start and end points."
                )

            # Step 2: Run VAD on both tracks
            self._update_status("Running VAD on Speaker A...")
            mono_a = get_mono(audio_a)
            audio_16k_a = resample_to_16k(mono_a, sr_a)
            regions_a = get_speech_regions(
                self.vad_model,
                self.vad_utils,
                audio_16k_a,
                threshold=settings["vad_threshold"],
            )
            self._update_progress(16)

            self._update_status("Running VAD on Speaker B...")
            mono_b = get_mono(audio_b)
            audio_16k_b = resample_to_16k(mono_b, sr_b)
            regions_b = get_speech_regions(
                self.vad_model,
                self.vad_utils,
                audio_16k_b,
                threshold=settings["vad_threshold"],
            )
            self._update_progress(23)

            # Step 3: Build cross-track ducking envelopes
            # This compares RMS levels between tracks to determine who's speaking
            self._update_status("Computing cross-track ducking envelopes...")
            (
                envelope_a,
                envelope_b,
                ducking_diagnostics,
            ) = build_validated_ducking_envelopes(
                mono_a,
                mono_b,
                sr_a,
                regions_a,
                regions_b,
                fade_ms=settings["fade_ms"],
                duck_db=settings["duck_db"],
                dominance_db=settings["dominance_db"],
                require_two_speakers=settings.get("master_enabled", False),
            )

            if ducking_diagnostics["ducking_bypassed"]:
                ducking_stage = {
                    "passed": False,
                    "decision": "bypassed because speaker detection was uncertain",
                }
            else:
                ducking_stage = validate_ducking_stage(
                    audio_a,
                    audio_b,
                    envelope_a,
                    envelope_b,
                    sr_a,
                    settings["duck_db"],
                )
                if not ducking_stage["passed"]:
                    if settings.get("master_enabled", False):
                        failed = [
                            name
                            for name, passed in ducking_stage["checks"].items()
                            if not passed
                        ]
                        raise ValueError(
                            "Ducking failed its pre-master checks "
                            f"({', '.join(failed)}). No podcast-ready master was "
                            "created."
                        )
                    envelope_a = np.ones(len(mono_a), dtype=np.float32)
                    envelope_b = np.ones(len(mono_b), dtype=np.float32)
                    ducking_diagnostics["ducking_bypassed"] = True
                    ducking_stage[
                        "decision"
                    ] = "bypassed after attenuation check failed"
            self._update_progress(28)

            # Step 4: Process each track through the audio chain
            self._update_status("Processing Speaker A...")

            def progress_a(frac):
                self._update_progress(28 + frac * 30)

            path_a, data_a = process_track_audio(
                audio_a,
                sr_a,
                dtype_a,
                envelope_a,
                file_a,
                output_dir,
                settings,
                progress_a,
                self._update_status,
            )
            data_a["speech_regions"] = regions_a

            self._update_status("Processing Speaker B...")

            def progress_b(frac):
                self._update_progress(58 + frac * 30)

            path_b, data_b = process_track_audio(
                audio_b,
                sr_b,
                dtype_b,
                envelope_b,
                file_b,
                output_dir,
                settings,
                progress_b,
                self._update_status,
            )
            data_b["speech_regions"] = regions_b

            report_a = validate_track(
                data_a["input_audio"],
                data_a["output_audio"],
                data_a["sr"],
                data_a["envelope"],
                data_a["speech_regions"],
                settings,
                data_a["limiter_gain"],
            )
            report_b = validate_track(
                data_b["input_audio"],
                data_b["output_audio"],
                data_b["sr"],
                data_b["envelope"],
                data_b["speech_regions"],
                settings,
                data_b["limiter_gain"],
            )

            master_path = None
            master_diagnostics = None
            if settings.get("master_enabled", False):
                failed_stem_checks = [
                    f"Speaker {speaker}: {name}"
                    for speaker, report in (("A", report_a), ("B", report_b))
                    for name, passed in report["checks"].items()
                    if not passed
                ]
                if failed_stem_checks:
                    raise ValueError(
                        "A cleaned stem failed its podcast-ready checks "
                        f"({', '.join(failed_stem_checks)}). The cleaned stems "
                        "were preserved, but no file was labeled podcast-ready."
                    )
                mix_diagnostics = validate_mix_stage(
                    data_a["output_audio"],
                    data_b["output_audio"],
                    envelope_a,
                    envelope_b,
                    sr_a,
                )
                if not mix_diagnostics["passed"]:
                    failed = [
                        name
                        for name, passed in mix_diagnostics["checks"].items()
                        if not passed
                    ]
                    raise ValueError(
                        "The unmastered mix failed its speaker-presence checks "
                        f"({', '.join(failed)}). The cleaned stems were preserved, "
                        "but no file was labeled podcast-ready."
                    )
                self._update_status("Building mastered mono mix...")
                self._update_progress(88)
                master_audio, master_diagnostics = build_podcast_master(
                    data_a["output_audio"],
                    data_b["output_audio"],
                    sr_a,
                    target_lufs=settings["lufs_target"],
                    true_peak_ceiling_db=settings["limiter_ceiling"],
                    speaker_a_mask=mix_diagnostics["speaker_a_mask"],
                    speaker_b_mask=mix_diagnostics["speaker_b_mask"],
                )
                master_path = master_output_path(file_a, file_b, output_dir)
                save_wav(master_path, sr_a, master_audio, np.dtype("float32"))

            # Run quality validation
            self._update_status("Running quality checks...")
            self._update_progress(91)

            self._update_progress(95)

            # Check ducking effectiveness per track
            ducking_a_db = validate_ducking(
                data_a["input_audio"],
                apply_gain_envelope(data_a["input_audio"], data_a["envelope"]),
                data_a["envelope"],
                data_a["sr"],
            )
            ducking_b_db = validate_ducking(
                data_b["input_audio"],
                apply_gain_envelope(data_b["input_audio"], data_b["envelope"]),
                data_b["envelope"],
                data_b["sr"],
            )
            self._update_progress(98)

            # Format the quality report
            report_text = format_quality_report(
                report_a, report_b, ducking_a_db, ducking_b_db, path_a, path_b
            )
            clusters = ducking_diagnostics["ratio_clusters_db"]
            calibration = "automatic" if clusters is not None else "neutral"
            report_text += (
                f"\n\nSpeaker detection: {calibration} calibration, "
                f"A primary {ducking_diagnostics['a_primary_pct']:.1f}%, "
                f"B primary {ducking_diagnostics['b_primary_pct']:.1f}%, "
                f"both/quiet {ducking_diagnostics['ambiguous_pct']:.1f}%."
            )
            attempts = ", ".join(
                f"{value:g}" for value in ducking_diagnostics["dominance_attempts_db"]
            )
            report_text += f"\nDetection threshold attempts: {attempts} dB."
            if ducking_diagnostics["ducking_bypassed"]:
                report_text += (
                    "\nDucking gate: BYPASSED safely; the app did not have enough "
                    "evidence to attenuate either microphone."
                )
            else:
                report_text += "\nDucking gate: PASS before mastering."
            if master_path is not None:
                report_text += (
                    f"\n\nPodcast-ready master: PASS"
                    f"\nMaster: {os.path.basename(master_path)}"
                    f"\nIntegrated loudness: "
                    f"{master_diagnostics['integrated_lufs']:.1f} LUFS"
                    f"\nTrue peak: {master_diagnostics['true_peak_db']:.1f} dBFS"
                    f"\nLimiter >1 dB: "
                    f"{master_diagnostics['limited_over_1_pct']:.1f}% of file"
                    f"\nFixed final gain: "
                    f"{master_diagnostics['fixed_gain_db']:.1f} dB"
                    f"\nMaster compressor: off"
                    f"\nPeak-to-loudness ratio: "
                    f"{master_diagnostics['plr_db']:.1f} dB"
                    f"\nSpeaker balance: "
                    f"{master_diagnostics['speaker_balance_db']:.1f} dB"
                )

            self._update_progress(100)
            self._update_status("Done!")
            self._result = (path_a, path_b, master_path)
            self._report = report_text
            self._done = True

        except Exception as e:
            self._error = str(e)
            self._update_status(f"Error: {e}")
            self._done = True

    def _update_progress(self, value):
        """Set progress value (called from background thread)."""
        self._progress_value = value

    def _update_status(self, text):
        """Set status text (called from background thread)."""
        self._status_text = text

    def _check_progress(self):
        """Poll background thread and update GUI (runs on main thread)."""
        self.progress_bar["value"] = self._progress_value
        self.status_label.config(text=self._status_text)

        if self._done:
            # Processing finished — re-enable button and show result
            self.process_btn.config(state="normal")
            self.processing = False

            if self._error:
                messagebox.showerror("Error", self._error)
            else:
                self._show_report(self._report)
            return

        # Keep polling
        self.after(100, self._check_progress)

    def _show_report(self, report_text):
        """Show quality report in a scrollable window with monospace text."""
        win = tk.Toplevel(self)
        win.title("Quality Report")
        win.resizable(True, True)

        # Monospace text widget so the table columns align
        text = tk.Text(
            win,
            wrap="none",
            font=("Courier", 12),
            width=72,
            height=28,
            padx=10,
            pady=10,
        )
        text.insert("1.0", report_text)
        text.config(state="disabled")  # Read-only
        text.grid(row=0, column=0, sticky="nsew")

        # Scrollbar
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.config(yscrollcommand=scrollbar.set)

        # Close button
        ttk.Button(win, text="Close", command=win.destroy).grid(
            row=1, column=0, columnspan=2, pady=10
        )

        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        # Bring to front
        win.lift()
        win.focus_force()


# ============================================================
# SECTION 11: ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Silero VAD recommends single-threaded torch for CPU inference
    torch.set_num_threads(1)

    app = DuckingApp()
    app.mainloop()
