---
title: Podcast Mic Ducking
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.55.0"
app_file: streamlit_app.py
pinned: false
---

# Podcast Mic Ducking

Clean up a two-person, two-microphone podcast without flattening the voices.

When two people record in the same room, each microphone captures its owner clearly and the other person faintly. Combining the raw tracks therefore adds a delayed, distant copy of each voice. Ducking identifies which microphone owns the voice at each moment and turns down the other microphone. The result keeps the close voice and reduces the echo-like bleed.

The app has two normal workflows:

- **Natural cleanup** creates two cleaned stems and changes only the microphone bleed.

- **Podcast-ready** creates two cleaned stems and a combined mono master. It also removes low-frequency rumble, balances loudness, applies gentle compression to each stem, and protects the final peaks.

Start with a preset. Change an advanced setting only when you can name the problem you are trying to fix.

## Quick start

1. Choose two synchronized recordings of the same conversation: Speaker A on one microphone and Speaker B on the other.
2. Select **Natural cleanup** if another editor will handle the mix, or **Podcast-ready** if you want a finished spoken-word master.
3. Process the files.
4. Read the quality report. A passing report means the mechanical safety checks passed; it does not replace listening.
5. Listen to several speaker changes, an overlap, the loudest passage, and the quietest passage before publishing.

The two inputs must begin at the same moment, use the same sample rate, and have the same duration. Ducking cannot repair recorder drift or align separately started files.

## Which preset should I use?

| Preset | Output | Processing | Use it when |
|---|---|---|---|
| Natural cleanup (recommended) | Two cleaned stems | Speaker detection, ducking, and smooth fades | You will edit, mix, or master elsewhere; you want the recorded tone and dynamics left alone |
| Podcast-ready | Two cleaned stems and one mono master | Ducking, 65 hertz high-pass filter, loudness staging, gentle 2:1 stem compression, stem peak protection, mono mix, one fixed final gain change, and true-peak limiting | The spoken conversation is ready for a routine master, or you plan to add theme music afterward |
| Custom | Two processed stems | Whatever stem controls are enabled; no combined master | You are diagnosing a specific problem or handing the stems to another mastering workflow |

Only the **Podcast-ready** selection creates a mastered file. Custom can normalize or limit the stems, but it does not sum them into a master.

Natural cleanup uses a −12 decibel duck level and 150 millisecond fades. Podcast-ready uses −15 decibels and the same fade length. Both use a default voice activity detection threshold of 0.5 and a dominance threshold of 3 decibels.

## The mastering approach, in plain language

The app treats correction and dynamics as different jobs.

Gain moves an entire recording up or down by the same amount. If a voice is 4 decibels too quiet, adding 4 decibels preserves the difference between that speaker's quiet and loud words. Compression changes those internal differences: it turns down the louder moments more than the quiet moments. Compression can improve intelligibility when a speaker is genuinely inconsistent, but too much makes a voice sound dense, clipped, or tiring.

Podcast-ready therefore uses three restrained operations:

1. It corrects microphone bleed and routine level differences.
2. It applies gentle 2:1 compression to each voice, where the two microphones can still be controlled separately.
3. It gives the combined mix one fixed gain change and uses a limiter only as a peak safety device.

The combined master has no master-bus compressor. Loud passages are allowed to remain louder than quiet passages. The target is −18 Loudness Units relative to Full Scale (LUFS), with a true-peak ceiling of −1 decibel true peak. This is a normal speech-program delivery level under [Audio Engineering Society guidance](https://aes.org/wp-content/uploads/2024/01/20210924_TD1008_v3.13.pdf), and it leaves enough room for natural speech peaks.

The limiter is the guardrail, not the engine. If it must reduce more than 1 decibel for more than 1 percent of the final program, the app rejects the master. A quieter, more open result is preferable to a nominally louder result produced by sustained limiting.

## What the advanced controls mean

Decibel controls use a logarithmic scale. Zero decibels means no level change. A negative gain makes audio quieter; a more negative duck level creates stronger bleed reduction. For loudness targets, a more negative LUFS value is quieter. For peaks, 0 decibels relative to full scale (dBFS) is the digital maximum, so −1 dBFS leaves 1 decibel of headroom.

### Ducking controls

| Control | Default | What it changes | When to adjust it |
|---|---:|---|---|
| Voice activity detection threshold | 0.5 | How confident the Silero model must be that speech exists | Raise it in small 0.05 steps if room noise repeatedly triggers speech. Lower it if quiet words are missed. Leave it alone when detection passes and the words sound intact. |
| Fade | 150 milliseconds | How gradually a microphone moves between full and ducked level | Lengthen it if transitions sound abrupt or pump. Shorten it cautiously for exceptionally rapid exchanges when too much bleed remains around each turn. |
| Duck level | −12 decibels natural; −15 decibels podcast-ready | How far the non-owning microphone is reduced | Make it 3 decibels more negative if audible bleed remains. Make it less negative if the room tone moves unnaturally or a voice sounds thin around turns. |
| Dominance | 3 decibels | How much louder one close microphone must be before the other is ducked | Raise it if the app chooses a speaker too eagerly or damages overlaps. Lower it if a quieter speaker is rarely assigned to their own microphone. The app retries once with a lower value when it cannot identify both speakers. |

Voice activity detection answers “is there speech?” Dominance answers “which microphone owns it?” The app compares the two microphones in 20 millisecond frames and calibrates the comparison to the recording's actual microphone-level difference. When neither microphone is clearly dominant—including many genuine overlaps—both remain open. Ducking is not source separation, and it should not delete one side of a conversation.

### Gain and loudness controls

| Control | Podcast-ready value | What it changes | When to adjust it |
|---|---:|---|---|
| Manual gain | Off; 0 decibels | Adds the same fixed gain to both stems before compression and normalization | Use only in a custom stem workflow. It cannot correct a difference between the two speakers because the same value is applied to both. |
| Loudness normalization | On | Measures loudness and applies a fixed gain change toward the target | Leave it on for Podcast-ready. Turn it off when another editor will set the levels. |
| LUFS target | −18 LUFS master | Sets the final Podcast-ready master target; Podcast-ready stems are staged separately at −19 LUFS | Lower it to −19 or −20 if unusually sharp peaks cause a limiter failure. A less negative value is louder and consumes more peak headroom. |

LUFS approximates perceived average loudness over time. It is not a peak measurement. Podcast-ready measures each stem during open speech, stages it before compression, and finishes it near −19 LUFS. The two stems are then summed, measured as a complete program, and moved once toward −18 LUFS.

### Compressor controls

| Control | Podcast-ready value | What it means | When to adjust it |
|---|---:|---|---|
| Compressor | On for stems | Reduces level only when the signal crosses the threshold | Turn it off in Custom when you want pure ducking and gain correction. Keep it gentle for ordinary stationary-microphone speech. |
| Threshold | −24 decibels | The level above which compression begins | Raise it toward zero to compress only the loudest moments. Lower it to affect more of the voice. If the voice sounds flattened, raising the threshold is usually the first compressor adjustment. |
| Ratio | 2:1 | Every 2 decibels above the threshold becomes 1 decibel at the output | Stay near 2:1 for routine speech. Higher ratios control peaks more aggressively and change the voice more. |
| Attack | 10 milliseconds | How quickly compression responds after the threshold is crossed | Lengthen it slightly if consonants and natural transients sound dull. Shorten it only when sharp peaks consistently escape. |
| Release | 150 milliseconds | How quickly compression stops after the signal falls below the threshold | Lengthen it if level changes sound nervous or pump. Shorten it if compression stays audible well after a loud word. |

The threshold is measured after the app stages the stem to a predictable level. That makes the default compressor behave similarly across recordings made on the same equipment.

### Limiter controls

| Control | Podcast-ready value | What it changes | When to adjust it |
|---|---:|---|---|
| Limiter | On | Prevents peaks from crossing a ceiling | Leave it on for Podcast-ready. It should catch isolated peaks, not reshape ordinary speech. |
| Limiter ceiling | −1 decibel relative to full scale | Sets the final true-peak ceiling | Keep −1 for routine delivery. A lower ceiling gives more safety but may require more limiting. Do not raise the ceiling to make a failed master pass. |

Podcast-ready cleaned stems are held to −3 decibels relative to full scale or lower, even when the final ceiling is −1. This leaves headroom when the stems are combined. The master limiter runs at four-times oversampling so it can detect inter-sample peaks that an ordinary sample-peak meter can miss.

## When to change a setting

Change one setting at a time, process the full episode again, and compare at the same playback volume. A short problem passage is useful for diagnosis, but the final setting must pass the whole episode.

| What you hear or see | First response |
|---|---|
| Echo-like bleed remains during clean single-speaker passages | Make Duck level 3 decibels more negative |
| Words sound clipped at speaker changes | Lengthen Fade; if the wrong microphone is being chosen, raise Dominance slightly |
| A quiet speaker is seldom recognized | Lower Dominance by 0.5 decibel; lower the voice activity detection threshold by 0.05 only if quiet speech is also being missed |
| Noise is treated as speech | Raise the voice activity detection threshold by 0.05 |
| Overlapping speech loses one person | Raise Dominance slightly so ambiguous frames keep both microphones open; avoid redesigning an otherwise good full-episode setting around one brief interjection |
| A stem sounds flattened | Raise the compressor threshold, reduce the ratio, or disable compression in Custom |
| The final limiter fails its activity gate | Lower the LUFS target by 1 decibel or repair an isolated peak manually; inspect the loudest section before changing compression |
| Speaker balance fails | Confirm the two files contain different close microphones. Correct the individual source or stem level in an editor; shared Manual gain cannot repair speaker imbalance |
| Speaker detection fails completely | Confirm the files are synchronized, equal in duration, and assigned to the correct speakers before changing thresholds |

If the report passes and the episode sounds natural, stop. The numbers are guardrails, not scores to maximize.

## How to read the quality report

The report contains three kinds of information:

- A **measurement** describes the file but has no universal ideal value.

- A **check** compares a measurement with a safety threshold and reports PASS or FAIL.

- An **acceptance gate** blocks the Podcast-ready master when failure could hide a damaged output. The cleaned stems are preserved when a later mastering gate fails.

The desktop report shows the detailed measurements and explicit stem checks described below. The web report is compact: its Speaker A and Speaker B columns show output loudness, output peak, envelope coverage, and duration. When Podcast-ready succeeds, the web table's Master column reports final loudness, true peak, limiter activity, fixed gain, and confirmation that master compression is off. The same blocking gates run in both versions even when the web table does not print every intermediate value.

### Stem measurements

| Report line | What it means | How to interpret it |
|---|---|---|
| LUFS (in → out) | Input loudness for the full source, then output loudness measured during open speech | With Podcast-ready, each output stem should be within 2 loudness units (LU) of −19 LUFS. The input and output use different measurement windows, so the change is useful context, not a literal gain reading. Natural cleanup does not enforce a loudness target. |
| Peak dBFS (in → out) | Highest digital sample before and after processing | It must not exceed 0 dBFS. Podcast-ready stems normally remain at or below −3 dBFS to leave mix headroom. This is a sample peak; the final master receives a stricter true-peak test. |
| Speech regions | Number of speech segments returned by Silero | Informational. These are model segments, not words, sentences, or turns. A high count can simply mean many short pauses. |
| Speech coverage | Percentage of the stem whose ducking envelope is more than half open | The automatic check accepts 5 to 95 percent. Near either extreme can indicate that one microphone was almost never assigned, though a genuinely lopsided interview can also produce an extreme value. |
| Speech time | Total speech marked by Silero on that microphone | Informational. Because both microphones hear both people, it is not a reliable estimate of that person's speaking time. |
| Maximum envelope slope | Largest sample-to-sample change in the ducking gain | Lower is smoother. Values above roughly 0.01 can indicate an abrupt transition, but this line is informational and should be judged by listening. |
| Limiter over 1 dB | Percentage of the stem receiving more than 1 decibel of limiter reduction | Podcast-ready requires no more than 3 percent. A failure means peak protection has become sustained level control. |
| Maximum limiter reduction | Strongest single limiter action anywhere in the stem | Informational. One isolated transient can produce a large maximum while affecting almost none of the episode; locate and listen to it if the number is surprising. |

### Stem checks

| Check | PASS means | What a failure suggests |
|---|---|---|
| No clipping | No output sample exceeded the digital range | Lower manual gain or loudness target and keep the limiter enabled |
| LUFS within target | Normalized stems are within 2 LU of their expected target | The source is too quiet, too peaky, or unable to reach the requested loudness safely |
| Duration preserved | The output contains exactly as many samples as the input | Do not publish; inspect the source format or processing error |
| Speech coverage normal | The envelope is open for 5 to 95 percent of the file | Check the file assignment, synchronization, voice activity detection, and dominance |
| Limiter activity gentle | More than 1 decibel of reduction occurs on no more than 3 percent of the stem | Lower gain or target loudness and inspect isolated peaks |
| Ducking Speaker A or B | Measured ducking reduced that microphone by at least 6 decibels | Confirm that the two tracks contain distinct microphones; then inspect Duck level and Dominance |

Podcast-ready also checks the ducking stage before any gain, compression, or normalization can disguise it. The active microphone must remain within 0.25 decibel of its original level, and the non-owning microphone must reach within 3 decibels of the requested duck amount. At the default −15 decibels, the measured attenuation must be at least 12 decibels.

### Speaker-detection lines

The desktop report adds a short detection summary:

- **A primary and B primary** are the shares of 20 millisecond frames confidently assigned to each close microphone.

- **Both/quiet** is the share left ambiguous. This includes room tone, silence, and overlaps where neither microphone clearly wins.

- **Detection threshold attempts** lists the dominance values tried. Two values mean the first attempt could not identify at least one second for both speakers, so the app retried conservatively.

- **Ducking gate: PASS** means the app identified both microphones and verified the expected attenuation. **BYPASSED safely** means Natural cleanup lacked enough evidence to duck without risking a voice, so it returned unchanged envelopes. Podcast-ready treats the same uncertainty as an error and creates no master.

### Podcast-ready master measurements

| Report line | Acceptance rule | What it tells you |
|---|---|---|
| Integrated loudness | −18 LUFS, within 1 LU | Average program loudness after the final limiter |
| True peak | At or below the selected ceiling, normally −1 decibel true peak | The reconstructed waveform should not overload during playback or encoding |
| Limiter over 1 dB | No more than 1 percent of the program | Final limiting is rare enough to remain a safety operation |
| Fixed final gain | Informational | The one level change used to move the premix toward its target |
| Master compressor | Off | Confirms that the final mix was not dynamically compressed |
| Peak-to-loudness ratio | Informational | The distance between average loudness and the highest true peak; a larger number generally indicates more transient headroom |
| Speaker balance | No more than 3 decibels during isolated speech | The two speakers remain comparably audible after mixing |

The master also has hidden blocking checks for finite sample values, exact duration, clipped samples, and the survival of both speakers in the mono sum. The unmastered mix must contain at least one second assigned to each speaker; each speaker must remain audible; and mixing must not reduce either owning stem by more than 6 decibels in its isolated regions.

When a master gate fails, the error names the failed check. No file is labeled Podcast-ready, but the cleaned stems remain available for inspection or manual repair.

### A worked example

Suppose a Podcast-ready report says that Speaker A moved from −23.4 to −19.2 LUFS, peaked at −3.0 dBFS, and received more than 1 decibel of stem limiting for 0.4 percent of the episode. The master finishes at −18.1 LUFS and −1.3 decibels true peak, with 0.2 percent final limiter activity and 0.7 decibel speaker balance.

That is an ordinary passing result. The stem is within 2 LU of its −19 target, its peak has mix headroom, and its limiter density is below 3 percent. The master is within 1 LU of −18, below the −1 true-peak ceiling, below the 1 percent final-limiter limit, and comfortably inside the 3 decibel speaker-balance limit. The measurements still cannot tell you whether a particular overlap or laugh sounds good, so the listening check remains necessary.

### What PASS does not mean

PASS means the app did not detect a mechanical failure covered by its gates. It cannot decide whether an interview has an awkward edit, whether room tone is aesthetically pleasing, whether a laugh is too loud for the audience, or whether theme music is balanced correctly after you add it.

Before publishing, listen to:

- several ordinary speaker changes;

- at least one genuine overlap;

- the quietest speaker passage;

- laughter or the loudest emphatic passage;

- the beginning and end of the recording; and

- the final program again after adding theme music or making edits.

## Automated tests and the batch report

The project has two different test layers.

Run the unit suite with:

```bash
python3 -m unittest -q test_ducking_app.py test_ducking_core_contracts.py test_ducking_core_engine.py
```

The 19 signal-processing tests use controlled synthetic signals to protect model loading, synchronized-file safety, speaker calibration, smooth envelopes, overlap handling, compression and limiter behavior, stem preservation, speaker balance, and master rejection. Eleven contract tests protect the versioned interface shared with Backstory. Five package tests protect synchronized edits, exact theme markers, media inspection, and an end-to-end synthetic render. A passing unit suite means the code still obeys those defined rules. It does not test a new real recording.

The full acceptance harness runs Podcast-ready on a directory of real episode pairs:

```bash
python3 batch_validate.py --root "/path/to/episode-folders" --report report.json
```

Each episode folder must contain exactly one raw Waveform Audio File Format (WAV) file with `host` in its name and one with `guest` in its name. The files must be synchronized 48 kilohertz recordings. Processed and mastered files are ignored.

The command prints one JavaScript Object Notation (JSON) object per episode and ends with a summary such as `SUMMARY 9/9 passed`. A nonzero exit status means at least one episode failed. The report groups the evidence by processing stage:

| JSON section | Contents | Main gate |
|---|---|---|
| `detection` | Calibrated microphone-ratio center, ratio clusters, primary-speaker shares, ambiguity share, dominance attempts, and bypass status | Both close microphones must be identified for at least one second |
| `ducking` | Attenuation of the non-owning microphone and level change on the owning microphone | Ducking reaches the requested neighborhood while the active microphone changes by no more than 0.25 decibel |
| `stem_peak_db` | Highest sample peak in each cleaned stem | Informational alongside the no-clipping check |
| `stem_limiting` | Percent above 1 and 3 decibels of limiter reduction, plus the maximum reduction | More than 1 decibel of reduction on no more than 3 percent of either stem |
| `mix` | Isolated-speaker duration, stem level, mixed level, retained level, and speaker-survival checks | Both speakers are present, audible, and lose no more than 6 decibels in the sum |
| `master` | Loudness, true peak, fixed gain, limiter density, peak-to-loudness ratio, speaker levels, balance, and all final checks | Every final check passes, including −18 LUFS within 1 LU, ceiling compliance, limiter density, and speaker balance |
| `ffmpeg` | Independent European Broadcasting Union R128 loudness, loudness range, and true peak from the rendered WAV | Independent loudness and true peak agree with the delivery limits |

The numerical field names can be read as follows:

| Field | Meaning |
|---|---|
| `duration_seconds` | Source program length; informational |
| `elapsed_seconds` | Computer processing time, not an audio-quality measurement |
| `ratio_center_db` | The calibrated midpoint between the two microphones' typical level ratios. It need not be zero because preamp gain, microphone sensitivity, and distance can differ. |
| `ratio_clusters_db` | The two typical microphone-ratio groups inferred from the conversation. Distinct clusters support confident ownership detection; their absolute values are not targets. |
| `a_primary_pct`, `b_primary_pct`, `ambiguous_pct` | Fractions assigned to Speaker A, Speaker B, or neither. They should add to approximately 100 percent and do not equal exact speaking-time shares. |
| `speaker_a_attenuation_db`, `speaker_b_attenuation_db`, and the corresponding `preserved_change_db` fields | Gain change while a microphone is ducked and while it owns the voice. A good result is near the requested negative duck level and near 0 decibels, respectively. |
| `limited_over_1_pct`, `limited_over_3_pct`, `max_reduction_db` | How widespread light and stronger limiting were, and the single strongest reduction. Density drives the gate; the maximum helps locate an isolated peak. |
| `speaker_*_stem_db`, `speaker_*_mix_db`, `speaker_*_retained_db` | Isolated-speech level before the sum, after the sum, and the difference. Retained level near 0 means the mono sum preserved that speaker; below −6 fails. |
| `premix_lufs`, `fixed_gain_db`, `integrated_lufs` | Loudness before final gain, the one final gain change, and loudness after limiting. Fixed gain is not compression. |
| `true_peak_db`, `plr_db` | Highest reconstructed peak and peak-to-loudness ratio. True peak has a ceiling; peak-to-loudness ratio is descriptive. |
| `speaker_a_db`, `speaker_b_db`, `speaker_balance_db` | Each speaker's isolated level in the master and their absolute difference. The difference must not exceed 3 decibels. |
| `checks` and `checks_passed` | Individual Boolean gates and their combined result. One false gate makes the stage fail. |

Loudness range in the FFmpeg section is descriptive, not a pass/fail target. A quiet, measured interview and an animated interview should not be forced to have the same loudness range.

## How it works: detailed processing sequence

Podcast-ready processes the recording in this order:

1. Convert each microphone to mono for analysis and resample a copy to 16 kilohertz for Silero voice activity detection.
2. Compare the synchronized microphones in 20 millisecond frames. Calibrate the midpoint between their level-ratio clusters, require the chosen Dominance margin, and remove switches shorter than 100 milliseconds.
3. Smooth the resulting gain envelopes with the selected Fade length. Ambiguous frames leave both microphones open.
4. Verify that both close microphones were identified. Retry once with a lower dominance margin when necessary.
5. Verify the raw ducking operation before any later processing.
6. On each stem, apply a 65 hertz high-pass filter, the ducking envelope, loudness staging, 2:1 compression at a −24 decibel threshold with 10 millisecond attack and 150 millisecond release, fixed-gain normalization to −19 LUFS during open speech, and peak protection at −3 decibels relative to full scale or lower.
7. Verify stem duration, clipping, loudness, speech coverage, and limiter density.
8. Sum the cleaned stems to mono and verify that both speakers survived the mix.
9. Measure the premix, apply one fixed gain change toward −18 LUFS, and run a four-times-oversampled true-peak limiter.
10. Accept the master only if every final gate passes.

Natural cleanup stops after the verified ducking envelope and saves the two cleaned stems. It does not high-pass, normalize, compress, or limit the voices.

## What the app does not do

Ducking is designed for two synchronized close microphones in one room. It does not:

- align tracks that started at different times or drifted apart;

- separate two voices recorded onto one mixed track;

- remove reverberation, broadband noise, mouth clicks, or electrical hum;

- edit content, remove pauses, or repair interrupted words;

- add equalization beyond the Podcast-ready 65 hertz rumble filter;

- add theme music, advertisements, or metadata; or

- replace a final human listen.

## Try it online

Use [mikeyjarrell.com/ducking](https://mikeyjarrell.com/ducking), upload both recordings, select a preset, and download the results. The web version accepts WAV, MPEG-1 Audio Layer III (MP3), Free Lossless Audio Codec (FLAC), MPEG-4 Audio (M4A), Ogg, and Advanced Audio Coding (AAC) inputs. Output is WAV.

The web app processes uploads on a remote server. Use the local app for sensitive recordings or when upload size and connection speed matter.

## Run it on your own computer

Local processing is faster for long recordings and avoids upload limits.

### 1. Install Python

Download Python 3.10 or newer from [python.org](https://www.python.org/downloads/). On Windows, select the installer option that adds Python to your PATH.

### 2. Download the app

Use GitHub's Code button and choose Download ZIP, or run:

```bash
git clone https://github.com/MikeyJarrell/ducking.git
cd ducking
```

### 3. Install the dependencies

```bash
pip install streamlit torch numpy scipy silero-vad soundfile
```

### 4. Run the web interface locally

```bash
streamlit run streamlit_app.py
```

### Alternative: desktop version

The desktop interface accepts WAV files and writes its output beside the inputs or in a selected output directory.

```bash
pip install torch numpy scipy silero-vad
python ducking_app.py
```

### macOS application

Build and install a standalone `/Applications/Ducking.app` with:

```bash
chmod +x make-app.sh
./make-app.sh
```

The build requires the python.org framework build of Python 3.12. Anaconda Python is not a framework build and cannot produce this py2app bundle. The installed application embeds Python, PyTorch, and its dependencies, so it is approximately 600 megabytes and does not need a system Python at runtime.

## Outputs and file formats

The desktop app accepts 16-bit integer, 32-bit integer, and 32-bit or 64-bit floating-point WAV sources. The web app also accepts MP3, FLAC, M4A, OGG, and AAC. WAV is preferred because it avoids lossy decoding before processing.

Cleaned stems use the source sample rate. Integer WAV sources retain their integer width; floating-point stems are written as 32-bit floating point. Their names end in `_processed.wav`. Podcast-ready also creates a mono, 32-bit floating-point WAV whose name ends in `_mastered.wav`.

Adding theme music or making later edits changes program loudness and peaks. Measure the complete program again after those changes.

## Project files

- [ducking_core](ducking_core) is the installable `ducking-core` package. It owns the canonical signal processing, contracts, synchronized edits, theme assembly, media inspection, encoding, and concrete `DuckingCore` service.

- [ducking_app.py](ducking_app.py) contains only the desktop interface and imports its audio behavior from `ducking_core`.

- [streamlit_app.py](streamlit_app.py) contains only the web interface and web-specific upload/download conversion; it imports its processing behavior from `ducking_core`.

- [batch_validate.py](batch_validate.py) runs the private episode regression corpus.

- [test_ducking_app.py](test_ducking_app.py) contains the unit tests.

- [test_ducking_core_contracts.py](test_ducking_core_contracts.py) and [test_ducking_core_engine.py](test_ducking_core_engine.py) protect the shared boundary and package behavior.

- [pyproject.toml](pyproject.toml) and [setup.py](setup.py) build the reusable Python 3.12 wheel. [make-app.sh](make-app.sh) uses the same setup file to build the macOS application.

- [PROJECT_INDEX.md](PROJECT_INDEX.md) records project status.

- [SESSION_LOG.md](SESSION_LOG.md) records major project decisions and handoffs in chronological order.

- [AGENTS.md](AGENTS.md) contains project instructions; [CLAUDE.md](CLAUDE.md) imports them for Claude Code.

Generated environments, application bundles, build directories, and tool caches are local artifacts. `.venv-build`, `build`, `dist`, and `.mypy_cache` are not source documentation or release inputs.

## Technical references

- [Audio Engineering Society TD1008: Recommendations for Loudness of Internet Audio Streaming and On-Demand Distribution](https://aes.org/wp-content/uploads/2024/01/20210924_TD1008_v3.13.pdf)

- [International Telecommunication Union BS.1770-5: Algorithms to measure audio programme loudness and true-peak audio level](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en)

- [European Broadcasting Union R128: Loudness normalisation and permitted maximum level of audio signals](https://tech.ebu.ch/publications/r128)

- [Silero voice activity detector](https://github.com/snakers4/silero-vad)
