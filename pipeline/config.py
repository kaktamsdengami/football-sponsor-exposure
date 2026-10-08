"""
Run configuration.

A single serialisable object carries every setting. It is written verbatim into
the run manifest, which makes each execution reproducible -- and lets an API
receive the same structure as JSON later without touching the pipeline.
"""

from dataclasses import dataclass, asdict, field


@dataclass
class RunConfig:
    # --- input ---
    video: str
    out_root: str = "output_videos"
    run_name: str | None = None          # defaults to run_<timestamp>

    # --- sampling ---
    # 2 img/s is plenty: LED creatives rotate roughly every 30 s and we only need
    # sub-second granularity on exposure time.
    sample_fps: float = 2.0
    max_frames: int | None = None        # for quick trials

    # --- advertising band ---
    n_rows: int = 3                      # rows in the reflowed band
    upscale: float = 2.0                 # magnification before reflowing
    jpeg_quality: int = 92               # board text must stay legible

    # Also writes overlay/<frame>.jpg: the original frame with the extracted
    # region outlined. Purely a visual check on what gets sent to the API.
    debug_overlay: bool = False

    # --- segmentation (pass 0) ---
    # Denser than detect: cut boundaries want sub-second precision, and the pass
    # is cheap (everything runs on a 480px downscale).
    segment_fps: float = 6.0
    min_gap: int = 3                     # samples between two accepted cuts
    min_purity: float = 0.60             # label agreement needed to route "auto"
    # A shot only stays on the automatic path if the band actually reads well.
    # The label alone is not enough: a zoomed duel or a high overhead angle is
    # still "wide_play" but the board is an out-of-focus wall, or barely in
    # frame. probe_band full-res frames per candidate shot decide it.
    probe_band: int = 5
    band_conf_thr: float = 0.55
    max_bad_frac: float = 0.25   # share of probed frames allowed to read badly
    per_shot: int = 2                    # frames offered per shot for review
    qa_frac: float = 0.05                # share of "auto" shots sampled for
                                         # human spot-checking
    seed: int = 0                        # makes the QA sample reproducible

    # Shot kinds detect runs over. A far-touchline board can only appear on a
    # wide or medium shot; close-ups and graphics never have one. Empty = all.
    detect_shot_types: tuple = ("wide_play", "medium")

    # --- broadcast graphics ---
    # Blank the streamer's burned-in overlays (watermark, score bug) before
    # anything reads the frame. They carry real brands -- a streaming watermark and a betting-brand score bug
    # on this broadcast -- that are on screen almost the whole match and are not
    # inventory the club sells. The regions are found from the video's own
    # temporal statistics, not hardcoded; see brand_reader/overlay_mask.py.
    mask_overlays: bool = True
    overlay_probe_frames: int = 96       # frames sampled to find the overlays

    # --- geometry (passed straight through to band_image) ---
    band_kwargs: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
