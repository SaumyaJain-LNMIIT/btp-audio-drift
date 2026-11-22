"""Audio loader — UrbanSound8K via soundata.

Downloads the dataset once to ``data_home``; subsequent runs skip the download.
Every waveform is resampled to a single target sample rate before it leaves the
loader, so MFCCs downstream are computed on a consistent sampling grid
(UrbanSound8K clips have mixed native sample rates).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np

logger = logging.getLogger("audio_drift")


class AudioLoader:
    """Load UrbanSound8K clips through the soundata API."""

    def __init__(self, data_home: str = "data/raw_audio") -> None:
        """Initialise the loader.

        Args:
            data_home: Directory where the dataset is stored / will be downloaded.
        """
        import soundata  # lazy: keeps this module importable without the dataset lib
        self.data_home = str(Path(data_home).resolve())
        self.dataset = soundata.initialize("urbansound8k", data_home=self.data_home)

    # ------------------------------------------------------------------
    # Download (idempotent — skips if data already present)
    # ------------------------------------------------------------------
    def download(self) -> None:
        """Download UrbanSound8K if not already present."""
        logger.info("Ensuring UrbanSound8K is downloaded to %s …", self.data_home)
        self.dataset.download()
        logger.info("Dataset ready.")

    # ------------------------------------------------------------------
    # Waveform preprocessing (resample + fixed length) — pure & testable
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess_waveform(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
        max_duration: float,
    ) -> np.ndarray:
        """Convert to mono, resample to ``target_sr``, pad/truncate to length.

        Args:
            audio: 1-D (mono) or 2-D (channels-first) waveform.
            orig_sr: The waveform's native sample rate.
            target_sr: Desired sample rate (e.g. 22,050).
            max_duration: Target duration in seconds.

        Returns:
            A 1-D float array of length ``int(max_duration * target_sr)`` at
            ``target_sr``.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        if orig_sr != target_sr:
            audio = librosa.resample(
                audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
            )

        target_len = int(max_duration * target_sr)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Load all clips
    # ------------------------------------------------------------------
    def load_all_clips(
        self,
        max_duration: float = 4.0,
        sample_rate: int = 22050,
    ) -> list[dict[str, Any]]:
        """Load all clips and return a list of metadata dicts.

        Each dict contains ``clip_id``, ``audio`` (resampled mono waveform),
        ``sr`` (== ``sample_rate``), ``orig_sr`` (native rate), ``label`` and
        ``fold``.

        Args:
            max_duration: Maximum clip duration in seconds (pad/truncate).
            sample_rate: Target sample rate every clip is resampled to.
        """
        clip_ids = self.dataset.clip_ids
        clips_data: list[dict[str, Any]] = []

        logger.info("Loading %d clips (resampling to %d Hz) …", len(clip_ids), sample_rate)
        for cid in clip_ids:
            try:
                clip = self.dataset.clip(cid)

                audio_signal, sr = clip.audio
                if audio_signal is None:
                    continue
                orig_sr = int(sr)

                audio_signal = self.preprocess_waveform(
                    audio_signal, orig_sr, sample_rate, max_duration
                )

                label = clip.tags.labels[0] if clip.tags and clip.tags.labels else "unknown"
                fold = int(clip.fold) if hasattr(clip, "fold") and clip.fold is not None else 0

                clips_data.append(
                    {
                        "clip_id": cid,
                        "audio": audio_signal,
                        "sr": sample_rate,
                        "orig_sr": orig_sr,
                        "label": label,
                        "fold": fold,
                    }
                )
            except Exception as exc:
                logger.warning("Skipping clip %s: %s", cid, exc)

        logger.info("Successfully loaded %d clips.", len(clips_data))
        return clips_data