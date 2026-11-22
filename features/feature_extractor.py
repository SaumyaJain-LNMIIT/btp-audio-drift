"""Feature extractor — MFCC + Δ + ΔΔ with privacy enforcement.

Raw audio references are discarded after feature computation.

Normalisation
-------------
To avoid train/stream leakage, scaling is now a *separate* step from extraction:

* ``extract_batch(..., scale=False)`` returns RAW features plus metadata.
* ``fit_scaler(train_features)`` fits a ``StandardScaler`` on the training
  partition only; ``transform(features)`` applies it to any split.
* ``save_scaler`` / ``load_scaler`` persist the fitted parameters.

``extract_batch(..., scale=True)`` keeps the legacy fit-on-everything behaviour
so existing callers are unaffected until they are migrated to the split-aware
flow. Prefer ``scale=False`` + ``fit_scaler`` in new code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("audio_drift")


class FeatureExtractor:
    """Extract MFCC-based features from raw audio waveforms."""

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        use_deltas: bool = True,
        normalize: bool = True,
    ) -> None:
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_deltas = use_deltas
        self.normalize = normalize
        self._scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # Single waveform → feature vector (raw, unscaled)
    # ------------------------------------------------------------------
    def extract_single(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract a fixed-length raw feature vector from one waveform."""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        features = [np.mean(mfcc, axis=1)]
        if self.use_deltas:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features.append(np.mean(delta, axis=1))
            features.append(np.mean(delta2, axis=1))
        return np.concatenate(features)

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------
    def extract_batch(
        self,
        clips: list[dict[str, Any]],
        save_path: str | Path | None = None,
        scale: bool = True,
        return_meta: bool = False,
    ):
        """Extract features for all clips and **delete raw audio** (privacy).

        Args:
            clips: Clip dicts from ``AudioLoader.load_all_clips()``.
            save_path: Optional directory to cache features + metadata.
            scale: If True (legacy), fit a scaler on ALL rows and transform.
                If False, return raw features (use ``fit_scaler`` afterwards).
            return_meta: If True, also return a metadata dict (clip_ids, folds,
                orig_srs).

        Returns:
            ``(features, labels_encoded, label_names[, meta])``.
        """
        feature_list: list[np.ndarray] = []
        label_list: list[str] = []
        clip_ids: list[str] = []
        folds: list[int] = []
        orig_srs: list[int] = []

        for clip in clips:
            feature_list.append(self.extract_single(clip["audio"], clip["sr"]))
            label_list.append(clip["label"])
            clip_ids.append(clip.get("clip_id", ""))
            folds.append(int(clip.get("fold", 0)))
            orig_srs.append(int(clip.get("orig_sr", clip.get("sr", 0))))
            clip["audio"] = None  # privacy: drop raw audio reference

        raw = np.stack(feature_list, axis=0).astype(np.float32)

        unique_labels = sorted(set(label_list))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        labels_encoded = np.array([label_to_idx[l] for l in label_list], dtype=np.int64)

        if scale:
            # Legacy fit-on-everything path (kept for backward compatibility).
            self._scaler = StandardScaler().fit(raw)
            features = self._scaler.transform(raw).astype(np.float32)
        else:
            features = raw

        logger.info(
            "Extracted features: shape=%s, labels=%d classes, scaled=%s",
            features.shape, len(unique_labels), scale,
        )

        meta = {
            "clip_ids": np.array(clip_ids),
            "folds": np.array(folds, dtype=np.int64),
            "orig_srs": np.array(orig_srs, dtype=np.int64),
        }

        if save_path is not None:
            self._save(save_path, features, labels_encoded, unique_labels, meta, scaled=scale)

        if return_meta:
            return features, labels_encoded, unique_labels, meta
        return features, labels_encoded, unique_labels

    # ------------------------------------------------------------------
    # Leakage-free scaling API
    # ------------------------------------------------------------------
    def fit_scaler(self, train_features: np.ndarray) -> StandardScaler:
        """Fit the scaler on the TRAINING partition only."""
        self._scaler = StandardScaler().fit(train_features)
        return self._scaler

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler to any split (call ``fit_scaler`` first)."""
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted; call fit_scaler() first.")
        return self._scaler.transform(features).astype(np.float32)

    def save_scaler(self, path: str | Path) -> None:
        """Persist the fitted scaler parameters as ``.npz``."""
        if self._scaler is None:
            raise RuntimeError("No fitted scaler to save.")
        np.savez(path, mean=self._scaler.mean_, scale=self._scaler.scale_)

    def load_scaler(self, path: str | Path) -> StandardScaler:
        """Load scaler parameters previously saved with :meth:`save_scaler`."""
        d = np.load(path)
        scaler = StandardScaler()
        scaler.mean_ = d["mean"]
        scaler.scale_ = d["scale"]
        scaler.var_ = d["scale"] ** 2
        scaler.n_features_in_ = len(d["mean"])
        self._scaler = scaler
        return scaler

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(
        self,
        save_path: str | Path,
        features: np.ndarray,
        labels: np.ndarray,
        label_names: list[str],
        meta: dict[str, np.ndarray],
        scaled: bool,
    ) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path / "features.npz",
            features=features,
            labels=labels,
            label_names=np.array(label_names),
            clip_ids=meta["clip_ids"],
            folds=meta["folds"],
            orig_srs=meta["orig_srs"],
        )
        config = {
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "use_deltas": self.use_deltas,
            "feature_dim": int(features.shape[1]),
            "scaled_in_place": bool(scaled),
        }
        with open(save_path / "feature_config.json", "w") as f:
            json.dump(config, f, indent=2)
        if self._scaler is not None:
            self.save_scaler(save_path / "scaler.npz")
        logger.info("Features + metadata saved to %s", save_path)

    # ------------------------------------------------------------------
    # Load cached features
    # ------------------------------------------------------------------
    @staticmethod
    def load_cached(cache_path: str | Path):
        """Load cached features (backward-compatible 3-tuple)."""
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"], data["label_names"].tolist()

    @staticmethod
    def load_cached_full(cache_path: str | Path) -> dict[str, Any]:
        """Load cached features WITH metadata (clip_ids, folds, orig_srs)."""
        data = np.load(cache_path, allow_pickle=True)
        out = {
            "features": data["features"],
            "labels": data["labels"],
            "label_names": data["label_names"].tolist(),
        }
        for key in ("clip_ids", "folds", "orig_srs"):
            if key in data:
                out[key] = data[key]
        return out

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector."""
        return self.n_mfcc * (3 if self.use_deltas else 1)