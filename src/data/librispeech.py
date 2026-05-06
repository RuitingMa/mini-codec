"""Fixed-length segment dataset over LibriSpeech FLAC files.

LibriSpeech is natively 16 kHz mono. The ``sample_rate`` constructor argument
is the *target* rate the dataset returns; if it differs from 16 kHz, a
``torchaudio`` ``Resample`` is applied per item. Utterances shorter than the
requested segment length are filtered out at construction time so that
``__getitem__`` is always safe to call.

The class is intentionally minimal: one tensor per item, no labels, no
metadata. Wrap it externally if you need anything more.
"""

from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

# LibriSpeech ships at 16 kHz; this is hard-coded to that contract.
_NATIVE_SAMPLE_RATE = 16000


class LibriSpeechSegments(Dataset):
    """Returns fixed-length mono waveform crops from a LibriSpeech split.

    Each item is a ``torch.float32`` tensor of shape ``[1, segment_samples]``,
    where ``segment_samples = round(sample_rate * segment_seconds)``.

    Parameters
    ----------
    root:
        Directory containing the ``LibriSpeech/<split>/`` tree. Concretely,
        files are expected at ``<root>/LibriSpeech/<split>/<spk>/<chap>/*.flac``
        which is the layout produced by ``torchaudio.datasets.LIBRISPEECH``.
    split:
        One of LibriSpeech's split names, e.g. ``"dev-clean"``.
    sample_rate:
        Target sample rate. If ``!= 16000``, a resampler is applied per item.
    segment_seconds:
        Length of returned crops, in seconds (at the target rate).
    random_crop:
        If True, a random offset is chosen each call. If False, every call
        returns the segment starting at sample 0 (deterministic; useful for
        eval).
    seed:
        Optional RNG seed for ``random_crop``. Reproducibility only — the
        crop order is not stable across PyTorch versions if you also rely on
        ``DataLoader`` shuffling.
    """

    NATIVE_SAMPLE_RATE = _NATIVE_SAMPLE_RATE

    def __init__(
        self,
        root: str | Path,
        split: str = "dev-clean",
        sample_rate: int = 16000,
        segment_seconds: float = 1.0,
        random_crop: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_samples = int(round(self.sample_rate * self.segment_seconds))
        self.random_crop = bool(random_crop)
        self._rng = random.Random(seed)

        # Resampler is built once and reused. None means "no resampling needed".
        if self.sample_rate != _NATIVE_SAMPLE_RATE:
            self._resample: torchaudio.transforms.Resample | None = (
                torchaudio.transforms.Resample(_NATIVE_SAMPLE_RATE, self.sample_rate)
            )
        else:
            self._resample = None

        # An utterance is usable iff it has enough native samples to cover the
        # requested segment after resampling. We compare in native frames so we
        # don't have to load every file.
        min_native_samples = int(
            round(self.segment_samples * _NATIVE_SAMPLE_RATE / self.sample_rate)
        )

        candidates = self._discover(self.root, self.split)
        self.files: list[Path] = []
        for path in candidates:
            # soundfile reads only the file header, so this is cheap even for
            # the full LibriSpeech ~2700-utterance dev-clean split.
            info = sf.info(str(path))
            if info.frames >= min_native_samples:
                self.files.append(path)

        if not self.files:
            raise RuntimeError(
                f"No usable utterances in {self.root}/LibriSpeech/{self.split} "
                f"(need >= {min_native_samples} native samples per file)."
            )

    @staticmethod
    def _discover(root: Path, split: str) -> list[Path]:
        target = root / "LibriSpeech" / split
        if not target.exists():
            raise FileNotFoundError(
                f"{target} does not exist. Download with "
                f"torchaudio.datasets.LIBRISPEECH(root='{root}', "
                f"url='{split}', download=True)"
            )
        return sorted(target.rglob("*.flac"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        # soundfile returns shape [T] for mono or [T, C] for multi-channel,
        # in float32 within [-1, 1]. We use it instead of torchaudio.load to
        # avoid the optional torchcodec backend dependency in torchaudio 2.11.
        wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr != _NATIVE_SAMPLE_RATE:
            raise ValueError(
                f"{path} reports sr={sr}; expected {_NATIVE_SAMPLE_RATE}."
            )

        # always_2d gives [T, C]; transpose to [C, T] for the rest of the pipeline.
        wav = torch.from_numpy(wav_np).transpose(0, 1).contiguous()
        # Force mono. LibriSpeech is mono in practice but defensive doesn't hurt.
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if self._resample is not None:
            wav = self._resample(wav)

        total = wav.shape[1]
        if self.random_crop:
            # randint is inclusive on both ends; we need start in [0, total - segment].
            start = self._rng.randint(0, total - self.segment_samples)
        else:
            start = 0
        return wav[:, start : start + self.segment_samples].contiguous()


def download_librispeech(root: str | Path, split: str = "dev-clean") -> Path:
    """Convenience wrapper around ``torchaudio.datasets.LIBRISPEECH`` that only
    triggers the download. The returned path is the directory the dataset
    class expects in ``root``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torchaudio.datasets.LIBRISPEECH(root=str(root), url=split, download=True)
    return root / "LibriSpeech" / split
