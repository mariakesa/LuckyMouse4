# -*- coding: utf-8 -*-
import os
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

load_dotenv()

# ── Fill this with your choices ─────────────────────────────────────────
CIFAR_HUB_MODELS = [
    "cifar10_resnet56",
    "cifar100_resnet56",
]
EMBEDDING_CACHE_DIR = "/home/maria/Documents/HuggingMouseData/CIFARLogits"
BATCH_SIZE = 256         # lower if you still OOM (e.g., 128 or 64)
USE_AMP = True           # mixed precision on GPU for memory savings
# ────────────────────────────────────────────────────────────────────────

# Make things deterministic(ish) and reduce overhead
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False


# ── Pipeline base ───────────────────────────────────────────────────────
class PipelineStep(ABC):
    @abstractmethod
    def process(self, data):
        pass


# ── Allen API singleton ─────────────────────────────────────────────────
class AllenAPI:
    _instance = None
    _boc = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def boc(self):
        if self._boc is None:
            allen_cache_path = os.environ.get("CAIM_ALLEN_CACHE_PATH")
            if not allen_cache_path:
                raise ValueError("Set CAIM_ALLEN_CACHE_PATH in .env to your Allen cache directory.")
            manifest_path = Path(allen_cache_path) / "brain_observatory_manifest.json"
            self._boc = BrainObservatoryCache(manifest_file=str(manifest_path))
        return self._boc

    def get_boc(self):
        return self.boc


allen_api = AllenAPI()


# ── Step 1: Fetch Allen stimuli ─────────────────────────────────────────
class AllenStimuliFetchStep(PipelineStep):
    SESSION_A = 501704220
    SESSION_B = 501559087
    SESSION_C = 501474098

    def __init__(self, boc):
        self.boc = boc

    def process(self, data):
        if isinstance(data, tuple):
            container_id, session, stimulus = data
            data = {"container_id": container_id, "session": session, "stimulus": stimulus}
        elif data is None:
            data = {}

        raw_data_dct = {
            "natural_movie_one":   self.boc.get_ophys_experiment_data(self.SESSION_A).get_stimulus_template("natural_movie_one"),
            "natural_movie_two":   self.boc.get_ophys_experiment_data(self.SESSION_C).get_stimulus_template("natural_movie_two"),
            "natural_movie_three": self.boc.get_ophys_experiment_data(self.SESSION_A).get_stimulus_template("natural_movie_three"),
            "natural_scenes":      self.boc.get_ophys_experiment_data(self.SESSION_B).get_stimulus_template("natural_scenes"),
        }

        data["raw_data_dct"] = raw_data_dct
        return data


# ── CIFAR utils ─────────────────────────────────────────────────────────
CIFAR10_STATS = {
    "mean": torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1),
    "std":  torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1),
}
CIFAR100_STATS = {
    "mean": torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1),
    "std":  torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1),
}

def infer_dataset_from_name(model_name: str) -> str:
    name = model_name.lower()
    if "cifar100" in name or "cifar-100" in name:
        return "cifar100"
    return "cifar10"

def cifar_stats_for(model_name: str):
    return CIFAR100_STATS if infer_dataset_from_name(model_name) == "cifar100" else CIFAR10_STATS


# ── Step 2: torch.hub CIFAR model → logits (streaming, memory-light) ───
class TorchHubCIFARLogitsStep(PipelineStep):
    def __init__(self, model_names, embedding_cache_dir: str, batch_size: int = 256, use_amp: bool = True):
        self.model_names = list(model_names)
        self.cache_dir = Path(embedding_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self._models = {}

    def _ensure_model(self, model_name: str):
        if model_name not in self._models:
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                model_name,
                pretrained=True
            ).to(self.device)
            model.eval()
            self._models[model_name] = model
        return self._models[model_name]

    def _cache_file_for(self, model_name: str) -> Path:
        return self.cache_dir / f"{model_name}_embeddings_logits.pkl"

    @staticmethod
    def _to_rgb_uint8(batch_np: np.ndarray) -> np.ndarray:
        """
        Input: (B, H, W) or (B, H, W, 1/3), any dtype
        Output: (B, H, W, 3) uint8
        """
        if batch_np.ndim == 3:
            batch_np = batch_np[..., None]
        if batch_np.shape[-1] == 1:
            batch_np = np.repeat(batch_np, 3, axis=-1)

        if batch_np.dtype != np.uint8:
            maxv = float(batch_np.max()) if batch_np.size else 1.0
            if maxv <= 1.5:
                batch_np = (batch_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                batch_np = np.clip(batch_np, 0, 255).astype(np.uint8)
        return batch_np

    def _prep_batch_tensor(self, batch_np: np.ndarray, model_name: str) -> torch.Tensor:
        """
        Convert a small batch of numpy images → normalized tensor on device.
        Output: (B, 3, 32, 32)
        """
        batch_np = self._to_rgb_uint8(batch_np)
        x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0  # (B,3,H,W)
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

        stats = cifar_stats_for(model_name)
        mean = stats["mean"]
        std = stats["std"]
        # normalize in float32 on CPU, then move
        x = (x - mean) / std
        return x.to(self.device, non_blocking=False)

    @torch.no_grad()
    def _logits_for_frames_streaming(self, model_name: str, frames_array: np.ndarray) -> np.ndarray:
        """
        Stream through frames in small batches, never holding the full tensor in memory.
        Returns a dense numpy array of logits (N, C).
        """
        model = self._ensure_model(model_name)

        # Do a tiny dry-run to get C (num classes) without big memory hit
        sample = frames_array[0:1]
        x0 = self._prep_batch_tensor(sample, model_name)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out0 = model(x0)
            if isinstance(out0, (list, tuple)):
                out0 = out0[0]
        num_classes = int(out0.shape[1])
        del x0, out0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        N = frames_array.shape[0]
        logits_np = np.empty((N, num_classes), dtype=np.float32)

        bs = self.batch_size
        for start in range(0, N, bs):
            end = min(start + bs, N)
            batch_np = frames_array[start:end]          # small slice of numpy only
            xb = self._prep_batch_tensor(batch_np, model_name)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = model(xb)
                if isinstance(out, (list, tuple)):
                    out = out[0]
            logits_np[start:end] = out.detach().float().cpu().numpy()

            # Free ASAP
            del xb, out
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return logits_np

    def process(self, data):
        if "raw_data_dct" not in data:
            raise ValueError("Expected 'raw_data_dct' in data. Run AllenStimuliFetchStep first.")

        raw_data_dct = data["raw_data_dct"]
        outputs = {}

        for model_name in self.model_names:
            cache_file = self._cache_file_for(model_name)
            if cache_file.exists():
                print(f"✅ Found cached logits for {model_name} at {cache_file}")
                with open(cache_file, "rb") as f:
                    outputs[model_name] = pickle.load(f)
                continue

            print(f"🔄 No cache for {model_name}. Generating logits...")
            per_stim = {}
            for stim_name, frames in raw_data_dct.items():
                per_stim[stim_name] = self._logits_for_frames_streaming(model_name, frames)

            with open(cache_file, "wb") as f:
                pickle.dump(per_stim, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"📁 Saved logits for {model_name} → {cache_file}")
            outputs[model_name] = per_stim

        data["cifar_logit_files"] = {m: str(self._cache_file_for(m)) for m in self.model_names}
        data["cifar_logits"] = outputs
        return data


# ── Orchestrate ────────────────────────────────────────────────────────
def main():
    if not CIFAR_HUB_MODELS:
        raise RuntimeError("Add at least one model name to CIFAR_HUB_MODELS (e.g., 'cifar10_resnet56').")

    boc = allen_api.get_boc()
    stimuli_step = AllenStimuliFetchStep(boc)
    logits_step = TorchHubCIFARLogitsStep(
        model_names=CIFAR_HUB_MODELS,
        embedding_cache_dir=EMBEDDING_CACHE_DIR,
        batch_size=BATCH_SIZE,
        use_amp=USE_AMP,
    )

    data = stimuli_step.process(None)
    data = logits_step.process(data)

    print("✅ Done. Per-model cache files:")
    for name, path in data["cifar_logit_files"].items():
        print(f"  • {name}: {path}")


if __name__ == "__main__":
    main()
