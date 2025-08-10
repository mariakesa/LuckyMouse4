import numpy as np
import torch
import pickle
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageClassification
from abc import ABC, abstractmethod
from dotenv import load_dotenv
load_dotenv()
class PipelineStep(ABC):
    @abstractmethod
    def process(self, data):
        pass


class AllenStimuliFetchStep(PipelineStep):
    SESSION_A = 501704220
    SESSION_B = 501559087
    SESSION_C = 501474098

    def __init__(self, boc):
        self.boc = boc

    def process(self, data):
        if isinstance(data, tuple):
            container_id, session, stimulus = data
            data = {'container_id': container_id, 'session': session, 'stimulus': stimulus}
        elif data is None:
            data = {}

        raw_data_dct = {
            'natural_movie_one': self.boc.get_ophys_experiment_data(self.SESSION_A).get_stimulus_template('natural_movie_one'),
            'natural_movie_two': self.boc.get_ophys_experiment_data(self.SESSION_C).get_stimulus_template('natural_movie_two'),
            'natural_movie_three': self.boc.get_ophys_experiment_data(self.SESSION_A).get_stimulus_template('natural_movie_three'),
            'natural_scenes': self.boc.get_ophys_experiment_data(self.SESSION_B).get_stimulus_template('natural_scenes')
        }

        data['raw_data_dct'] = raw_data_dct
        return data

class ImageToLogitEmbeddingStep(PipelineStep):
    def __init__(self, embedding_cache_dir: str):
        self.model_name = "google/vit-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.eval()

        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_prefix = self.model_name.replace('/', '_')
        self.embeddings_file = self.embedding_cache_dir / f"{self.model_prefix}_embeddings_logits.pkl"

    def process(self, data):
        raw_data_dct = data['raw_data_dct']

        if self.embeddings_file.exists():
            print(f"✅ Found cached logits for model {self.model_prefix} at:\n{self.embeddings_file}")
            data['embedding_file'] = str(self.embeddings_file)
            return data

        print(f"🔄 No logits cache found for model {self.model_prefix}. Generating now...")
        embeddings_dict = {}
        for stim_name, frames_array in raw_data_dct.items():
            embeddings = self._process_stims(frames_array)
            embeddings_dict[stim_name] = embeddings

        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"📁 Saved logit embeddings to: {self.embeddings_file}")

        data['embedding_file'] = str(self.embeddings_file)
        return data

    def _process_stims(self, frames_array):
        n_frames = len(frames_array)
        frames_3ch = np.repeat(frames_array[:, None, :, :], 3, axis=1)

        all_logits = []
        for i in range(n_frames):
            inputs = self.processor(images=frames_3ch[i], return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze().cpu().numpy()
            all_logits.append(logits)

        return np.stack(all_logits)
    


from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os

import os
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

class AllenAPI:
    """Singleton class for accessing Allen Brain Observatory API"""
    
    _instance = None
    _boc = None  # Lazy-loaded BrainObservatoryCache instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def boc(self):
        """Lazy-load BrainObservatoryCache only when accessed."""
        if self._boc is None:
            allen_cache_path = os.environ.get('CAIM_ALLEN_CACHE_PATH')
            if not allen_cache_path:
                raise ValueError("AllenAPI requires a valid cache path. Set `CAIM_ALLEN_CACHE_PATH` in .env.")

            manifest_path = Path(allen_cache_path) / 'brain_observatory_manifest.json'
            self._boc = BrainObservatoryCache(manifest_file=str(manifest_path))

        return self._boc

    def get_boc(self):
        """Retrieve the BrainObservatoryCache object, ensuring it is initialized."""
        return self.boc

# Create a global instance so that all files can use it
allen_api = AllenAPI()

# Instantiate AllenSDK cache and pipeline steps
allen_cache_path = Path(os.environ.get('CAIM_ALLEN_CACHE_PATH'))
boc =allen_api.get_boc()  # set your correct path
stimuli_step = AllenStimuliFetchStep(boc)
embedding_step = ImageToLogitEmbeddingStep(embedding_cache_dir="/home/maria/Documents/HuggingMouseData/MouseViTEmbeddings")
# Run pipeline
data = stimuli_step.process(None)
data = embedding_step.process(data)

# Confirm result
print("✅ Final output embedding file:", data["embedding_file"])
