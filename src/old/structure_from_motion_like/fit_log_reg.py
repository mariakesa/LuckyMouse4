import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
unified_data_path = '/home/maria/LuckyMouse4/data/unified_neuron_data.pkl'
neural_data_path = '/home/maria/LuckyMouse4/data/hybrid_neural_responses.npy'
vit_embedding_path = '/home/maria/LuckyMouse4/data/google_vit-base-patch16-224_embeddings_logits.pkl'
output_path = '/home/maria/LuckyMouse4/data/visp_significant_neuron_weights.pkl'

n_images = 118
n_repeats = 50
k_shot = 80         # Number of stimuli per mini-task
n_tasks = 200        # Number of resampled few-shot tasks per neuron

# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading unified neuron metadata...")
df = pd.read_pickle(unified_data_path)
visp_neurons_significant = df[(df['brain_area'] == 'VISp') & (df['status'] == 'significant')]
neuron_indices = visp_neurons_significant['neuron_idx'].tolist()

print("Loading neural responses...")
neural_data = np.load(neural_data_path)

print("Loading ViT embeddings...")
with open(vit_embedding_path, 'rb') as f:
    embeddings_raw = pickle.load(f)
image_embeddings = embeddings_raw['natural_scenes']
D = image_embeddings.shape[1]

# ─── Core Few-Shot Weight Extraction Function ────────────────────────────────
def compute_fewshot_pga_weights(neuron_idx, k_shot=15, n_tasks=200):
    """
    For a given neuron index, resample many small logistic regression tasks
    and return a matrix of normalized weight vectors on S^{D-1}.
    """
    neuron_response = neural_data[neuron_idx]
    stimulus_responses = neuron_response.reshape(n_images, n_repeats)
    weight_vectors = []

    for _ in range(n_tasks):
        image_idxs = np.random.choice(n_images, size=k_shot, replace=False)

        X = []
        y = []
        for idx in image_idxs:
            for rep in range(n_repeats):
                X.append(image_embeddings[idx])
                y.append(stimulus_responses[idx, rep])

        X = np.stack(X)
        y = np.array(y)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        try:
            clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500)
            clf.fit(X, y)
            w = clf.coef_[0]
            w /= np.linalg.norm(w)
            weight_vectors.append(w)
        except Exception:
            continue  # Skip any failed fits

    return np.stack(weight_vectors) if weight_vectors else None

# ─── Run for All Significant VISp Neurons ─────────────────────────────────────
all_neuron_weights = {}

print(f"Processing {len(neuron_indices)} VISp significant neurons...")
for neuron_idx in tqdm(neuron_indices):
    weights = compute_fewshot_pga_weights(neuron_idx, k_shot=k_shot, n_tasks=n_tasks)
    if weights is not None:
        all_neuron_weights[neuron_idx] = weights

# ─── Save Results ─────────────────────────────────────────────────────────────
print(f"Saving results to {output_path}...")
with open(output_path, 'wb') as f:
    pickle.dump(all_neuron_weights, f)

print("Done!")
