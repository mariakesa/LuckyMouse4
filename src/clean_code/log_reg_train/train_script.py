import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

# ─── Load Data ───────────────────────────────────────────────────────────────
dataset_path_dict = {
    "embeddings": "/home/maria/Documents/HuggingMouseData/MouseViTEmbeddings/google_vit-base-patch16-224_embeddings_logits.pkl",
    "neural": "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/hybrid_neural_responses.npy"  # trial-wise binary spikes
}
dataset_path_dict = {
    "embeddings": "/home/maria/Documents/HuggingMouseData/CIFARLogits/cifar10_resnet56_embeddings_logits.pkl",
    "neural": "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/hybrid_neural_responses.npy"  # trial-wise binary spikes
}
dataset_path_dict = {
    "embeddings": "/home/maria/Documents/HuggingMouseData/CIFARLogits/cifar100_resnet56_embeddings_logits.pkl",
    "neural": "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/hybrid_neural_responses.npy"  # trial-wise binary spikes
}


data_save_name ={
    "cifar10": ["/home/maria/LuckyMouse4/data/cifar10_r2.npy", "/home/maria/LuckyMouse4/data/cifar10_vectors.npy"],
    "cifar100": ["/home/maria/LuckyMouse4/data/cifar100_r2.npy", "/home/maria/LuckyMouse4/data/cifar100_vectors.npy"]
}
with open(dataset_path_dict['embeddings'], "rb") as f:
    embeddings_raw = pickle.load(f)
embeddings = embeddings_raw['natural_scenes']  # shape: (118, 1000)
print("Full embedding shape:", embeddings.shape)

neural_data = np.load(dataset_path_dict["neural"])  # shape: (neurons, 5900)
print("Neural data shape:", neural_data.shape)

# ─── Construct Trial-wise Design Matrix ──────────────────────────────────────
n_images = embeddings.shape[0]        # 118
n_trials = 50
n_total = n_images * n_trials         # 5900

X_all = np.repeat(embeddings, n_trials, axis=0)  # shape: (5900, 1000)

# ─── Split Train/Test ────────────────────────────────────────────────────────
n_train_images = 100
n_test_images = 18

train_mask = np.isin(np.tile(np.arange(n_images), n_trials), np.arange(n_train_images))
test_mask = np.isin(np.tile(np.arange(n_images), n_trials), np.arange(n_train_images, n_images))

X_train = X_all[train_mask]
X_test = X_all[test_mask]
print("Train trials:", X_train.shape[0], "| Test trials:", X_test.shape[0])

# ─── Fit Model for Each Neuron ───────────────────────────────────────────────
n_neurons = neural_data.shape[0]
embedding_dim = X_train.shape[1]

r2_test_scores = []
representation_vectors_train = np.zeros((n_neurons, embedding_dim))
failures = []

for i in tqdm(range(n_neurons), desc="Fitting neurons"):
    y_all = neural_data[i]
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    try:
        model = LogisticRegression(solver="liblinear", max_iter=1000)
        model.fit(X_train, y_train)
        probs_pred = model.predict_proba(X_test)[:, 1]
        r2 = r2_score(y_test, probs_pred)
        r2_test_scores.append(r2)

        representation_vectors_train[i] = model.coef_[0]

        print(f"Neuron {i}: R² (probs) = {r2:.4f}")
    except Exception as e:
        print(f"Neuron {i}: Error - {e}")
        r2_test_scores.append(None)
        representation_vectors_train[i] = np.nan
        failures.append(i)

# ─── Save Outputs ────────────────────────────────────────────────────────────
np.save(data_save_name['cifar100'][0], r2_test_scores)
np.save(data_save_name['cifar100'][1], representation_vectors_train)
print("Saved R² scores and representation vectors (trained on train set only).")

if failures:
    print(f"Failed to fit {len(failures)} neurons: {failures}")
