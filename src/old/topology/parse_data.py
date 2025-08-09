import re
import numpy as np
import pandas as pd
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

def parse_r2_file(filepath):
    pattern = re.compile(r"Neuron\s+(\d+):\s+R²\s+\(probs\)\s+=\s+([-+]?\d*\.\d+|\d+)")
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                neuron_idx = int(match.group(1))
                r2 = float(match.group(2))
                if r2 > 0.1:
                    status = 'significant'
                elif r2 > 0.0:
                    status = 'weakly_significant'
                else:
                    status = 'non_significant'
                data.append((neuron_idx, r2, status))
    return pd.DataFrame(data, columns=['neuron_idx', 'r2', 'status'])

def build_area_map(specimen_ids, cells, neuron_indices):
    area_map = {}
    for neuron_idx in neuron_indices:
        specimen_id = specimen_ids[neuron_idx]
        area_row = cells[cells['cell_specimen_id'] == specimen_id]
        area_name = area_row['area'].values[0] if not area_row.empty else 'Unknown'
        area_map[neuron_idx] = area_name
    return area_map

def load_representation_vectors(vec_path):
    return np.load(vec_path)

def create_unified_data(r2_file_path, vec_path, specimen_ids_path, cells_df):
    df_r2 = parse_r2_file(r2_file_path)
    vectors = load_representation_vectors(vec_path)
    specimen_ids = np.load(specimen_ids_path)
    area_map = build_area_map(specimen_ids, cells_df, df_r2['neuron_idx'].values)
    
    df_r2['brain_area'] = df_r2['neuron_idx'].map(area_map)
    df_r2['vector'] = df_r2['neuron_idx'].apply(lambda idx: vectors[idx] if idx < len(vectors) else None)
    return df_r2

if __name__ == "__main__":
    r2_file_path = "/home/maria/LuckyMouse4/data/r2_log.txt"
    vec_path = '/home/maria/LuckyMouse4/data/logreg_representation_vectors_train_only.npy'
    specimen_ids_path = '/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/cell_specimen_ids_in_order.npy'
    
    # Initialize Brain Observatory Cache and load cell metadata
    boc = BrainObservatoryCache(
        manifest_file=str(Path('/home/maria/Documents/AllenBrainObservatory') / 'brain_observatory_manifest.json'))
    cells = boc.get_cell_specimens()
    cells_df = pd.DataFrame(cells)
    
    unified_df = create_unified_data(r2_file_path, vec_path, specimen_ids_path, cells_df)
    unified_df.to_pickle('/home/maria/LuckyMouse4/data/unified_neuron_data.pkl')
    print(f"Unified data saved with {len(unified_df)} entries")
