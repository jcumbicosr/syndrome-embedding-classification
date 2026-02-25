import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def load_pickle(file_path: str)  ->  Dict:
    """Load data from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from: {file_path}: {e}")
        raise

def flatten_data(file_path: str, 
                 expected_dim: int = 320) -> pd.DataFrame:
    """ Flattens the hierarchical dictionary into a pandas DataFrame and 
    performs data integrity checks. """
    # Load the raw data
    raw_data = load_pickle(file_path)

    records = []
    skipped_entries = 0
    for syndrome_id, subjects in raw_data.items():
        if not isinstance(subjects, dict):
            continue # Skip inconsistent data formats

        for subject_id, images in subjects.items():
            if not isinstance(images, dict):
                skipped_entries += 1
                continue # Skip inconsistent data formats

            for image_id, embedding in images.items():
                # Check for missing embeddings
                if embedding is None:
                    skipped_entries += 1
                    continue
                # Check for correct embedding dimensions
                embedding = np.asarray(embedding)
                if embedding.shape[0] != expected_dim:
                    skipped_entries += 1
                    continue
                # Check for NaN values
                if np.isnan(embedding).any():
                    skipped_entries += 1
                    continue

                # Add valid record to the list
                records.append({
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id,
                    'embedding': embedding
                })

    logger.info(f"Skipped {skipped_entries} entries due to data integrity issues.")

    return pd.DataFrame(records)

def plot_syndrome_distribution(df: pd.DataFrame, 
                               output_dir: str = "results/figures/", 
                               show: bool = False) -> None:
    """Plots the distribution of images per syndrome and saves the plot to the specified directory."""
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    syndrome_counts = df['syndrome_id'].value_counts()
    syndrome_counts = syndrome_counts.sort_values(ascending=False)

    # Create a bar plot of the syndrome distribution
    fig, ax = plt.subplots(figsize=(12, 6))  # More explicit approach
    bars = syndrome_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.bar_label(bars.containers[0], label_type='edge', fontsize=10)
    ax.set_xlabel('Syndrome ID', fontsize=14)
    ax.set_ylabel('Number of Images', fontsize=14)
    ax.set_title('Distribution of Images per Syndrome', fontsize=16)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    # Save the plot
    plot_path = output_dir / "syndrome_distribution.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Syndrome distribution plot saved to: {plot_path}")

    # Show the plot if requested
    if show:
        fig.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    # data = load_pickle('data/mini_gm_public_v0.1.p')
    df = flatten_data('data/mini_gm_public_v0.1.p')
    plot_syndrome_distribution(df, show=False)
    print("Data loaded.")