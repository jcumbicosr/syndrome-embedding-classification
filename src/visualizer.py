import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def visualize_embeddings(data: pd.DataFrame, 
                         perplexity: int=30, 
                         n_iter: int=1000,
                         random_state: int=2,
                         output_path: str = "results/figures/",
                         show: bool = False ) -> None:
    """Reduces embeddings to 2D using t-SNE and plots them"""

    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    X = np.stack(data["embedding"].values)
    reduced_embeddings = tsne.fit_transform(X)

    # Create a color map based on syndrome_id
    unique_labels = np.unique(data["syndrome_id"])
    color_map = plt.get_cmap('tab10', len(unique_labels))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, label in enumerate(unique_labels):
        indices = data["syndrome_id"] == label
        ax.scatter(reduced_embeddings[indices, 0], 
                   reduced_embeddings[indices, 1], 
                   color=color_map(idx), 
                   label=label, 
                   alpha=0.7)

    ax.legend(title="Syndromes")
    ax.set_title("t-SNE 2D Visualization of Image Embeddings by Syndrome ID", fontsize=16)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(alpha=0.3)

    # Ensure the output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / "tsne_visualization.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"t-SNE visualization saved to: {plot_path}")

    if show:
        fig.show()
    

if __name__ == "__main__":
    from processing import flatten_data
    logging.basicConfig(level=logging.INFO)

    # Example usage
    df = flatten_data('data/mini_gm_public_v0.1.p')
    visualize_embeddings(df, perplexity=30)