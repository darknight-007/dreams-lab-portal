"""
Post-training analysis scripts for Multispectral Vision Transformer results.

After training multispectral_vit.py, use these scripts to:
1. Visualize latent space
2. Cluster similar tiles
3. Find similar tiles
4. Analyze geological patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import json
from typing import List, Tuple
import argparse


def load_latents(latent_file: str = 'multispectral_latents.npy',
                 paths_file: str = 'multispectral_tile_paths.txt') -> Tuple[np.ndarray, List[str]]:
    """Load saved latent representations and tile paths."""
    latents = np.load(latent_file)
    with open(paths_file, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(latents)} latent representations")
    print(f"Latent dimension: {latents.shape[1]}")
    return latents, paths


def visualize_latent_space(latents: np.ndarray, paths: List[str], 
                          output_file: str = 'latent_space_visualization.png',
                          method: str = 'both'):
    """Visualize latent space using PCA and/or t-SNE."""
    
    print("\n" + "="*80)
    print("Visualizing Latent Space")
    print("="*80)
    
    if method in ['pca', 'both']:
        print("\nComputing PCA...")
        pca = PCA(n_components=2)
        latents_2d_pca = pca.fit_transform(latents)
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    if method in ['tsne', 'both']:
        print("\nComputing t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        latents_2d_tsne = tsne.fit_transform(latents)
    
    # Create plots
    if method == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = [axes]
    
    idx = 0
    
    if method in ['pca', 'both']:
        axes[idx].scatter(latents_2d_pca[:, 0], latents_2d_pca[:, 1], 
                         alpha=0.6, s=20)
        axes[idx].set_title('Latent Space (PCA)')
        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[idx].grid(True, alpha=0.3)
        idx += 1
    
    if method in ['tsne', 'both']:
        axes[idx].scatter(latents_2d_tsne[:, 0], latents_2d_tsne[:, 1], 
                         alpha=0.6, s=20)
        axes[idx].set_title('Latent Space (t-SNE)')
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    plt.close()


def cluster_tiles(latents: np.ndarray, paths: List[str], 
                  n_clusters: int = 10, method: str = 'kmeans',
                  output_file: str = 'tile_clusters.json'):
    """Cluster tiles based on latent representations."""
    
    print("\n" + "="*80)
    print(f"Clustering Tiles ({method}, {n_clusters} clusters)")
    print("="*80)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(latents)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(latents)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"DBSCAN found {n_clusters} clusters (+ {sum(labels == -1)} noise points)")
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Calculate silhouette score
    if method == 'kmeans' and n_clusters > 1:
        silhouette = silhouette_score(latents, labels)
        print(f"Silhouette score: {silhouette:.3f}")
    
    # Organize by cluster
    clusters = {}
    for i, (label, path) in enumerate(zip(labels, paths)):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            'path': path,
            'latent_index': i
        })
    
    # Print cluster sizes
    print("\nCluster sizes:")
    for label in sorted(clusters.keys()):
        if label == -1:
            print(f"  Noise: {len(clusters[label])} tiles")
        else:
            print(f"  Cluster {label}: {len(clusters[label])} tiles")
    
    # Save clusters
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=2)
    print(f"\nSaved clusters to: {output_file}")
    
    return labels, clusters


def visualize_clusters(latents: np.ndarray, labels: np.ndarray,
                       output_file: str = 'clustered_latent_space.png'):
    """Visualize latent space colored by cluster assignments."""
    
    print("\nVisualizing clusters...")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique labels
    unique_labels = set(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black for noise
            mask = labels == label
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c='black', marker='x', s=20, alpha=0.5, label='Noise')
        else:
            mask = labels == label
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c=[color], s=20, alpha=0.6, label=f'Cluster {label}')
    
    ax.set_title('Latent Space Clusters (PCA projection)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved cluster visualization to: {output_file}")
    plt.close()


def find_similar_tiles(latents: np.ndarray, paths: List[str],
                       query_index: int, n_similar: int = 10):
    """Find tiles most similar to a query tile."""
    
    print("\n" + "="*80)
    print(f"Finding tiles similar to: {paths[query_index]}")
    print("="*80)
    
    query_latent = latents[query_index]
    
    # Compute cosine similarities
    similarities = np.dot(latents, query_latent) / (
        np.linalg.norm(latents, axis=1) * np.linalg.norm(query_latent)
    )
    
    # Get top N most similar (excluding the query itself)
    top_indices = np.argsort(similarities)[::-1][1:n_similar+1]
    
    print(f"\nTop {n_similar} most similar tiles:")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i:2d}. {Path(paths[idx]).name} (similarity: {similarities[idx]:.4f})")
    
    return top_indices, similarities[top_indices]


def analyze_cluster_statistics(clusters: dict, paths: List[str]):
    """Analyze statistics of each cluster."""
    
    print("\n" + "="*80)
    print("Cluster Statistics")
    print("="*80)
    
    for label in sorted(clusters.keys()):
        if label == -1:
            continue
        
        cluster_paths = [c['path'] for c in clusters[label]]
        
        # Extract SET number from path if available
        set_numbers = []
        for path in cluster_paths:
            if 'SET' in path:
                parts = Path(path).parts
                for part in parts:
                    if 'SET' in part:
                        set_numbers.append(part)
                        break
        
        print(f"\nCluster {label}:")
        print(f"  Size: {len(cluster_paths)} tiles")
        if set_numbers:
            set_counts = {s: set_numbers.count(s) for s in set(set_numbers)}
            print(f"  SET distribution: {set_counts}")
        
        # Show sample files
        print(f"  Sample files:")
        for path in cluster_paths[:5]:
            print(f"    {Path(path).name}")


def create_tile_map(latents: np.ndarray, paths: List[str],
                   output_file: str = 'tile_similarity_map.png'):
    """Create a similarity map showing relationships between tiles."""
    
    print("\n" + "="*80)
    print("Creating Tile Similarity Map")
    print("="*80)
    
    # Compute pairwise distances
    print("Computing pairwise similarities...")
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Sample subset if too many tiles
    max_tiles = 500
    if len(latents) > max_tiles:
        print(f"Sampling {max_tiles} tiles for visualization...")
        indices = np.random.choice(len(latents), max_tiles, replace=False)
        latents_sample = latents[indices]
        paths_sample = [paths[i] for i in indices]
    else:
        latents_sample = latents
        paths_sample = paths
        indices = np.arange(len(latents))
    
    similarity_matrix = cosine_similarity(latents_sample)
    
    # Visualize as heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    ax.set_title('Tile Similarity Matrix')
    ax.set_xlabel('Tile Index')
    ax.set_ylabel('Tile Index')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved similarity map to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze Multispectral ViT results')
    parser.add_argument('--latent_file', type=str, default='multispectral_latents.npy',
                       help='Path to latent representations file')
    parser.add_argument('--paths_file', type=str, default='multispectral_tile_paths.txt',
                       help='Path to tile paths file')
    parser.add_argument('--visualize', action='store_true',
                       help='Create latent space visualizations')
    parser.add_argument('--cluster', action='store_true',
                       help='Cluster tiles by similarity')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters (default: 10)')
    parser.add_argument('--cluster_method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan'],
                       help='Clustering method')
    parser.add_argument('--similarity', type=int, default=None,
                       help='Find similar tiles to tile at this index')
    parser.add_argument('--n_similar', type=int, default=10,
                       help='Number of similar tiles to find')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    args = parser.parse_args()
    
    # Load latents
    if not Path(args.latent_file).exists():
        print(f"ERROR: Latent file not found: {args.latent_file}")
        print("Run multispectral_vit.py with --extract_latents first!")
        return
    
    latents, paths = load_latents(args.latent_file, args.paths_file)
    
    # Run analyses
    if args.all or args.visualize:
        visualize_latent_space(latents, paths, method='both')
    
    if args.all or args.cluster:
        labels, clusters = cluster_tiles(latents, paths, 
                                       n_clusters=args.n_clusters,
                                       method=args.cluster_method)
        visualize_clusters(latents, labels)
        analyze_cluster_statistics(clusters, paths)
    
    if args.similarity is not None:
        find_similar_tiles(latents, paths, args.similarity, args.n_similar)
    
    if args.all:
        create_tile_map(latents, paths)
    
    if not args.all and not args.visualize and not args.cluster and args.similarity is None:
        print("\nNo analysis selected. Use --all to run everything, or:")
        print("  --visualize : Create latent space visualizations")
        print("  --cluster   : Cluster tiles by similarity")
        print("  --similarity N : Find tiles similar to tile at index N")


if __name__ == '__main__':
    main()

