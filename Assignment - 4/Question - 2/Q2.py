import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from PIL import Image
import requests
from io import BytesIO
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '1'


def load_image(image_path=None, url=None, max_size=150):
    """Load and optionally downsample an image."""
    if url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        print(f"Image downsampled from ({width}, {height}) to ({new_width}, {new_height})")
    
    return np.array(img)


def create_feature_vectors(img_array):
    height, width, _ = img_array.shape
    row_indices, col_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    row_indices = row_indices.flatten()
    col_indices = col_indices.flatten()
    red = img_array[:, :, 0].flatten()
    green = img_array[:, :, 1].flatten()
    blue = img_array[:, :, 2].flatten()
    
    raw_features = np.column_stack([row_indices, col_indices, red, green, blue]).astype(float)
    
    # Normalize each feature individually to [0, 1]
    features = np.zeros_like(raw_features)
    for i in range(5):
        min_val = raw_features[:, i].min()
        max_val = raw_features[:, i].max()
        if max_val > min_val:
            features[:, i] = (raw_features[:, i] - min_val) / (max_val - min_val)
        else:
            features[:, i] = 0.0
    
    return features, (height, width)


def gmm_cross_validation(features, k_range, n_folds=5, random_state=42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_results = {
        'k_values': list(k_range),
        'mean_log_likelihood': [],
        'std_log_likelihood': [],
        'all_fold_scores': []
    }
    
    print(f"\nPerforming {n_folds}-fold Cross-Validation for GMM Model Order Selection")
  
    for k in k_range:
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features)):
            train_data = features[train_idx]
            val_data = features[val_idx]
            
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                max_iter=200,
                n_init=3,
                random_state=random_state + fold_idx,
                init_params='kmeans'
            )
            gmm.fit(train_data)
            val_log_likelihood = gmm.score(val_data)
            fold_scores.append(val_log_likelihood)
        
        mean_ll = np.mean(fold_scores)
        std_ll = np.std(fold_scores)
        
        cv_results['mean_log_likelihood'].append(mean_ll)
        cv_results['std_log_likelihood'].append(std_ll)
        cv_results['all_fold_scores'].append(fold_scores)
        
        print(f"K = {k:2d}: Mean Log-Likelihood = {mean_ll:8.4f} (+/- {std_ll:.4f})")
    
    best_idx = np.argmax(cv_results['mean_log_likelihood'])
    best_k = cv_results['k_values'][best_idx]
    
    print("=" * 70)
    print(f"Best number of components: K = {best_k}")
    print(f"Best mean log-likelihood: {cv_results['mean_log_likelihood'][best_idx]:.4f}")
    
    return best_k, cv_results

def fit_final_gmm(features, n_components, random_state=42):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=300,
        n_init=5,
        random_state=random_state,
        init_params='kmeans'
    )
    gmm.fit(features)
    return gmm


def segment_image(gmm, features, original_shape):
    posteriors = gmm.predict_proba(features)
    labels = np.argmax(posteriors, axis=1)
    labels = labels.reshape(original_shape)
    return labels, posteriors


def create_label_image(labels, n_components):
    if n_components == 1:
        return np.zeros_like(labels, dtype=np.uint8)
    grayscale_values = np.linspace(0, 255, n_components).astype(np.uint8)
    return grayscale_values[labels]


def create_color_segmentation(labels, n_components):
    np.random.seed(42)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_components, 20)))[:n_components, :3]
    colors = (colors * 255).astype(np.uint8)
    
    height, width = labels.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for k in range(n_components):
        mask = labels == k
        color_image[mask] = colors[k]
    
    return color_image


def plot_cv_results(cv_results, best_k):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    k_values = cv_results['k_values']
    mean_ll = cv_results['mean_log_likelihood']
    std_ll = cv_results['std_log_likelihood']
    
    # Plot 1: Mean log-likelihood with error bars
    ax1 = axes[0]
    ax1.errorbar(k_values, mean_ll, yerr=std_ll, fmt='o-', capsize=5,
                 color='blue', linewidth=2, markersize=8)
    ax1.axvline(x=best_k, color='red', linestyle='--', linewidth=2,
                label=f'Best K = {best_k}')
    ax1.scatter([best_k], [mean_ll[k_values.index(best_k)]],
                color='red', s=200, zorder=5, marker='*')
    ax1.set_xlabel('Number of Components (K)', fontsize=12)
    ax1.set_ylabel('Mean Validation Log-Likelihood', fontsize=12)
    ax1.set_title('K-Fold Cross-Validation: Model Order Selection', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Plot 2: Box plot of fold scores
    ax2 = axes[1]
    ax2.boxplot(cv_results['all_fold_scores'], positions=k_values, widths=0.6)
    ax2.axvline(x=best_k, color='red', linestyle='--', linewidth=2,
                label=f'Best K = {best_k}')
    ax2.set_xlabel('Number of Components (K)', fontsize=12)
    ax2.set_ylabel('Validation Log-Likelihood', fontsize=12)
    ax2.set_title('Distribution of Fold Scores Across K Values', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig('cv_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_segmentation_results(original_image, labels, n_components):
    grayscale_labels = create_label_image(labels, n_components)
    color_labels = create_color_segmentation(labels, n_components)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image\n(Tiger - BSDS300 #108073)', fontsize=13)
    axes[0].axis('off')
    
    axes[1].imshow(grayscale_labels, cmap='gray')
    axes[1].set_title(f'GMM Segmentation (Grayscale)\nK = {n_components} components', fontsize=13)
    axes[1].axis('off')
    
    axes[2].imshow(color_labels)
    axes[2].set_title(f'GMM Segmentation (Color-coded)\nK = {n_components} components', fontsize=13)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    return fig, grayscale_labels, color_labels


def print_gmm_parameters(gmm, n_components):
    print("\n" + "=" * 70)
    print("FINAL GMM MODEL PARAMETERS")
    print("=" * 70)
    
    print(f"\nNumber of components: {n_components}")
    print(f"Covariance type: full")
    
    print(f"\nComponent weights (mixing proportions):")
    for k in range(n_components):
        print(f"  Component {k+1}: {gmm.weights_[k]:.4f}")
    
    print(f"\nComponent means (5D feature space):")
    print("  Features: [row, col, R, G, B] (all normalized to [0,1])")
    for k in range(n_components):
        mean = gmm.means_[k]
        print(f"  Component {k+1}: [row={mean[0]:.3f}, col={mean[1]:.3f}, "
              f"R={mean[2]:.3f}, G={mean[3]:.3f}, B={mean[4]:.3f}]")
    
    print(f"\nModel Statistics:")
    print(f"  Log-likelihood (lower bound): {gmm.lower_bound_:.4f}")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Number of iterations: {gmm.n_iter_}")


def main():

    # Tiger image URL (GitHub mirror - reliable)
    image_url = "https://raw.githubusercontent.com/BIDS/BSDS500/master/BSDS500/data/images/train/108073.jpg"
    # The user can also use other images from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images.html link
    
    # 1. LOAD IMAGE 
    print("\n1. LOADING AND PREPROCESSING IMAGE")
    print("-" * 40)
    
    try:
        img_array = load_image(url=image_url, max_size=150)
        print(f"Successfully loaded Tiger image (108073)")
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Please check your internet connection or use a local image.")
        return None
    
    print(f"Image shape: {img_array.shape}")
    print(f"Total pixels: {img_array.shape[0] * img_array.shape[1]}")
    
    # 2. CREATE FEATURES 
    print("\n2. CREATING 5-DIMENSIONAL FEATURE VECTORS")
    print("-" * 40)
    features, original_shape = create_feature_vectors(img_array)
    print(f"Feature vector shape: {features.shape}")
    print(f"Features: [row_index, col_index, R, G, B]")
    print(f"All features normalized to [0, 1] (5D unit hypercube)")
    
    # 3. K-FOLD CROSS-VALIDATION 
    print("\n3. K-FOLD CROSS-VALIDATION FOR MODEL ORDER SELECTION")
    print("-" * 40)
    
    k_range = range(2, 12)  # Test K = 2 to 11
    n_folds = 5             # 5-fold CV
    
    print(f"Testing K values: {list(k_range)}")
    print(f"Number of folds: {n_folds}")
    print(f"Objective: Maximum average validation log-likelihood")
    
    best_k, cv_results = gmm_cross_validation(
        features, k_range=k_range, n_folds=n_folds, random_state=42
    )
    
    # 4. PLOT CV RESULTS
    print("\n4. VISUALIZING CROSS-VALIDATION RESULTS")
    print("-" * 40)
    plot_cv_results(cv_results, best_k)
    print("Saved: cv_results.png")
    
    # 5. FIT FINAL GMM
    print("\n5. FITTING FINAL GMM MODEL")
    print("-" * 40)
    print(f"Training GMM with K = {best_k} components on full dataset...")
    final_gmm = fit_final_gmm(features, best_k, random_state=42)
    print("GMM fitting complete!")
    
    # Print GMM parameters
    print_gmm_parameters(final_gmm, best_k)
    
    # 6. SEGMENT IMAGE
    print("\n6. SEGMENTING IMAGE")
    print("-" * 40)
    print("Assigning pixels to most likely component using posterior probabilities")
    labels, posteriors = segment_image(final_gmm, features, original_shape)
    print(f"Segmentation complete!")
    print(f"Label image shape: {labels.shape}")
    
    # Segment statistics
    print(f"\nPixels per segment:")
    for k in range(best_k):
        count = np.sum(labels == k)
        pct = 100 * count / labels.size
        print(f"  Segment {k+1}: {count:6d} pixels ({pct:5.1f}%)")
    
    # 7. PLOT RESULTS
    print("\n7. VISUALIZING SEGMENTATION RESULTS")
    print("-" * 40)
    fig, grayscale_img, color_img = plot_segmentation_results(img_array, labels, best_k)
    print("Saved: segmentation_results.png")
    
    # SUMMARY
    print("\nSUMMARY")
    print(f"Image: Tiger (BSDS300 #108073)")
    print(f"Original size: {img_array.shape[:2]} (downsampled for efficiency)")
    print(f"Feature vector: 5D [row, col, R, G, B] normalized to [0,1]")
    print(f"Cross-validation: {n_folds}-fold")
    print(f"K values tested: {list(k_range)}")
    print(f"Optimal K (max validation log-likelihood): {best_k}")
    print(f"Best mean log-likelihood: {cv_results['mean_log_likelihood'][cv_results['k_values'].index(best_k)]:.4f}")
    print("\nOutput files:")
    print("  - cv_results.png (Cross-validation hyperparameter selection)")
    print("  - segmentation_results.png (Original vs Segmented image)")
    return img_array, labels, final_gmm, cv_results
  
if __name__ == "__main__":
    original_image, segment_labels, gmm_model, cv_results = main()
