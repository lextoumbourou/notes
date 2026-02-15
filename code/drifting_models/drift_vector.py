"""
Drift Vector Visualization for MNIST

Generates visualization images for the drift field computation from:
"Generative Modeling via Drifting" (Deng et al., 2026)

Usage:
    uv run --with numpy --with scipy --with pillow --with tensorflow python drift_vector.py

Output images are saved to public/notes/_media/ with prefix 'drifting-models-'
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.special import softmax

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = (SCRIPT_DIR / "../../notes/_media").resolve()

def generate_noisy_samples(base_imgs, size=28):
    """Generate visually distinct noisy versions of base images.

    Each sample uses a different base image with a different gradient overlay,
    making them easy to distinguish while showing they're noisy digits.
    """
    patterns = []
    np.random.seed(123)
    noise_scale = 0.5  # Very heavy noise

    for i, base_img in enumerate(base_imgs):
        base = base_img.reshape(size, size)

        if i == 0:
            # Pattern 0: Horizontal gradient overlay
            grad = np.linspace(-0.4, 0.4, size).reshape(1, -1).repeat(size, axis=0)
        elif i == 1:
            # Pattern 1: Vertical gradient overlay
            grad = np.linspace(-0.4, 0.4, size).reshape(-1, 1).repeat(size, axis=1)
        else:
            # Pattern 2: Diagonal gradient overlay
            x, y = np.meshgrid(np.linspace(-0.4, 0.4, size), np.linspace(-0.4, 0.4, size))
            grad = (x + y) / 2

        noisy = np.clip(base + grad + np.random.randn(size, size) * noise_scale, 0, 1)
        patterns.append(noisy)

    return np.array(patterns)

# Load MNIST
try:
    from tensorflow.keras.datasets import mnist
    def load_mnist():
        (x_train, y_train), _ = mnist.load_data()
        return x_train, y_train
except ImportError:
    from sklearn.datasets import fetch_openml
    def load_mnist():
        mnist_data = fetch_openml('mnist_784', version=1, as_frame=False)
        x_train = mnist_data.data.reshape(-1, 28, 28).astype(np.uint8)
        y_train = mnist_data.target.astype(int)
        return x_train, y_train


def compute_V(x, y_pos, y_neg, temperature=0.05):
    """Compute drift vector V with all intermediate values."""
    N, D = x.shape
    N_pos, N_neg = y_pos.shape[0], y_neg.shape[0]
    tau = temperature * np.sqrt(D)

    # Step 1: L2 distances
    dist_pos = cdist(x, y_pos, metric='euclidean')
    dist_neg = cdist(x, y_neg, metric='euclidean')
    dist_neg_raw = dist_neg.copy()

    # Step 2: Mask self-distances
    if N == N_neg:
        dist_neg = dist_neg + np.eye(N) * 1e6

    # Step 3-4: Logits and softmax
    logit = np.concatenate([-dist_pos / tau, -dist_neg / tau], axis=1)
    A_row = softmax(logit, axis=1)
    A_col = softmax(logit, axis=0)
    A = np.sqrt(A_row * A_col)

    # Step 5: Split attention
    A_pos, A_neg = A[:, :N_pos], A[:, N_pos:]

    # Step 6: Cross-weighting
    W_pos = A_pos * np.sum(A_neg, axis=1, keepdims=True)
    W_neg = A_neg * np.sum(A_pos, axis=1, keepdims=True)

    # Step 7: Compute drift
    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    V = drift_pos - drift_neg

    return {
        'dist_pos': dist_pos, 'dist_neg_raw': dist_neg_raw, 'dist_neg_masked': dist_neg,
        'A': A, 'A_pos': A_pos, 'A_neg': A_neg,
        'W_pos': W_pos, 'W_neg': W_neg,
        'drift_pos': drift_pos, 'drift_neg': drift_neg, 'V': V
    }


def create_matrix_viz(matrix, row_imgs, col_imgs, output_path, cell_format=".2f"):
    """Create matrix visualization with images as headers."""
    from PIL import Image, ImageDraw, ImageFont

    n_rows, n_cols = matrix.shape
    img_size, cell_height, img_padding, margin = 60, 70, 10, 20
    col_cell_width = img_size + img_padding

    total_width = margin + img_size + 10 + n_cols * col_cell_width + margin
    total_height = margin + img_size + 10 + n_rows * cell_height + margin

    img = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()

    col_start_x = margin + img_size + 10
    row_start_y = margin + img_size + 10

    # Column headers
    for j, col_img in enumerate(col_imgs):
        gray = (col_img * 255).astype(np.uint8)
        pil = Image.fromarray(gray).resize((img_size, img_size), Image.NEAREST)
        img.paste(pil, (col_start_x + j * col_cell_width + img_padding // 2, margin))

    # Rows
    for i, row_img in enumerate(row_imgs):
        gray = (row_img * 255).astype(np.uint8)
        pil = Image.fromarray(gray).resize((img_size, img_size), Image.NEAREST)
        img.paste(pil, (margin, row_start_y + i * cell_height + (cell_height - img_size) // 2))

        for j in range(n_cols):
            cell_x = col_start_x + j * col_cell_width
            cell_y = row_start_y + i * cell_height
            draw.rectangle([cell_x, cell_y, cell_x + col_cell_width, cell_y + cell_height], outline='lightgray')

            val = matrix[i, j]
            text = "1e6" if val > 1e5 else f"{val:{cell_format}}"
            bbox = draw.textbbox((0, 0), text, font=font)
            draw.text((cell_x + (col_cell_width - bbox[2]) // 2, cell_y + (cell_height - bbox[3]) // 2), text, fill='black', font=font)

    img.save(output_path)


def create_drift_viz(drift_pos, drift_neg, V, row_imgs, output_path):
    """Create drift vectors visualization."""
    from PIL import Image, ImageDraw, ImageFont

    n_rows = len(row_imgs)
    img_size, cell_height, margin = 60, 70, 20
    col_width = img_size + 20

    total_width = margin + img_size + 10 + 3 * col_width + margin
    total_height = margin + 30 + n_rows * cell_height + margin

    img = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()

    col_start_x = margin + img_size + 10
    row_start_y = margin + 30

    # Headers
    for j, header in enumerate(["drift_pos", "drift_neg", "V"]):
        bbox = draw.textbbox((0, 0), header, font=font)
        draw.text((col_start_x + j * col_width + (col_width - bbox[2]) // 2, margin), header, fill='black', font=font)

    # Rows
    for i, row_img in enumerate(row_imgs):
        gray = (row_img * 255).astype(np.uint8)
        pil = Image.fromarray(gray).resize((img_size, img_size), Image.NEAREST)
        img.paste(pil, (margin, row_start_y + i * cell_height + (cell_height - img_size) // 2))

        for j, data in enumerate([drift_pos[i], drift_neg[i], V[i]]):
            d = data.reshape(28, 28)
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255
            pil = Image.fromarray(d_norm.astype(np.uint8)).resize((img_size, img_size), Image.NEAREST)
            img.paste(pil, (col_start_x + j * col_width + 10, row_start_y + i * cell_height + 5))

    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Generate drift vector visualizations')
    parser.add_argument('--output', '-o', default=str(DEFAULT_OUTPUT), help='Output directory')
    parser.add_argument('--prefix', '-p', default='drifting-models-', help='Filename prefix')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST...")
    x_train, y_train = load_mnist()
    x_train_norm = x_train.astype(np.float32) / 255.0

    np.random.seed(42)
    idx_3 = np.where(y_train == 3)[0]

    # 5 positive samples (real MNIST 3s)
    pos_idx = np.random.choice(idx_3, 5, replace=False)
    y_pos = x_train_norm[pos_idx].reshape(5, -1)

    # 3 generated samples - each based on a different 3 with heavy noise
    base_idx = np.random.choice(np.setdiff1d(idx_3, pos_idx), 3, replace=False)
    base_imgs = x_train_norm[base_idx]
    noisy_samples = generate_noisy_samples(base_imgs)
    x_query = noisy_samples.reshape(3, -1)
    y_neg = x_query  # negatives are the same as generated samples

    print("Computing drift vectors...")
    data = compute_V(x_query, y_pos, y_neg, temperature=0.04)

    x_imgs = x_query.reshape(-1, 28, 28)
    pos_imgs = y_pos.reshape(-1, 28, 28)
    neg_imgs = y_neg.reshape(-1, 28, 28)

    print("Generating visualizations...")
    prefix = args.prefix

    # No color tinting needed - patterns are visually distinct
    create_matrix_viz(data['dist_pos'], x_imgs, pos_imgs, f"{output_dir}/{prefix}01_dist_pos.png")
    create_matrix_viz(data['dist_neg_raw'], x_imgs, neg_imgs, f"{output_dir}/{prefix}02a_dist_neg_raw.png")
    create_matrix_viz(data['dist_neg_masked'], x_imgs, neg_imgs, f"{output_dir}/{prefix}02b_dist_neg_masked.png")
    create_matrix_viz(data['A'], x_imgs, np.concatenate([pos_imgs, neg_imgs]), f"{output_dir}/{prefix}03_attention.png")
    create_matrix_viz(data['A_pos'], x_imgs, pos_imgs, f"{output_dir}/{prefix}04a_A_pos.png")
    create_matrix_viz(data['A_neg'], x_imgs, neg_imgs, f"{output_dir}/{prefix}04b_A_neg.png")
    create_matrix_viz(data['W_pos'], x_imgs, pos_imgs, f"{output_dir}/{prefix}05a_W_pos.png")
    create_matrix_viz(data['W_neg'], x_imgs, neg_imgs, f"{output_dir}/{prefix}05b_W_neg.png")
    create_drift_viz(data['drift_pos'], data['drift_neg'], data['V'], x_imgs, f"{output_dir}/{prefix}06_drift_vectors.png")

    print(f"Saved 9 images to {output_dir}/{prefix}*.png")


if __name__ == "__main__":
    main()
