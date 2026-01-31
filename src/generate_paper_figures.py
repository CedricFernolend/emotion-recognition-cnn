"""
Generate professional paper-style architecture figures for V1, V2, V3 models.
Full HD (1920x1080) resolution, suitable for publications and presentations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects

# Output settings
OUTPUT_DIR = '../results/paper_figures'
DPI = 150  # 1920x1080 at this DPI with figsize=(12.8, 7.2)
FIGSIZE = (12.8, 7.2)  # Results in 1920x1080 at 150 DPI

# Color scheme (professional, colorblind-friendly)
COLORS = {
    'input': '#2E86AB',       # Steel blue
    'conv': '#A23B72',        # Deep pink/magenta
    'pool': '#F18F01',        # Orange
    'attention_se': '#C73E1D', # Red-orange
    'attention_sp': '#3B1F2B', # Dark purple
    'fc': '#5C4D7D',          # Purple
    'output': '#2E86AB',      # Steel blue
    'skip': '#95A5A6',        # Gray
    'text': '#2C3E50',        # Dark blue-gray
    'bg': '#FFFFFF',          # White
    'grid': '#ECF0F1',        # Light gray
}


def draw_3d_block(ax, x, y, w, h, depth=0.15, color='#3498db', label='', sublabel='', fontsize=10):
    """Draw a 3D-style block with depth effect."""
    # Main face
    main = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.95, zorder=3
    )
    ax.add_patch(main)

    # Top face (lighter)
    top_color = plt.cm.colors.to_rgba(color)
    top_color = tuple([min(1, c + 0.2) for c in top_color[:3]] + [0.9])
    top = Polygon([
        (x, y + h),
        (x + depth, y + h + depth),
        (x + w + depth, y + h + depth),
        (x + w, y + h)
    ], facecolor=top_color, edgecolor='white', linewidth=1, zorder=2)
    ax.add_patch(top)

    # Right face (darker)
    right_color = plt.cm.colors.to_rgba(color)
    right_color = tuple([max(0, c - 0.15) for c in right_color[:3]] + [0.9])
    right = Polygon([
        (x + w, y),
        (x + w + depth, y + depth),
        (x + w + depth, y + h + depth),
        (x + w, y + h)
    ], facecolor=right_color, edgecolor='white', linewidth=1, zorder=2)
    ax.add_patch(right)

    # Labels
    if label:
        txt = ax.text(x + w/2, y + h/2 + (0.08 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color='white', zorder=4)
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black', alpha=0.3)])

    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.12, sublabel,
                ha='center', va='center', fontsize=fontsize-2, color='white', alpha=0.9, zorder=4)


def draw_arrow(ax, start, end, color='#7f8c8d', style='->', lw=2):
    """Draw an arrow between points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                              connectionstyle="arc3,rad=0"), zorder=1)


def draw_attention_module(ax, x, y, w, h, label, color, fontsize=8):
    """Draw a small attention module."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9, zorder=5
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=6)


def draw_dimension_label(ax, x, y, text, fontsize=8):
    """Draw dimension annotation."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=COLORS['text'], alpha=0.7, style='italic')


def generate_v1_figure():
    """Generate V1: Baseline CNN figure."""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=COLORS['bg'])
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6.75, 3.5, 'V1: Baseline CNN Architecture', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLORS['text'])
    ax.text(6.75, 3.1, '3 Convolutional Blocks with Residual Connections | ~1.8M Parameters',
            ha='center', va='center', fontsize=12, color=COLORS['text'], alpha=0.7)

    y_center = 1.5
    x = 0.3

    # Input
    draw_3d_block(ax, x, y_center - 0.4, 0.8, 0.8, 0.1, COLORS['input'], 'Input', '64×64×3')
    draw_dimension_label(ax, x + 0.4, y_center - 0.7, '64×64', 9)
    x += 1.1
    draw_arrow(ax, (x, y_center), (x + 0.3, y_center))
    x += 0.5

    # Block 1
    draw_3d_block(ax, x, y_center - 0.5, 1.2, 1.0, 0.12, COLORS['conv'], 'Block 1', '64 ch')
    # Skip connection visualization
    ax.annotate('', xy=(x + 1.2, y_center + 0.3), xytext=(x, y_center + 0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['skip'], lw=1.5,
                              connectionstyle="arc3,rad=-0.3"), zorder=1)
    ax.text(x + 0.6, y_center + 0.7, 'skip', fontsize=7, color=COLORS['skip'], ha='center')
    x += 1.5
    draw_arrow(ax, (x, y_center), (x + 0.2, y_center))
    x += 0.35

    # Pool 1
    draw_3d_block(ax, x, y_center - 0.3, 0.5, 0.6, 0.08, COLORS['pool'], 'Pool', '2×2')
    draw_dimension_label(ax, x + 0.25, y_center - 0.55, '→32×32', 8)
    x += 0.75
    draw_arrow(ax, (x, y_center), (x + 0.3, y_center))
    x += 0.5

    # Block 2
    draw_3d_block(ax, x, y_center - 0.5, 1.2, 1.0, 0.12, COLORS['conv'], 'Block 2', '128 ch')
    ax.annotate('', xy=(x + 1.2, y_center + 0.3), xytext=(x, y_center + 0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['skip'], lw=1.5,
                              connectionstyle="arc3,rad=-0.3"), zorder=1)
    x += 1.5
    draw_arrow(ax, (x, y_center), (x + 0.2, y_center))
    x += 0.35

    # Pool 2
    draw_3d_block(ax, x, y_center - 0.3, 0.5, 0.6, 0.08, COLORS['pool'], 'Pool', '2×2')
    draw_dimension_label(ax, x + 0.25, y_center - 0.55, '→16×16', 8)
    x += 0.75
    draw_arrow(ax, (x, y_center), (x + 0.3, y_center))
    x += 0.5

    # Block 3
    draw_3d_block(ax, x, y_center - 0.5, 1.2, 1.0, 0.12, COLORS['conv'], 'Block 3', '256 ch')
    ax.annotate('', xy=(x + 1.2, y_center + 0.3), xytext=(x, y_center + 0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['skip'], lw=1.5,
                              connectionstyle="arc3,rad=-0.3"), zorder=1)
    x += 1.5
    draw_arrow(ax, (x, y_center), (x + 0.2, y_center))
    x += 0.35

    # Pool 3
    draw_3d_block(ax, x, y_center - 0.3, 0.5, 0.6, 0.08, COLORS['pool'], 'Pool', '2×2')
    draw_dimension_label(ax, x + 0.25, y_center - 0.55, '→8×8', 8)
    x += 0.75
    draw_arrow(ax, (x, y_center), (x + 0.3, y_center))
    x += 0.5

    # Global Average Pooling
    draw_3d_block(ax, x, y_center - 0.35, 0.7, 0.7, 0.1, COLORS['pool'], 'GAP', '')
    draw_dimension_label(ax, x + 0.35, y_center - 0.55, '→256', 8)
    x += 1.0
    draw_arrow(ax, (x, y_center), (x + 0.3, y_center))
    x += 0.5

    # FC layers
    draw_3d_block(ax, x, y_center - 0.4, 0.9, 0.8, 0.1, COLORS['fc'], 'FC', '256→128')
    x += 1.2
    draw_arrow(ax, (x, y_center), (x + 0.3, y_center))
    x += 0.5

    # Output
    draw_3d_block(ax, x, y_center - 0.4, 0.8, 0.8, 0.1, COLORS['output'], 'Output', '6 classes')

    # Legend
    legend_y = 0.2
    legend_items = [
        (COLORS['conv'], 'Conv Block (3×3 Conv + BN + ReLU)'),
        (COLORS['pool'], 'Pooling Layer'),
        (COLORS['skip'], 'Residual Skip Connection'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(Rectangle((0.5 + i*4.5, legend_y - 0.15), 0.3, 0.3,
                               facecolor=color, edgecolor='white', linewidth=1))
        ax.text(0.9 + i*4.5, legend_y, label, fontsize=9, va='center', color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'v1_architecture_paper.png'), dpi=DPI,
                bbox_inches='tight', facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("✓ Created: v1_architecture_paper.png (1920×1080)")


def generate_v2_figure():
    """Generate V2: SE Attention CNN figure."""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=COLORS['bg'])
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1.2, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7.75, 3.5, 'V2: Channel Attention Architecture (SE-Net)', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLORS['text'])
    ax.text(7.75, 3.1, '4 Convolutional Blocks with Squeeze-and-Excitation Attention | ~5.5M Parameters',
            ha='center', va='center', fontsize=12, color=COLORS['text'], alpha=0.7)

    y_center = 1.5
    x = 0.2

    # Input
    draw_3d_block(ax, x, y_center - 0.35, 0.7, 0.7, 0.1, COLORS['input'], 'Input', '64×64')
    x += 0.95
    draw_arrow(ax, (x, y_center), (x + 0.25, y_center))
    x += 0.4

    # Blocks with SE attention
    blocks = [('Block 1', '64 ch', '32×32'), ('Block 2', '128 ch', '16×16'),
              ('Block 3', '256 ch', '8×8'), ('Block 4', '512 ch', '4×4')]

    for block_name, channels, output_size in blocks:
        # Conv block
        draw_3d_block(ax, x, y_center - 0.45, 1.1, 0.9, 0.11, COLORS['conv'], block_name, channels)

        # SE Attention module (below)
        se_x = x + 0.25
        se_y = y_center - 0.95
        draw_attention_module(ax, se_x, se_y, 0.6, 0.35, 'SE', COLORS['attention_se'], 9)
        # Connection lines
        ax.plot([x + 0.55, x + 0.55], [y_center - 0.45, se_y + 0.35],
                color=COLORS['attention_se'], lw=1.5, ls='--', alpha=0.7)

        # Skip connection
        ax.annotate('', xy=(x + 1.1, y_center + 0.25), xytext=(x, y_center + 0.25),
                    arrowprops=dict(arrowstyle='->', color=COLORS['skip'], lw=1.5,
                                  connectionstyle="arc3,rad=-0.25"), zorder=1)

        x += 1.35
        draw_arrow(ax, (x, y_center), (x + 0.15, y_center))
        x += 0.28

        # Pool
        draw_3d_block(ax, x, y_center - 0.25, 0.4, 0.5, 0.07, COLORS['pool'], 'P', '')
        draw_dimension_label(ax, x + 0.2, y_center - 0.5, f'→{output_size}', 7)
        x += 0.6
        draw_arrow(ax, (x, y_center), (x + 0.25, y_center))
        x += 0.4

    # Global Average Pooling
    draw_3d_block(ax, x, y_center - 0.3, 0.6, 0.6, 0.08, COLORS['pool'], 'GAP', '')
    x += 0.85
    draw_arrow(ax, (x, y_center), (x + 0.25, y_center))
    x += 0.4

    # FC
    draw_3d_block(ax, x, y_center - 0.35, 0.8, 0.7, 0.1, COLORS['fc'], 'FC', '512→128')
    x += 1.05
    draw_arrow(ax, (x, y_center), (x + 0.25, y_center))
    x += 0.4

    # Output
    draw_3d_block(ax, x, y_center - 0.35, 0.7, 0.7, 0.1, COLORS['output'], 'Output', '6 classes')

    # SE Block explanation (bottom right)
    box_x, box_y = 11.5, -0.7
    ax.add_patch(FancyBboxPatch((box_x, box_y), 4, 1.1,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor='#F8F9FA', edgecolor=COLORS['attention_se'],
                                linewidth=2, alpha=0.9))
    ax.text(box_x + 2, box_y + 0.9, 'Squeeze-and-Excitation (SE)', fontsize=10,
            fontweight='bold', ha='center', color=COLORS['attention_se'])
    ax.text(box_x + 2, box_y + 0.55, 'Global Pool → FC → ReLU → FC → Sigmoid', fontsize=8,
            ha='center', color=COLORS['text'])
    ax.text(box_x + 2, box_y + 0.25, 'Learns channel importance weights', fontsize=8,
            ha='center', color=COLORS['text'], style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'v2_architecture_paper.png'), dpi=DPI,
                bbox_inches='tight', facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("✓ Created: v2_architecture_paper.png (1920×1080)")


def generate_v3_figure():
    """Generate V3: Dual Attention CNN figure."""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=COLORS['bg'])
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1.5, 4.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7.75, 3.7, 'V3: Dual Attention Architecture (SE + Spatial)', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLORS['text'])
    ax.text(7.75, 3.3, '4 Convolutional Blocks with Channel and Spatial Attention | ~5.5M Parameters',
            ha='center', va='center', fontsize=12, color=COLORS['text'], alpha=0.7)

    y_center = 1.5
    x = 0.2

    # Input
    draw_3d_block(ax, x, y_center - 0.35, 0.7, 0.7, 0.1, COLORS['input'], 'Input', '64×64')
    x += 0.95
    draw_arrow(ax, (x, y_center), (x + 0.25, y_center))
    x += 0.4

    # Blocks with dual attention
    blocks = [('Block 1', '64 ch', '32×32'), ('Block 2', '128 ch', '16×16'),
              ('Block 3', '256 ch', '8×8'), ('Block 4', '512 ch', '4×4')]

    for block_name, channels, output_size in blocks:
        # Conv block
        draw_3d_block(ax, x, y_center - 0.45, 1.1, 0.9, 0.11, COLORS['conv'], block_name, channels)

        # SE Attention module (below left)
        se_x = x + 0.05
        se_y = y_center - 1.05
        draw_attention_module(ax, se_x, se_y, 0.5, 0.35, 'SE', COLORS['attention_se'], 8)
        ax.plot([x + 0.3, x + 0.3], [y_center - 0.45, se_y + 0.35],
                color=COLORS['attention_se'], lw=1.5, ls='--', alpha=0.7)

        # Spatial Attention module (below right)
        sp_x = x + 0.55
        sp_y = y_center - 1.05
        draw_attention_module(ax, sp_x, sp_y, 0.5, 0.35, 'Spatial', COLORS['attention_sp'], 7)
        ax.plot([x + 0.8, x + 0.8], [y_center - 0.45, sp_y + 0.35],
                color=COLORS['attention_sp'], lw=1.5, ls='--', alpha=0.7)

        # Skip connection
        ax.annotate('', xy=(x + 1.1, y_center + 0.25), xytext=(x, y_center + 0.25),
                    arrowprops=dict(arrowstyle='->', color=COLORS['skip'], lw=1.5,
                                  connectionstyle="arc3,rad=-0.25"), zorder=1)

        x += 1.35
        draw_arrow(ax, (x, y_center), (x + 0.15, y_center))
        x += 0.28

        # Pool
        draw_3d_block(ax, x, y_center - 0.25, 0.4, 0.5, 0.07, COLORS['pool'], 'P', '')
        draw_dimension_label(ax, x + 0.2, y_center - 0.5, f'→{output_size}', 7)
        x += 0.6
        draw_arrow(ax, (x, y_center), (x + 0.25, y_center))
        x += 0.4

    # Global Average Pooling
    draw_3d_block(ax, x, y_center - 0.3, 0.6, 0.6, 0.08, COLORS['pool'], 'GAP', '')
    x += 0.85
    draw_arrow(ax, (x, y_center), (x + 0.25, y_center))
    x += 0.4

    # FC with BatchNorm
    draw_3d_block(ax, x, y_center - 0.35, 0.8, 0.7, 0.1, COLORS['fc'], 'FC+BN', '512→128')
    x += 1.05
    draw_arrow(ax, (x, y_center), (x + 0.25, y_center))
    x += 0.4

    # Output
    draw_3d_block(ax, x, y_center - 0.35, 0.7, 0.7, 0.1, COLORS['output'], 'Output', '6 classes')

    # Attention explanations (bottom)
    # SE explanation
    box_x, box_y = 0.5, -1.2
    ax.add_patch(FancyBboxPatch((box_x, box_y), 4.8, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor='#F8F9FA', edgecolor=COLORS['attention_se'],
                                linewidth=2, alpha=0.9))
    ax.text(box_x + 2.4, box_y + 0.7, 'SE: Channel Attention', fontsize=10,
            fontweight='bold', ha='center', color=COLORS['attention_se'])
    ax.text(box_x + 2.4, box_y + 0.35, 'Learns WHICH features matter', fontsize=9,
            ha='center', color=COLORS['text'])
    ax.text(box_x + 2.4, box_y + 0.1, 'GlobalPool → FC → Sigmoid → Scale', fontsize=7,
            ha='center', color=COLORS['text'], alpha=0.7)

    # Spatial explanation
    box_x = 5.8
    ax.add_patch(FancyBboxPatch((box_x, box_y), 4.8, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor='#F8F9FA', edgecolor=COLORS['attention_sp'],
                                linewidth=2, alpha=0.9))
    ax.text(box_x + 2.4, box_y + 0.7, 'Spatial Attention', fontsize=10,
            fontweight='bold', ha='center', color=COLORS['attention_sp'])
    ax.text(box_x + 2.4, box_y + 0.35, 'Learns WHERE to focus', fontsize=9,
            ha='center', color=COLORS['text'])
    ax.text(box_x + 2.4, box_y + 0.1, 'AvgPool + MaxPool → Conv 7×7 → Sigmoid', fontsize=7,
            ha='center', color=COLORS['text'], alpha=0.7)

    # Training features
    box_x = 11.1
    ax.add_patch(FancyBboxPatch((box_x, box_y), 4.4, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor='#F8F9FA', edgecolor=COLORS['fc'],
                                linewidth=2, alpha=0.9))
    ax.text(box_x + 2.2, box_y + 0.7, 'Training Enhancements', fontsize=10,
            fontweight='bold', ha='center', color=COLORS['fc'])
    ax.text(box_x + 2.2, box_y + 0.35, '• Label Smoothing (0.1)', fontsize=8,
            ha='center', color=COLORS['text'])
    ax.text(box_x + 2.2, box_y + 0.1, '• LR Scheduler • Full Augmentation', fontsize=8,
            ha='center', color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'v3_architecture_paper.png'), dpi=DPI,
                bbox_inches='tight', facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("✓ Created: v3_architecture_paper.png (1920×1080)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating paper-quality figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Resolution: 1920×1080 ({DPI} DPI)")
    print("=" * 50)

    generate_v1_figure()
    generate_v2_figure()
    generate_v3_figure()

    print("=" * 50)
    print(f"\n✓ All figures saved to: {OUTPUT_DIR}/")
    print("  - v1_architecture_paper.png")
    print("  - v2_architecture_paper.png")
    print("  - v3_architecture_paper.png")


if __name__ == "__main__":
    main()
