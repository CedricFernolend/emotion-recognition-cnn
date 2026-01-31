"""
Generate clean horizontal architecture diagrams for V1, V2, and V3 models.

Creates presentation-ready visualizations showing the structure of each model.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUTPUT_DIR = '../results/architecture_diagrams'


def create_block(ax, x, y, width, height, label, sublabel=None, color='#3498db', alpha=0.9, fontsize=10):
    """Create a rounded rectangle block."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor='white', linewidth=2, alpha=alpha
    )
    ax.add_patch(box)

    # Main label
    ax.text(x, y + (0.08 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='white')

    # Sublabel
    if sublabel:
        ax.text(x, y - 0.12, sublabel,
                ha='center', va='center', fontsize=fontsize-2, color='white', alpha=0.9)


def create_arrow(ax, start, end, color='#7f8c8d'):
    """Create a horizontal arrow between blocks."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))


def create_attention_badge(ax, x, y, label, color):
    """Create a small attention badge below a block."""
    box = FancyBboxPatch(
        (x - 0.35, y - 0.15), 0.7, 0.3,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=7, fontweight='bold', color='white')


def generate_v1_diagram():
    """Generate V1 horizontal architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6.25, 2.2, 'V1: Baseline CNN', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2c3e50')

    # Colors
    input_color = '#27ae60'
    conv_color = '#3498db'
    pool_color = '#e67e22'
    fc_color = '#9b59b6'
    output_color = '#e74c3c'

    x_pos = 0.5
    y = 1.0

    # Input
    create_block(ax, x_pos, y, 0.9, 0.8, 'Input', '64×64', input_color)
    x_pos += 0.7
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # Block 1
    create_block(ax, x_pos, y, 1.0, 0.8, 'Block 1', '64 ch', conv_color)
    x_pos += 0.75
    create_arrow(ax, (x_pos, y), (x_pos + 0.3, y))
    x_pos += 0.55
    create_block(ax, x_pos, y, 0.6, 0.5, 'Pool', None, pool_color, fontsize=8)
    x_pos += 0.5
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.65

    # Block 2
    create_block(ax, x_pos, y, 1.0, 0.8, 'Block 2', '128 ch', conv_color)
    x_pos += 0.75
    create_arrow(ax, (x_pos, y), (x_pos + 0.3, y))
    x_pos += 0.55
    create_block(ax, x_pos, y, 0.6, 0.5, 'Pool', None, pool_color, fontsize=8)
    x_pos += 0.5
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.65

    # Block 3
    create_block(ax, x_pos, y, 1.0, 0.8, 'Block 3', '256 ch', conv_color)
    x_pos += 0.75
    create_arrow(ax, (x_pos, y), (x_pos + 0.3, y))
    x_pos += 0.55
    create_block(ax, x_pos, y, 0.6, 0.5, 'Pool', None, pool_color, fontsize=8)
    x_pos += 0.5
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.65

    # Global Pool
    create_block(ax, x_pos, y, 0.8, 0.6, 'GAP', None, pool_color, fontsize=9)
    x_pos += 0.6
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # FC
    create_block(ax, x_pos, y, 0.9, 0.7, 'FC', '256→128', fc_color, fontsize=9)
    x_pos += 0.7
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # Output
    create_block(ax, x_pos, y, 1.0, 0.8, 'Output', '6 classes', output_color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'v1_architecture.png'), dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: v1_architecture.png")


def generate_v2_diagram():
    """Generate V2 horizontal architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7.5, 2.2, 'V2: SE Attention', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2c3e50')

    # Colors
    input_color = '#27ae60'
    conv_color = '#3498db'
    pool_color = '#e67e22'
    attention_color = '#9b59b6'
    fc_color = '#8e44ad'
    output_color = '#e74c3c'

    x_pos = 0.5
    y = 1.0

    # Input
    create_block(ax, x_pos, y, 0.9, 0.8, 'Input', '64×64', input_color)
    x_pos += 0.7
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # Blocks 1-4 with SE attention
    for i, (ch, label) in enumerate([(64, 'Block 1'), (128, 'Block 2'), (256, 'Block 3'), (512, 'Block 4')]):
        create_block(ax, x_pos, y, 1.0, 0.8, label, f'{ch} ch', conv_color)
        create_attention_badge(ax, x_pos, y - 0.55, 'SE', attention_color)
        x_pos += 0.75
        create_arrow(ax, (x_pos, y), (x_pos + 0.3, y))
        x_pos += 0.55
        create_block(ax, x_pos, y, 0.6, 0.5, 'Pool', None, pool_color, fontsize=8)
        x_pos += 0.5
        create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
        x_pos += 0.65

    # Global Pool
    create_block(ax, x_pos, y, 0.8, 0.6, 'GAP', None, pool_color, fontsize=9)
    x_pos += 0.6
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # FC
    create_block(ax, x_pos, y, 0.9, 0.7, 'FC', '512→128', fc_color, fontsize=9)
    x_pos += 0.7
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # Output
    create_block(ax, x_pos, y, 1.0, 0.8, 'Output', '6 classes', output_color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'v2_architecture.png'), dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: v2_architecture.png")


def generate_v3_diagram():
    """Generate V3 (V4) horizontal architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 4.5))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.8, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7.5, 2.2, 'V3: Dual Attention (SE + Spatial)', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2c3e50')

    # Colors
    input_color = '#27ae60'
    conv_color = '#3498db'
    pool_color = '#e67e22'
    se_color = '#9b59b6'
    spatial_color = '#e74c3c'
    fc_color = '#8e44ad'
    output_color = '#c0392b'

    x_pos = 0.5
    y = 1.0

    # Input
    create_block(ax, x_pos, y, 0.9, 0.8, 'Input', '64×64', input_color)
    x_pos += 0.7
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # Blocks 1-4 with dual attention
    for i, (ch, label) in enumerate([(64, 'Block 1'), (128, 'Block 2'), (256, 'Block 3'), (512, 'Block 4')]):
        create_block(ax, x_pos, y, 1.0, 0.8, label, f'{ch} ch', conv_color)
        # SE badge
        create_attention_badge(ax, x_pos - 0.25, y - 0.55, 'SE', se_color)
        # Spatial badge
        create_attention_badge(ax, x_pos + 0.25, y - 0.55, 'Sp', spatial_color)
        x_pos += 0.75
        create_arrow(ax, (x_pos, y), (x_pos + 0.3, y))
        x_pos += 0.55
        create_block(ax, x_pos, y, 0.6, 0.5, 'Pool', None, pool_color, fontsize=8)
        x_pos += 0.5
        create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
        x_pos += 0.65

    # Global Pool
    create_block(ax, x_pos, y, 0.8, 0.6, 'GAP', None, pool_color, fontsize=9)
    x_pos += 0.6
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # FC
    create_block(ax, x_pos, y, 0.9, 0.7, 'FC', '512→128', fc_color, fontsize=9)
    x_pos += 0.7
    create_arrow(ax, (x_pos, y), (x_pos + 0.4, y))
    x_pos += 0.7

    # Output
    create_block(ax, x_pos, y, 1.0, 0.8, 'Output', '6 classes', output_color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'v3_architecture.png'), dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: v3_architecture.png")


def generate_comparison_diagram():
    """Generate stacked comparison of all architectures."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 8))

    # Colors
    input_color = '#27ae60'
    conv_color = '#3498db'
    pool_color = '#e67e22'
    se_color = '#9b59b6'
    spatial_color = '#e74c3c'
    fc_color = '#8e44ad'
    output_color = '#c0392b'

    models = [
        ('V1: Baseline CNN (66.8%)', 3, False, False),
        ('V2: SE Attention (68.7%)', 4, True, False),
        ('V3: Dual Attention (71.7%)', 4, True, True),
    ]

    for ax, (title, num_blocks, has_se, has_spatial) in zip(axes, models):
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.3, 1.8)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title on left
        ax.text(-0.3, 0.8, title, ha='left', va='center',
                fontsize=12, fontweight='bold', color='#2c3e50')

        x_pos = 2.5
        y = 0.8

        # Input
        create_block(ax, x_pos, y, 0.7, 0.6, 'Input', None, input_color, fontsize=8)
        x_pos += 0.55
        create_arrow(ax, (x_pos, y), (x_pos + 0.3, y))
        x_pos += 0.55

        # Blocks
        channels = [64, 128, 256] if num_blocks == 3 else [64, 128, 256, 512]
        for i, ch in enumerate(channels):
            create_block(ax, x_pos, y, 0.8, 0.6, f'B{i+1}', f'{ch}', conv_color, fontsize=8)
            if has_se and has_spatial:
                create_attention_badge(ax, x_pos - 0.18, y - 0.45, 'SE', se_color)
                create_attention_badge(ax, x_pos + 0.18, y - 0.45, 'Sp', spatial_color)
            elif has_se:
                create_attention_badge(ax, x_pos, y - 0.45, 'SE', se_color)
            x_pos += 0.6
            create_arrow(ax, (x_pos, y), (x_pos + 0.2, y))
            x_pos += 0.4
            create_block(ax, x_pos, y, 0.4, 0.4, 'P', None, pool_color, fontsize=7)
            x_pos += 0.35
            create_arrow(ax, (x_pos, y), (x_pos + 0.25, y))
            x_pos += 0.5

        # Global Pool
        create_block(ax, x_pos, y, 0.6, 0.5, 'GAP', None, pool_color, fontsize=8)
        x_pos += 0.5
        create_arrow(ax, (x_pos, y), (x_pos + 0.3, y))
        x_pos += 0.55

        # FC
        create_block(ax, x_pos, y, 0.7, 0.5, 'FC', None, fc_color, fontsize=8)
        x_pos += 0.55
        create_arrow(ax, (x_pos, y), (x_pos + 0.3, y))
        x_pos += 0.55

        # Output
        create_block(ax, x_pos, y, 0.8, 0.6, 'Out', '6', output_color, fontsize=8)

    plt.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'architecture_comparison.png'), dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: architecture_comparison.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)

    print("\nGenerating horizontal architecture diagrams...")
    generate_v1_diagram()
    generate_v2_diagram()
    generate_v3_diagram()
    generate_comparison_diagram()

    print("\n" + "=" * 50)
    print(f"All diagrams saved to: {OUTPUT_DIR}")
    print("=" * 50)

    # List files
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
