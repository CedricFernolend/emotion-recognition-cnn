"""Spatial Attention Visualization - Extract and visualize learned attention maps."""

import torch
import numpy as np
import cv2


class SpatialAttentionExtractor:
    """Extract spatial attention maps from EmotionCNN model.

    Unlike GradCAM which approximates importance via gradients, this directly
    extracts the learned spatial attention maps from the model's SpatialAttention
    modules - showing exactly where the model focuses.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.attention_maps = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all SpatialAttention modules."""

        def make_hook(name):
            def hook(module, input, output):
                # The spatial attention module computes: x * attention_map
                # We can get the attention map by: output / input (where input > 0)
                # But safer: recompute the attention map directly
                x = input[0]
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                combined = torch.cat([avg_out, max_out], dim=1)
                attn = torch.sigmoid(module.conv(combined))
                self.attention_maps[name] = attn.detach().cpu()
            return hook

        # Hook into each layer's spatial attention
        self.model.layer1.spatial.register_forward_hook(make_hook('layer1'))
        self.model.layer2.spatial.register_forward_hook(make_hook('layer2'))
        self.model.layer3.spatial.register_forward_hook(make_hook('layer3'))
        self.model.layer4.spatial.register_forward_hook(make_hook('layer4'))

    def extract(self, image_tensor, target_class=None):
        """Extract attention maps for an input image.

        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Optional class index (not used, kept for API consistency)

        Returns:
            dict: Attention maps for each layer
            int: Predicted class index
        """
        self.attention_maps = {}

        with torch.no_grad():
            output = self.model(image_tensor)
            pred_idx = output.argmax(dim=1).item()

        # Convert attention maps to numpy
        result = {}
        for name, attn in self.attention_maps.items():
            # attn shape: (1, 1, H, W) -> (H, W)
            result[name] = attn[0, 0].numpy()

        return result, pred_idx

    def get_combined_attention(self, image_tensor, weights=None):
        """Get a weighted combination of attention maps from all layers.

        Args:
            image_tensor: Preprocessed image tensor
            weights: Optional dict of layer weights. Default emphasizes deeper layers.

        Returns:
            np.ndarray: Combined attention heatmap normalized to [0, 1]
            int: Predicted class index
        """
        if weights is None:
            # Default: weight deeper layers more (they're more semantic)
            weights = {'layer1': 0.1, 'layer2': 0.2, 'layer3': 0.3, 'layer4': 0.4}

        attention_maps, pred_idx = self.extract(image_tensor)

        # Find the largest spatial size to upsample to
        max_size = max(attn.shape[0] for attn in attention_maps.values())

        combined = np.zeros((max_size, max_size), dtype=np.float32)

        for name, attn in attention_maps.items():
            # Upsample to common size
            if attn.shape[0] != max_size:
                attn = cv2.resize(attn, (max_size, max_size), interpolation=cv2.INTER_LINEAR)
            combined += weights.get(name, 0.25) * attn

        # Normalize to [0, 1]
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        return combined, pred_idx

    def get_layer_attention(self, image_tensor, layer='layer4'):
        """Get attention map from a specific layer.

        Args:
            image_tensor: Preprocessed image tensor
            layer: Which layer to extract from ('layer1', 'layer2', 'layer3', 'layer4')

        Returns:
            np.ndarray: Attention heatmap normalized to [0, 1]
            int: Predicted class index
        """
        attention_maps, pred_idx = self.extract(image_tensor)
        attn = attention_maps.get(layer, attention_maps['layer4'])

        # Normalize to [0, 1]
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        return attn, pred_idx


class MultiScaleAttentionVisualizer:
    """Visualize attention maps at multiple scales side-by-side."""

    def __init__(self, model):
        self.extractor = SpatialAttentionExtractor(model)

    def create_visualization(self, image_tensor, face_bgr, target_size=(640, 480)):
        """Create a multi-scale attention visualization.

        Args:
            image_tensor: Preprocessed image tensor
            face_bgr: Original face crop in BGR format
            target_size: Output image size (width, height)

        Returns:
            np.ndarray: Visualization image in BGR format
            str: Predicted emotion label
        """
        from config import EMOTION_LABELS

        attention_maps, pred_idx = self.extractor.extract(image_tensor)

        # Create output canvas
        width, height = target_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Layout: original face on left, 4 attention maps in 2x2 grid on right
        face_width = width // 3
        face_height = height

        # Resize and place original face
        face_resized = cv2.resize(face_bgr, (face_width, face_height))
        canvas[:, :face_width] = face_resized

        # Place attention maps in 2x2 grid
        grid_width = (width - face_width) // 2
        grid_height = height // 2

        layer_positions = [
            ('layer1', 0, 0),
            ('layer2', 1, 0),
            ('layer3', 0, 1),
            ('layer4', 1, 1),
        ]

        for layer_name, col, row in layer_positions:
            attn = attention_maps[layer_name]

            # Normalize and colorize
            attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            attn_color = cv2.applyColorMap(np.uint8(255 * attn_norm), cv2.COLORMAP_JET)
            attn_resized = cv2.resize(attn_color, (grid_width, grid_height),
                                       interpolation=cv2.INTER_NEAREST)

            # Add label
            cv2.putText(attn_resized, layer_name, (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Resolution info
            h, w = attention_maps[layer_name].shape
            cv2.putText(attn_resized, f"{w}x{h}", (5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Place on canvas
            x_start = face_width + col * grid_width
            y_start = row * grid_height
            canvas[y_start:y_start+grid_height, x_start:x_start+grid_width] = attn_resized

        return canvas, EMOTION_LABELS[pred_idx]
