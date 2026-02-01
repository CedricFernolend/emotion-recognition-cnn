"""Model factory for multi-version support."""
import torch


def create_model(version: str, dropout_rate: float = 0.5):
    """
    Factory function to create model by version.

    Args:
        version: Model version ('v1', 'v2', 'v3', 'v4')
        dropout_rate: Dropout rate for the model (default: 0.5)

    Returns:
        Model instance (not loaded with weights)
    """
    if version == 'v1':
        from models.v1_baseline import EmotionCNN_V1
        return EmotionCNN_V1(num_classes=6, dropout_rate=dropout_rate)
    elif version == 'v2':
        from models.v2_model import EmotionCNN_V2
        return EmotionCNN_V2(num_classes=6, dropout_rate=dropout_rate)
    elif version == 'v3.5':
        from models.v3_5 import EmotionCNN
        return EmotionCNN(num_classes=6, dropout_rate=dropout_rate)
    elif version == 'v4':
        from models.v4_final import EmotionCNN
        return EmotionCNN(num_classes=6, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model version: {version}. Available: v1, v2, v3.5, v4")


def load_model(version: str, path: str = None):
    """
    Load a trained model.

    Args:
        version: Model version
        path: Optional path to model weights. If None, uses default from config.

    Returns:
        Model with loaded weights
    """
    from configs import load_config

    config = load_config(version)
    model = create_model(version)
    path = path or config.MODEL_SAVE_PATH

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))

    return model


def get_gradcam_target_layer(model, version: str):
    """
    Get the appropriate target layer for GradCAM based on model version.

    Args:
        model: The model instance
        version: Model version

    Returns:
        Target layer for GradCAM
    """
    if version == 'v1':
        # V1: 3-block model, target last conv in block3
        return model.block3.body[-2]  # Conv2d before final BN
    elif version == 'v2':
        # V2: 4-block model with SE, target last conv in block4
        return model.block4.body[-2]  # Conv2d before final BN
    elif version == 'v3.5':
        # V3.5: 4-block model without attention (same structure as v4)
        return model.layer4.body[-2]  # Conv2d before final BN
    elif version == 'v4':
        # V4: 4-block model with SE + Spatial attention
        # body[-2] is the last Conv2d (body[-3] was ReLU - wrong!)
        return model.layer4.body[-2]  # Conv2d before final BN
    else:
        raise ValueError(f"Unknown model version for GradCAM: {version}")


def get_model_info(version: str):
    """Get information about a model version."""
    from configs import load_config

    config = load_config(version)
    return {
        'version': config.VERSION,
        'name': config.VERSION_NAME,
        'description': config.VERSION_DESCRIPTION,
    }
