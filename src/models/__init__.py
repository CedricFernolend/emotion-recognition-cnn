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
    elif version == 'v3':
        from models.v3_model import EmotionCNN_V3
        return EmotionCNN_V3(num_classes=6, dropout_rate=dropout_rate)
    elif version == 'v4':
        from models.v4_final import EmotionCNN
        return EmotionCNN(num_classes=6, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model version: {version}")


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
        # Last conv in block3 for 3-block model
        return model.block3.body[-2]  # Conv2d before final BN
    elif version == 'v2':
        # V2 has 4 blocks with SE attention - target last conv in block4
        return model.block4.body[-2]  # Conv2d before final BN in block4
    elif version == 'v3':
        # Depends on architecture - update when defined
        if hasattr(model, 'block4'):
            return model.block4.body[-2]
        else:
            return model.block3.body[-2]
    else:  # v4
        # Last conv in layer4 for 4-block model
        return model.layer4.body[-3]


def get_model_info(version: str):
    """Get information about a model version."""
    from configs import load_config

    config = load_config(version)
    return {
        'version': config.VERSION,
        'name': config.VERSION_NAME,
        'description': config.VERSION_DESCRIPTION,
    }
