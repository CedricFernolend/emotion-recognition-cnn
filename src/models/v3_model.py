"""
V3 Model: To Be Defined

This model will be designed after analyzing v2 results.
For now, it's a copy of v1 - modify based on v2 analysis.
"""
from models.v1_baseline import EmotionCNN_V1


# Placeholder: same as v1 until we define v3 improvements
class EmotionCNN_V3(EmotionCNN_V1):
    """V3 model - TBD after v2 analysis."""
    pass


def create_model():
    return EmotionCNN_V3(num_classes=6)
