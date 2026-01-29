"""
V2 Model: To Be Defined

This model will be designed after analyzing v1 results.
For now, it's a copy of v1 - modify based on v1 analysis.
"""
from models.v1_baseline import EmotionCNN_V1


# Placeholder: same as v1 until we define v2 improvements
class EmotionCNN_V2(EmotionCNN_V1):
    """V2 model - TBD after v1 analysis."""
    pass


def create_model():
    return EmotionCNN_V2(num_classes=6)
