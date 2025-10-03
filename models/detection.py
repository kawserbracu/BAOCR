from typing import Any

def build_detection_model(pretrained: bool = True) -> Any:
    """
    Build a DocTR detection predictor with DB + MobileNetV3 Small backbone.
    Returns the predictor (inference module). Training routines can wrap this.
    """
    try:
        from doctr.models import detection_predictor
        # Use an architecture available in python-doctr 1.0.0
        model = detection_predictor(arch="db_resnet50", pretrained=pretrained)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to build detection model: {e}")
