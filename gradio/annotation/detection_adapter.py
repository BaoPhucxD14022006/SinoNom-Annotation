from pathlib import Path
from .detection_import import load_detection


def model_assets_available(options):
    if options is None:
        return False
    cfg = getattr(options, 'vague_det_config', None)
    weights = getattr(options, 'vague_det_weights', None)
    ocr = getattr(options, 'ocr_det_executable', None)
    return bool(
        cfg and weights and ocr
        and Path(cfg).is_file()
        and Path(weights).is_file()
        and Path(ocr).is_file()
    )


def detect(image_path, options):
    path = Path(image_path)
    output_dir = getattr(options, 'output_dir', None) if options is not None else None
    if output_dir:
        uploaded = load_detection(path.name, output_dir)
        if uploaded is not None:
            return uploaded
    if getattr(options, 'skip_detection', False):
        raise ValueError('Detection is disabled (--skip-detection). Draw boxes manually.')
    source_mode = getattr(options, 'detection_source', 'json')
    if source_mode == 'json' and not model_assets_available(options):
        raise ValueError('No uploaded box JSON found for this image. Upload box JSON/ZIP in Step 1 or draw boxes manually.')
    from text_detection import run_detection_pipeline
    from text_detection.main import build_detection_document
    return build_detection_document(path, run_detection_pipeline(str(path), options))
