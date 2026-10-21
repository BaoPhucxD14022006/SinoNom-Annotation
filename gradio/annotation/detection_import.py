"""Import and persist precomputed character detection bounding boxes from JSON or ZIP."""
import json
import math
import zipfile
from pathlib import PurePosixPath, Path
from PIL import Image
from .bbox import validate_coordinates
from .io import _unique, atomic_write, load_image_list, read_json

MAX_UPLOAD_FILE_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 200 * 1024 * 1024
MAX_UPLOAD_FILES = 5000


class DetectionUploadResult(dict):
    """Mapping of ``{stem: detection_doc}`` with ``.skipped`` image names."""

    def __init__(self, *args, skipped=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped = list(skipped or [])

    @property
    def records(self):
        return self

    @property
    def skipped_images(self):
        return self.skipped

    @property
    def warnings(self):
        if not self.skipped:
            return []
        return [f'Skipped unknown images: {", ".join(self.skipped)}']


def detections_dir(output_dir):
    return Path(output_dir) / '.state' / 'detections'


def _resolve_images(images):
    if images is None:
        return None
    if isinstance(images, (str, Path)):
        path = Path(images)
        if path.is_dir():
            return load_image_list(path)
        return [path]
    return [Path(p) for p in images]


def _validate_safe_zip_member(name):
    norm = name.replace('\\', '/')
    posix = PurePosixPath(norm)
    if (norm.startswith('/') or norm.startswith('\\')
            or ':' in norm or '..' in posix.parts):
        raise ValueError(f'Unsafe path in ZIP archive: {name}')
    return posix


def _is_ignored_zip_member(posix_path):
    parts = posix_path.parts
    if not parts:
        return True
    if '__MACOSX' in parts or posix_path.name == '.DS_Store':
        return True
    return False


def _read_raw_detection_records(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Detection upload file does not exist: {path}')
    suffix = path.suffix.lower()
    if suffix == '.json':
        if path.stat().st_size > MAX_UPLOAD_FILE_BYTES:
            raise ValueError('Detection JSON file exceeds maximum allowed size.')
        data = read_json(path)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            if not data:
                raise ValueError('Detection JSON array is empty.')
            return data
        raise ValueError('Detection JSON must be an object or an array of objects.')
    if suffix == '.zip':
        if not zipfile.is_zipfile(path):
            raise ValueError('Invalid ZIP archive.')
        records = []
        total_bytes = 0
        file_count = 0
        with zipfile.ZipFile(path, 'r') as archive:
            for info in archive.infolist():
                posix = _validate_safe_zip_member(info.filename)
                if info.is_dir() or info.filename.endswith(('/', '\\')) or _is_ignored_zip_member(posix):
                    continue
                file_count += 1
                if file_count > MAX_UPLOAD_FILES:
                    raise ValueError('ZIP archive contains too many files.')
                if posix.suffix.lower() != '.json':
                    raise ValueError(f'ZIP archive may only contain .json files: {info.filename}')
                if info.file_size > MAX_UPLOAD_FILE_BYTES:
                    raise ValueError(f'File in ZIP archive is too large: {info.filename}')
                total_bytes += info.file_size
                if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                    raise ValueError('ZIP archive total uncompressed size exceeds limit.')
                with archive.open(info, 'r') as handle:
                    raw_bytes = handle.read(MAX_UPLOAD_FILE_BYTES + 1)
                if len(raw_bytes) > MAX_UPLOAD_FILE_BYTES:
                    raise ValueError(f'File in ZIP archive exceeds size limit: {info.filename}')
                try:
                    parsed = json.loads(raw_bytes.decode('utf-8'), object_pairs_hook=_unique)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f'Invalid JSON in {info.filename}: {exc}') from exc
                if posix.name in ('inscription_content.json', 'progress_manifest.json', 'content.json'):
                    continue
                if isinstance(parsed, dict):
                    if 'bounding_boxes' not in parsed:
                        continue
                    if not parsed.get('image'):
                        parsed = dict(parsed, image=posix.stem)
                    records.append(parsed)
                elif isinstance(parsed, list):
                    records.extend(
                        item for item in parsed
                        if not isinstance(item, dict) or 'bounding_boxes' in item
                    )
                else:
                    raise ValueError(f'JSON in {info.filename} must be an object or array.')
        if not records:
            raise ValueError('ZIP archive contains no detection JSON documents.')
        return records
    raise ValueError('Unsupported file format. Upload a .json or .zip file.')


def _normalize_box_coordinates(raw_bbox, size=None, label=''):
    prefix = f'{label}: ' if label else ''
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError(f'{prefix}A bounding box must contain four coordinates.')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in raw_bbox):
        raise ValueError(f'{prefix}Coordinates must be finite numbers.')
    coords = [int(round(v)) for v in raw_bbox]
    if size is not None:
        return validate_coordinates(coords, size)
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError(f'{prefix}Bounding box has invalid dimensions.')
    return coords


def normalize_detection_document(raw_doc, canonical_image_name, size=None):
    if not isinstance(raw_doc, dict):
        raise ValueError('Each detection entry must be a JSON object.')
    raw_boxes = raw_doc.get('bounding_boxes')
    if not isinstance(raw_boxes, dict):
        raise ValueError(f'{canonical_image_name}: bounding_boxes must be an object.')
    items = list(raw_boxes.items())
    if all(isinstance(k, str) and k.isdecimal() for k, _ in items):
        items.sort(key=lambda pair: int(pair[0]))
    normalized_boxes = {}
    for index, (orig_key, box) in enumerate(items, 1):
        if isinstance(box, (list, tuple)):
            box = {'bbox': list(box)}
        if not isinstance(orig_key, str) or not orig_key.strip() or not isinstance(box, dict):
            raise ValueError(f'{canonical_image_name}: invalid bounding box entry {orig_key!r}.')
        coords = _normalize_box_coordinates(
            box.get('bbox'), size=size, label=f'{canonical_image_name} box {orig_key}')
        status = box.get('status', 'intact')
        if status is None:
            status = 'intact'
        if status not in ('intact', 'damaged'):
            raise ValueError(f'{canonical_image_name}: invalid status {status!r} for box {orig_key}.')
        normalized_boxes[str(index)] = {
            'bbox': coords,
            'status': status,
        }
    return {
        'image': canonical_image_name,
        'bounding_boxes': normalized_boxes,
    }


def parse_detection_upload(path, images=None):
    """Parse a JSON or ZIP upload into ``{stem: detection_doc}`` and validate against images."""
    raw_records = _read_raw_detection_records(path)
    image_list = _resolve_images(images)
    by_name = {}
    by_stem = {}
    size_cache = {}
    if image_list is not None:
        for img_path in image_list:
            resolved = Path(img_path).resolve()
            by_name[resolved.name] = resolved
            by_stem[resolved.stem] = resolved

    docs = {}
    skipped = []
    for record in raw_records:
        if not isinstance(record, dict):
            raise ValueError('Each detection entry must be a JSON object.')
        raw_image = record.get('image')
        if not isinstance(raw_image, str) or not raw_image.strip():
            raise ValueError('Detection document is missing a valid "image" field.')
        image_token = Path(raw_image.strip().replace('\\', '/')).name
        stem_token = Path(image_token).stem
        if not stem_token:
            raise ValueError(f'Invalid image name in detection document: {raw_image!r}')

        if image_list is not None:
            matched = by_name.get(image_token) or by_stem.get(stem_token)
            if matched is None:
                skipped.append(raw_image.strip())
                continue
            stem = matched.stem
            canonical_name = matched.name
            if matched not in size_cache:
                with Image.open(matched) as im:
                    size_cache[matched] = list(im.size)
            size = size_cache[matched]
        else:
            stem = stem_token
            canonical_name = image_token
            size = None

        docs[stem] = normalize_detection_document(record, canonical_name, size=size)

    return DetectionUploadResult(docs, skipped=skipped)


def store_detections(docs, output_dir):
    """Persist validated detection documents under ``output_dir/.state/detections/<stem>.json``."""
    if not isinstance(docs, dict):
        raise ValueError('Detections to store must be a mapping of {stem: document}.')
    target_dir = detections_dir(output_dir)
    written = {}
    for stem, doc in docs.items():
        if not isinstance(stem, str) or not stem.strip() or '/' in stem or '\\' in stem or '..' in stem:
            raise ValueError(f'Invalid image stem: {stem!r}')
        image_name = doc.get('image') if isinstance(doc, dict) else None
        normalized = normalize_detection_document(doc, image_name or f'{stem}.jpg')
        dest = target_dir / f'{stem}.json'
        atomic_write(dest, normalized)
        written[stem] = dest
    return written


def load_detection(stem_or_image, output_dir, image_size=None):
    """Load a stored detection document for ``stem_or_image``, or ``None`` if absent."""
    if not output_dir or not stem_or_image:
        return None
    name = Path(str(stem_or_image)).name
    stem = Path(name).stem
    path = detections_dir(output_dir) / f'{stem}.json'
    if not path.is_file():
        return None
    raw_doc = read_json(path)
    canonical_name = name if '.' in name else raw_doc.get('image', f'{stem}.jpg')
    return normalize_detection_document(raw_doc, canonical_name, size=image_size)


def has_detection(stem_or_image, output_dir):
    if not output_dir or not stem_or_image:
        return False
    stem = Path(str(stem_or_image)).stem
    return bool(stem and (detections_dir(output_dir) / f'{stem}.json').is_file())


def count_stored_detections(images, output_dir):
    if not output_dir or not images:
        return 0
    return sum(1 for image in images if has_detection(Path(image).stem, output_dir))
