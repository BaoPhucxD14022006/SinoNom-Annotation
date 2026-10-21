"""Save and restore annotation progress archives (.zip) and split final export archives."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import tempfile
import zipfile

from PIL import Image

from .detection_import import (MAX_UPLOAD_FILES, MAX_UPLOAD_FILE_BYTES,
                               MAX_UPLOAD_TOTAL_BYTES, _is_ignored_zip_member,
                               _resolve_images, _validate_safe_zip_member,
                               normalize_detection_document)
from .export import (collect_annotations, collect_content_documents,
                     collect_source_mismatches)
from .io import (_unique, atomic_write, issue_types, load_annotation,
                 load_source_mismatch, read_json, validate_document,
                 validate_source_mismatch_document)
from .text_extraction import validate_content_document

EXPORT_JSON_NAMES = frozenset({
    'text_annotations.json',
    'inscription_content.json',
    'source_mismatches.json',
})
MANIFEST_NAME = 'progress_manifest.json'


def _fingerprint(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _document_hash(doc):
    return _fingerprint(json.dumps(doc, sort_keys=True, ensure_ascii=False))


def _annotations_to_text(annotations):
    if not isinstance(annotations, dict) or not annotations:
        return ''
    return ''.join(str(annotations[key]) for key in sorted(annotations, key=int))


def _read_safe_zip_json_entries(archive_path):
    path = Path(archive_path)
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError('Upload a valid .zip archive.')
    entries = {}
    total_bytes = 0
    file_count = 0
    with zipfile.ZipFile(path, 'r') as bundle:
        for info in bundle.infolist():
            posix = _validate_safe_zip_member(info.filename)
            if info.is_dir() or info.filename.endswith(('/', '\\')) or _is_ignored_zip_member(posix):
                continue
            file_count += 1
            if file_count > MAX_UPLOAD_FILES:
                raise ValueError('Archive contains too many files.')
            if posix.suffix.lower() != '.json':
                raise ValueError(f'Archive may only contain .json files: {info.filename}')
            if info.file_size > MAX_UPLOAD_FILE_BYTES:
                raise ValueError(f'Archive member exceeds size limit: {info.filename}')
            total_bytes += info.file_size
            if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                raise ValueError('Archive total uncompressed size exceeds limit.')
            with bundle.open(info, 'r') as handle:
                raw = handle.read(MAX_UPLOAD_FILE_BYTES + 1)
            if len(raw) > MAX_UPLOAD_FILE_BYTES:
                raise ValueError(f'Archive member exceeds size limit: {info.filename}')
            rel_key = posix.as_posix()
            if rel_key in entries:
                raise ValueError(f'Duplicate file in archive: {rel_key}')
            try:
                entries[rel_key] = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f'Invalid JSON in {rel_key}: {exc}') from exc
    if not entries:
        raise ValueError('Archive contains no .json files.')
    return entries


def _image_maps(images):
    image_list = _resolve_images(images)
    if not image_list:
        raise ValueError('The image folder is empty.')
    by_name = {}
    by_stem = {}
    size_cache = {}
    for p in image_list:
        resolved = Path(p).resolve()
        by_name[resolved.name] = resolved
        by_stem[resolved.stem] = resolved

    def get_size(img_path):
        if img_path not in size_cache:
            with Image.open(img_path) as im:
                size_cache[img_path] = list(im.size)
        return size_cache[img_path]

    return image_list, by_name, by_stem, get_size


def _has_existing_output_data(output_dir):
    output = Path(output_dir)
    if not output.is_dir():
        return False
    for child in output.iterdir():
        if child.name == '.backup':
            continue
        if child.is_file() and child.suffix.lower() == '.json':
            return True
        if child.is_dir() and child.name in ('source_mismatches', '.state'):
            if any(p.is_file() for p in child.rglob('*.json')):
                return True
    return False


def _backup_and_clear_output_dir(output_dir, perform_backup=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backup_dir = None
    if perform_backup and _has_existing_output_data(output):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup_dir = output / '.backup' / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        for child in list(output.iterdir()):
            if child.name == '.backup':
                continue
            if child.is_file() and child.suffix.lower() == '.json':
                shutil.copy2(child, backup_dir / child.name)
            elif child.is_dir() and child.name in ('source_mismatches', '.state'):
                shutil.copytree(child, backup_dir / child.name, dirs_exist_ok=True)

    for child in list(output.iterdir()):
        if child.name == '.backup':
            continue
        if child.is_file() and child.suffix.lower() == '.json':
            child.unlink(missing_ok=True)
        elif child.is_dir() and child.name in ('source_mismatches', '.state'):
            shutil.rmtree(child, ignore_errors=True)
    return backup_dir


def _normalize_output_and_images(first, second):
    if isinstance(first, (list, tuple, set)):
        return second, first
    if isinstance(second, (list, tuple, set)):
        return first, second
    return first, second


def build_progress_archive(output_dir, images, config_info=None, archive_path=None, titles=None):
    """Package committed annotations, mismatches, content, sidecar hashes, and uploaded detections into a ZIP."""
    output_dir, images = _normalize_output_and_images(output_dir, images)
    image_list = _resolve_images(images)
    if not image_list:
        raise ValueError('The image folder is empty.')
    output = Path(output_dir)
    annotations = collect_annotations(image_list, output, allow_empty=True)
    mismatches = collect_source_mismatches(image_list, output, allow_empty=True)
    committed_images = [
        img for img in image_list
        if ((output / f'{Path(img).stem}.json').is_file()
            or (output / 'source_mismatches' / f'{Path(img).stem}.json').is_file())
    ]
    content_docs = collect_content_documents(
        committed_images or image_list, output, allow_empty=True, titles=titles)

    files_to_write = {}
    completed_images = []

    for doc in annotations:
        stem = Path(doc['image']).stem
        completed_images.append(doc['image'])
        files_to_write[f'{stem}.json'] = doc
        meta_path = output / '.state' / f'{stem}.json'
        if meta_path.is_file():
            files_to_write[f'.state/{stem}.json'] = read_json(meta_path)
        else:
            files_to_write[f'.state/{stem}.json'] = {
                'text_hash': _fingerprint(_annotations_to_text(doc.get('annotations', {}))),
                'document_hash': _document_hash(doc),
            }

    for doc in mismatches:
        stem = Path(doc['image']).stem
        completed_images.append(doc['image'])
        files_to_write[f'source_mismatches/{stem}.json'] = doc
        meta_path = output / '.state' / 'source_mismatches' / f'{stem}.json'
        if meta_path.is_file():
            files_to_write[f'.state/source_mismatches/{stem}.json'] = read_json(meta_path)
        else:
            source_text = doc.get('source_text') or ''
            files_to_write[f'.state/source_mismatches/{stem}.json'] = {
                'text_hash': _fingerprint(source_text),
                'document_hash': _document_hash(doc),
            }

    registry_path = output / '.state' / 'content.json'
    if registry_path.is_file():
        raw_registry = read_json(registry_path)
        if isinstance(raw_registry, list) and raw_registry:
            files_to_write['.state/content.json'] = raw_registry
    elif content_docs:
        files_to_write['.state/content.json'] = content_docs

    detection_images = []
    det_dir = output / '.state' / 'detections'
    if det_dir.is_dir():
        by_stem = {Path(p).stem: Path(p) for p in image_list}
        for det_file in sorted(det_dir.glob('*.json')):
            if det_file.stem in by_stem:
                det_doc = read_json(det_file)
                files_to_write[f'.state/detections/{det_file.stem}.json'] = det_doc
                detection_images.append(by_stem[det_file.stem].name)

    if not files_to_write:
        raise ValueError('No saved annotations or uploaded detections to save.')

    manifest = {
        'format': 'sinonom-progress',
        'version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config_info or {},
        'completed_images': sorted(completed_images),
        'detection_images': sorted(detection_images),
    }
    files_to_write[MANIFEST_NAME] = manifest

    output.mkdir(parents=True, exist_ok=True)
    if archive_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive = output / f'progress_{timestamp}.zip'
    else:
        archive = Path(archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)

    temporary = archive.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for rel_path, payload in sorted(files_to_write.items()):
                bundle.writestr(rel_path, json.dumps(payload, ensure_ascii=False, indent=2))
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def _validate_annotation_object(doc, image_name, size):
    required = {'image', 'bounding_boxes', 'annotations', 'image_resize', 'crop'}
    if not isinstance(doc, dict) or not required.issubset(doc) or not set(doc).issubset(required):
        raise ValueError(f'{image_name}: invalid annotation document schema.')
    normalized = deepcopy(doc)
    normalized['image'] = image_name
    validate_document(normalized, image_name, size)
    if not normalized.get('annotations'):
        raise ValueError(f'{image_name}: character annotations are missing.')
    return normalized


def _validate_mismatch_object(doc, image_name, size):
    if not isinstance(doc, dict):
        raise ValueError(f'{image_name}: invalid source mismatch document.')
    normalized = deepcopy(doc)
    normalized['image'] = image_name
    normalized['inscription_code'] = Path(image_name).stem
    normalized['issue_type'] = issue_types(normalized.get('issue_type'))
    return validate_source_mismatch_document(normalized, image_name, size)


def _infer_text_hash(stem, doc, is_mismatch, content_by_stem, annotation_title=None):
    content_entry = content_by_stem.get(stem)
    if content_entry and isinstance(content_entry.get('content'), dict):
        heading = annotation_title or 'Nguyên văn chữ Hán Nôm'
        text = content_entry['content'].get(heading)
        if isinstance(text, str) and text:
            return _fingerprint(text)
    if is_mismatch and isinstance(doc.get('source_text'), str) and doc['source_text']:
        return _fingerprint(doc['source_text'])
    return _fingerprint(_annotations_to_text(doc.get('annotations', {})))


def split_export_archive(archive_path, output_dir, images, titles=None,
                         annotation_title=None, backup=True):
    """Restore per-image files in ``output_dir`` from an ``annotations.zip`` export archive."""
    output_dir, images = _normalize_output_and_images(output_dir, images)
    entries = _read_safe_zip_json_entries(archive_path)
    unknown_files = set(entries) - EXPORT_JSON_NAMES
    if unknown_files:
        raise ValueError(f'Unexpected files in export archive: {sorted(unknown_files)}')
    _, by_name, by_stem, get_size = _image_maps(images)

    raw_annotations = entries.get('text_annotations.json', [])
    raw_mismatches = entries.get('source_mismatches.json', [])
    raw_contents = entries.get('inscription_content.json', [])
    if (not isinstance(raw_annotations, list)
            or not isinstance(raw_mismatches, list)
            or not isinstance(raw_contents, list)):
        raise ValueError('Export archive files must contain top-level JSON arrays.')

    skipped = []
    valid_annotations = {}
    valid_mismatches = {}
    content_by_stem = {}
    content_order = []

    for item in raw_contents:
        if not isinstance(item, dict) or not isinstance(item.get('image'), str):
            raise ValueError('Invalid content record in inscription_content.json.')
        raw_img = item['image'].strip()
        token = Path(raw_img.replace('\\', '/')).name
        matched = by_name.get(token) or by_stem.get(Path(token).stem)
        if matched is None:
            skipped.append(raw_img)
            continue
        stem = matched.stem
        if stem in content_by_stem:
            raise ValueError(f'Duplicate content record for {matched.name}.')
        normalized = deepcopy(item)
        normalized['image'] = matched.name
        normalized['inscription_code'] = stem
        validate_content_document(normalized, matched.name, None)
        if titles is not None:
            normalized['content'] = {t: normalized['content'].get(t) for t in titles}
            validate_content_document(normalized, matched.name, titles)
        content_by_stem[stem] = normalized
        content_order.append(stem)

    for item in raw_annotations:
        if not isinstance(item, dict) or not isinstance(item.get('image'), str):
            raise ValueError('Invalid annotation record in text_annotations.json.')
        raw_img = item['image'].strip()
        token = Path(raw_img.replace('\\', '/')).name
        matched = by_name.get(token) or by_stem.get(Path(token).stem)
        if matched is None:
            skipped.append(raw_img)
            continue
        stem = matched.stem
        if stem in valid_annotations:
            raise ValueError(f'Duplicate annotation record for {matched.name}.')
        valid_annotations[stem] = _validate_annotation_object(item, matched.name, get_size(matched))

    for item in raw_mismatches:
        if not isinstance(item, dict) or not isinstance(item.get('image'), str):
            raise ValueError('Invalid source mismatch record in source_mismatches.json.')
        raw_img = item['image'].strip()
        token = Path(raw_img.replace('\\', '/')).name
        matched = by_name.get(token) or by_stem.get(Path(token).stem)
        if matched is None:
            skipped.append(raw_img)
            continue
        stem = matched.stem
        if stem in valid_annotations:
            raise ValueError(f'Both annotation and source mismatch exist for {matched.name}.')
        if stem in valid_mismatches:
            raise ValueError(f'Duplicate source mismatch record for {matched.name}.')
        valid_mismatches[stem] = _validate_mismatch_object(item, matched.name, get_size(matched))

    output = Path(output_dir)
    existing_detections = {}
    det_dir = output / '.state' / 'detections'
    if det_dir.is_dir():
        for det_file in det_dir.glob('*.json'):
            try:
                existing_detections[det_file.stem] = read_json(det_file)
            except Exception:
                pass

    backup_dir = _backup_and_clear_output_dir(output, perform_backup=backup)

    for stem, doc in valid_annotations.items():
        atomic_write(output / f'{stem}.json', doc)
        atomic_write(output / '.state' / f'{stem}.json', {
            'text_hash': _infer_text_hash(stem, doc, False, content_by_stem, annotation_title),
            'document_hash': _document_hash(doc),
        })

    for stem, doc in valid_mismatches.items():
        atomic_write(output / 'source_mismatches' / f'{stem}.json', doc)
        atomic_write(output / '.state' / 'source_mismatches' / f'{stem}.json', {
            'text_hash': _infer_text_hash(stem, doc, True, content_by_stem, annotation_title),
            'document_hash': _document_hash(doc),
        })

    if content_order:
        atomic_write(output / '.state' / 'content.json',
                     [content_by_stem[stem] for stem in content_order])

    for stem, det_doc in existing_detections.items():
        atomic_write(output / '.state' / 'detections' / f'{stem}.json', det_doc)

    skipped_sorted = sorted(set(skipped))
    return {
        'mode': 'export',
        'restored_annotations': len(valid_annotations) + len(valid_mismatches),
        'restored_normal_annotations': len(valid_annotations),
        'restored_mismatches': len(valid_mismatches),
        'restored_content': len(content_order),
        'restored_detections': 0,
        'skipped': skipped_sorted,
        'warnings': [f'Skipped unknown images: {", ".join(skipped_sorted)}'] if skipped_sorted else [],
        'backup_dir': str(backup_dir) if backup_dir else None,
    }


def restore_progress_archive(archive_path, output_dir, images, titles=None,
                             annotation_title=None, backup=True):
    """Validate and restore a progress ZIP (or an ``annotations.zip`` export archive) into ``output_dir``."""
    output_dir, images = _normalize_output_and_images(output_dir, images)
    entries = _read_safe_zip_json_entries(archive_path)
    if set(entries).issubset(EXPORT_JSON_NAMES):
        return split_export_archive(
            archive_path, output_dir, images,
            titles=titles, annotation_title=annotation_title, backup=backup)

    _, by_name, by_stem, get_size = _image_maps(images)
    skipped = []
    valid_annotations = {}
    valid_mismatches = {}
    annotation_metas = {}
    mismatch_metas = {}
    valid_detections = {}
    content_by_stem = {}
    content_order = []

    for rel_path, data in entries.items():
        parts = PurePosixPath(rel_path).parts
        if rel_path == MANIFEST_NAME:
            if not isinstance(data, dict):
                raise ValueError('Invalid progress_manifest.json.')
            continue

        if len(parts) == 1 and parts[0].endswith('.json'):
            stem = Path(parts[0]).stem
            matched = by_stem.get(stem)
            if matched is None:
                skipped.append(data.get('image', parts[0]) if isinstance(data, dict) else parts[0])
                continue
            if stem in valid_mismatches:
                raise ValueError(f'Both annotation and source mismatch exist for {matched.name}.')
            valid_annotations[stem] = _validate_annotation_object(data, matched.name, get_size(matched))
            continue

        if len(parts) == 2 and parts[0] == 'source_mismatches' and parts[1].endswith('.json'):
            stem = Path(parts[1]).stem
            matched = by_stem.get(stem)
            if matched is None:
                skipped.append(data.get('image', parts[1]) if isinstance(data, dict) else parts[1])
                continue
            if stem in valid_annotations:
                raise ValueError(f'Both annotation and source mismatch exist for {matched.name}.')
            valid_mismatches[stem] = _validate_mismatch_object(data, matched.name, get_size(matched))
            continue

        if rel_path == '.state/content.json':
            if not isinstance(data, list):
                raise ValueError('The saved-content registry must be a JSON array.')
            for item in data:
                if not isinstance(item, dict) or not isinstance(item.get('image'), str):
                    raise ValueError('Invalid saved content document in .state/content.json.')
                raw_img = item['image'].strip()
                token = Path(raw_img.replace('\\', '/')).name
                matched = by_name.get(token) or by_stem.get(Path(token).stem)
                if matched is None:
                    skipped.append(raw_img)
                    continue
                stem = matched.stem
                if stem in content_by_stem:
                    raise ValueError(f'Duplicate saved content for {matched.name}.')
                normalized = deepcopy(item)
                normalized['image'] = matched.name
                normalized['inscription_code'] = stem
                validate_content_document(normalized, matched.name, None)
                if titles is not None:
                    normalized['content'] = {t: normalized['content'].get(t) for t in titles}
                    validate_content_document(normalized, matched.name, titles)
                content_by_stem[stem] = normalized
                content_order.append(stem)
            continue

        if len(parts) == 2 and parts[0] == '.state' and parts[1].endswith('.json'):
            stem = Path(parts[1]).stem
            if isinstance(data, dict):
                annotation_metas[stem] = data
            continue

        if (len(parts) == 3 and parts[0] == '.state'
                and parts[1] == 'source_mismatches' and parts[2].endswith('.json')):
            stem = Path(parts[2]).stem
            if isinstance(data, dict):
                mismatch_metas[stem] = data
            continue

        if (len(parts) == 3 and parts[0] == '.state'
                and parts[1] == 'detections' and parts[2].endswith('.json')):
            stem = Path(parts[2]).stem
            matched = by_stem.get(stem)
            if matched is None:
                skipped.append(data.get('image', parts[2]) if isinstance(data, dict) else parts[2])
                continue
            valid_detections[stem] = normalize_detection_document(
                data, matched.name, size=get_size(matched))
            continue

        raise ValueError(f'Unexpected file in progress archive: {rel_path}')

    output = Path(output_dir)
    backup_dir = _backup_and_clear_output_dir(output, perform_backup=backup)

    for stem, doc in valid_annotations.items():
        atomic_write(output / f'{stem}.json', doc)
        meta = annotation_metas.get(stem, {})
        text_hash = (meta.get('text_hash')
                     if isinstance(meta.get('text_hash'), str) and meta.get('text_hash')
                     else _infer_text_hash(stem, doc, False, content_by_stem, annotation_title))
        atomic_write(output / '.state' / f'{stem}.json', {
            'text_hash': text_hash,
            'document_hash': _document_hash(doc),
        })

    for stem, doc in valid_mismatches.items():
        atomic_write(output / 'source_mismatches' / f'{stem}.json', doc)
        meta = mismatch_metas.get(stem, {})
        text_hash = (meta.get('text_hash')
                     if isinstance(meta.get('text_hash'), str) and meta.get('text_hash')
                     else _infer_text_hash(stem, doc, True, content_by_stem, annotation_title))
        atomic_write(output / '.state' / 'source_mismatches' / f'{stem}.json', {
            'text_hash': text_hash,
            'document_hash': _document_hash(doc),
        })

    if content_order:
        atomic_write(output / '.state' / 'content.json',
                     [content_by_stem[stem] for stem in content_order])

    for stem, det_doc in valid_detections.items():
        atomic_write(output / '.state' / 'detections' / f'{stem}.json', det_doc)

    skipped_sorted = sorted(set(skipped))
    return {
        'mode': 'progress',
        'restored_annotations': len(valid_annotations) + len(valid_mismatches),
        'restored_normal_annotations': len(valid_annotations),
        'restored_mismatches': len(valid_mismatches),
        'restored_content': len(content_order),
        'restored_detections': len(valid_detections),
        'skipped': skipped_sorted,
        'warnings': [f'Skipped unknown images: {", ".join(skipped_sorted)}'] if skipped_sorted else [],
        'backup_dir': str(backup_dir) if backup_dir else None,
    }
