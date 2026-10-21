import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation.detection_adapter import detect
from annotation.detection_import import (
    count_stored_detections,
    has_detection,
    load_detection,
    parse_detection_upload,
    store_detections,
)
from annotation.workflow import Workflow


class DetectionImportTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.image_dir = self.root / 'images'
        self.output_dir = self.root / 'output'
        self.image_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        self.images = []
        for stem in ('12305', '12306'):
            img_path = self.image_dir / f'{stem}.jpg'
            Image.new('RGB', (200, 300), color=(255, 255, 255)).save(img_path)
            self.images.append(img_path)

        self.source_json = self.root / 'source.json'
        self.source_json.write_text(json.dumps([
            {
                'noi_dung': [
                    {
                        'ky_hieu': '12305',
                        'chuyen_muc': [
                            {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '永樂'}
                        ],
                    },
                    {
                        'ky_hieu': '12306',
                        'chuyen_muc': [
                            {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '大'}
                        ],
                    },
                ]
            }
        ], ensure_ascii=False), encoding='utf-8')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parse_single_json_array_and_zip(self):
        single_file = self.root / 'single.json'
        single_file.write_text(json.dumps({
            'image': '12305.jpg',
            'bounding_boxes': {
                '1': {'bbox': [10.2, 20.7, 50.1, 80.9]},
                '2': {'bbox': [60, 20, 100, 80], 'status': 'damaged', 'damage_level': 2},
            },
        }), encoding='utf-8')
        res_single = parse_detection_upload(single_file, self.images)
        self.assertEqual(set(res_single.records), {'12305'})
        self.assertEqual(
            res_single.records['12305']['bounding_boxes']['1'],
            {'bbox': [10, 21, 50, 81], 'status': 'intact'},
        )
        self.assertEqual(
            res_single.records['12305']['bounding_boxes']['2']['status'],
            'damaged',
        )

        array_file = self.root / 'array.json'
        array_file.write_text(json.dumps([
            {
                'image': '12305.jpg',
                'bounding_boxes': {'1': {'bbox': [10, 20, 50, 80]}},
            },
            {
                'image': '12306',
                'bounding_boxes': {'1': [15, 25, 55, 85]},
            },
        ]), encoding='utf-8')
        res_array = parse_detection_upload(array_file, self.images)
        self.assertEqual(set(res_array.records), {'12305', '12306'})
        self.assertEqual(res_array.records['12306']['image'], '12306.jpg')

        zip_file = self.root / 'boxes.zip'
        with zipfile.ZipFile(zip_file, 'w') as zf:
            zf.writestr('12305.json', json.dumps({
                'bounding_boxes': {'1': {'bbox': [1, 2, 30, 40]}},
            }))
            zf.writestr('nested/12306.json', json.dumps({
                'image': '12306.jpg',
                'bounding_boxes': {'1': {'bbox': [5, 6, 40, 50]}},
            }))
        res_zip = parse_detection_upload(zip_file, self.images)
        self.assertEqual(set(res_zip.records), {'12305', '12306'})

        export_zip = self.root / 'annotations_export.zip'
        with zipfile.ZipFile(export_zip, 'w') as zf:
            zf.writestr('text_annotations.json', json.dumps([
                {
                    'image': '12305.jpg',
                    'bounding_boxes': {'1': {'bbox': [10, 10, 40, 40], 'status': 'intact'}},
                }
            ]))
            zf.writestr('inscription_content.json', json.dumps([
                {
                    'image': '12305.jpg',
                    'inscription_code': '12305',
                    'content': {'Nguyên văn chữ Hán Nôm': '永'},
                }
            ]))
            zf.writestr('source_mismatches.json', json.dumps([]))
        res_export = parse_detection_upload(export_zip, self.images)
        self.assertEqual(set(res_export.records), {'12305'})

    def test_out_of_bounds_and_unknown_image_handling(self):
        bad_file = self.root / 'bad_coords.json'
        bad_file.write_text(json.dumps({
            'image': '12305.jpg',
            'bounding_boxes': {'1': {'bbox': [10, 20, 250, 80]}},
        }), encoding='utf-8')
        with self.assertRaises(ValueError):
            parse_detection_upload(bad_file, self.images)

        mixed_file = self.root / 'mixed.json'
        mixed_file.write_text(json.dumps([
            {
                'image': '99999.jpg',
                'bounding_boxes': {'1': {'bbox': [10, 20, 50, 80]}},
            },
            {
                'image': '12305.jpg',
                'bounding_boxes': {'1': {'bbox': [10, 20, 50, 80]}},
            },
        ]), encoding='utf-8')
        res = parse_detection_upload(mixed_file, self.images)
        self.assertEqual(set(res.records), {'12305'})
        self.assertEqual(res.skipped_images, ['99999.jpg'])
        self.assertTrue(res.warnings)

    def test_skip_detection_with_uploaded_json_and_saved_annotation_priority(self):
        upload_file = self.root / 'upload.json'
        upload_file.write_text(json.dumps({
            'image': '12305.jpg',
            'bounding_boxes': {
                '1': {'bbox': [10, 10, 50, 50], 'status': 'intact'},
                '2': {'bbox': [10, 60, 50, 100], 'status': 'intact'},
            },
        }), encoding='utf-8')
        parsed = parse_detection_upload(upload_file, self.images)
        store_detections(parsed.records, self.output_dir)
        self.assertEqual(count_stored_detections(self.images, self.output_dir), 1)
        self.assertTrue(has_detection('12305.jpg', self.output_dir))
        self.assertIsNotNone(load_detection('12305.jpg', self.output_dir))

        opts = SimpleNamespace(
            image_dir=self.image_dir,
            source_json=self.source_json,
            output_dir=self.output_dir,
            content_titles=('Nguyên văn chữ Hán Nôm',),
            annotation_title='Nguyên văn chữ Hán Nôm',
            skip_detection=True,
            detection_source='json',
        )
        engine = Workflow(opts)

        # Open image -> verify content -> Next to Step 3 -> detect from JSON works even with skip_detection=True
        state = engine.open_image(self.images[0])
        self.assertEqual(state['current_step'], 2)
        state = engine.apply(state, 'next')
        self.assertEqual(state['current_step'], 3)
        state = engine.apply(state, 'detect')
        self.assertTrue(state['detection_loaded'])
        self.assertEqual(len(state['regions']), 2)

        # Sort -> Next through steps -> Save Annotation
        state = engine.apply(state, 'sort_boxes')
        state = engine.apply(state, 'next')  # step 4
        state = engine.apply(state, 'next')  # step 5
        state = engine.apply(state, 'next')  # step 6
        state = engine.apply(state, 'save')

        # Upload different JSON boxes for 12305.jpg; saved annotation must take priority on reopen
        new_upload = self.root / 'upload_new.json'
        new_upload.write_text(json.dumps({
            'image': '12305.jpg',
            'bounding_boxes': {
                '1': {'bbox': [100, 100, 140, 140], 'status': 'damaged'},
                '2': {'bbox': [100, 150, 140, 190], 'status': 'damaged'},
            },
        }), encoding='utf-8')
        store_detections(parse_detection_upload(new_upload, self.images).records, self.output_dir)

        reopened = engine.open_image(self.images[0])
        bboxes = [r['bbox'] for r in reopened['regions'].values()]
        self.assertEqual(bboxes, [[10, 10, 50, 50], [10, 60, 50, 100]])

        # Explicit detect overwrites with newly uploaded JSON boxes
        reopened = engine.apply(reopened, 'next')
        overwritten = engine.apply(reopened, 'detect')
        bboxes_after = [r['bbox'] for r in overwritten['regions'].values()]
        self.assertEqual(bboxes_after, [[100, 100, 140, 140], [100, 150, 140, 190]])

    def test_detect_boxes_raises_friendly_error_when_no_json_in_json_mode(self):
        opts = SimpleNamespace(
            output_dir=self.output_dir,
            skip_detection=False,
            detection_source='json',
        )
        with self.assertRaises(ValueError) as ctx:
            detect(self.images[1], opts)
        self.assertIn('No uploaded box JSON found', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
