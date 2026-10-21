import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation.detection_import import has_detection, parse_detection_upload, store_detections
from annotation.export import (
    collect_annotations,
    collect_content_documents,
    collect_source_mismatches,
    save_export_archive,
)
from annotation.progress import build_progress_archive, restore_progress_archive
from annotation.workflow import Workflow


class ProgressArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.image_dir = self.root / 'images'
        self.output_dir = self.root / 'output'
        self.image_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        self.images = []
        for stem in ('12305', '12306', '12307'):
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
                    {
                        'ky_hieu': '12307',
                        'chuyen_muc': [
                            {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '明'}
                        ],
                    },
                ]
            }
        ], ensure_ascii=False), encoding='utf-8')

        self.opts = SimpleNamespace(
            image_dir=self.image_dir,
            source_json=self.source_json,
            output_dir=self.output_dir,
            content_titles=('Nguyên văn chữ Hán Nôm',),
            annotation_title='Nguyên văn chữ Hán Nôm',
            skip_detection=True,
            detection_source='json',
        )
        self.engine = Workflow(self.opts)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _populate_sample_progress(self):
        # 1. Upload detection boxes for 12305 and 12307
        upload_file = self.root / 'detections.json'
        upload_file.write_text(json.dumps([
            {
                'image': '12305.jpg',
                'bounding_boxes': {
                    '1': {'bbox': [10, 10, 50, 50], 'status': 'intact'},
                    '2': {'bbox': [10, 60, 50, 100], 'status': 'damaged'},
                },
            },
            {
                'image': '12307.jpg',
                'bounding_boxes': {
                    '1': {'bbox': [20, 20, 60, 60], 'status': 'intact'},
                },
            },
        ]), encoding='utf-8')
        store_detections(
            parse_detection_upload(upload_file, self.images).records,
            self.output_dir,
        )

        # 2. Annotate and save 12305 as a normal annotation
        s1 = self.engine.open_image(self.images[0])
        s1 = self.engine.apply(s1, 'next')  # Step 3
        s1 = self.engine.apply(s1, 'detect')
        s1 = self.engine.apply(s1, 'sort_boxes')
        s1 = self.engine.apply(s1, 'next')  # Step 4
        s1 = self.engine.apply(s1, 'statuses', {
            'statuses': {uid: box['status'] for uid, box in s1['regions'].items()},
            'damage_levels': {
                uid: (2 if box['status'] == 'damaged' else None)
                for uid, box in s1['regions'].items()
            },
        })
        s1 = self.engine.apply(s1, 'next')  # Step 5
        s1 = self.engine.apply(s1, 'next')  # Step 6
        s1 = self.engine.apply(s1, 'save')

        # 3. Annotate and save 12306 as a source mismatch
        s2 = self.engine.open_image(self.images[1])
        s2 = self.engine.apply(s2, 'next')  # Step 3
        s2 = self.engine.apply(s2, 'add', {'bbox': [10, 10, 40, 40]})
        s2 = self.engine.apply(s2, 'add', {'bbox': [10, 50, 40, 80]})
        s2 = self.engine.apply(s2, 'confirm_source_mismatch', {
            'issue_type': 'missing_text',
            'note': 'Two characters on stone',
        })
        s2 = self.engine.apply(s2, 'sort_boxes')
        s2 = self.engine.apply(s2, 'next')  # Step 4
        s2 = self.engine.apply(s2, 'next')  # Step 5
        s2 = self.engine.apply(s2, 'next')  # Step 6
        s2 = self.engine.apply(s2, 'save')

    def test_progress_round_trip(self):
        self._populate_sample_progress()
        before_ann = collect_annotations(self.images, self.output_dir, allow_empty=True)
        before_mm = collect_source_mismatches(self.images, self.output_dir, allow_empty=True)
        before_content = collect_content_documents(
            [self.images[0], self.images[1]], self.output_dir, allow_empty=True
        )

        archive_src = build_progress_archive(self.output_dir, self.images)
        saved_zip = self.root / 'saved_progress.zip'
        shutil.copy2(archive_src, saved_zip)

        # Wipe output_dir completely
        shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        summary = restore_progress_archive(saved_zip, self.output_dir, self.images)
        self.assertEqual(summary['mode'], 'progress')
        self.assertEqual(summary['restored_annotations'], 2)
        self.assertEqual(summary['restored_detections'], 2)
        self.assertTrue(has_detection('12307.jpg', self.output_dir))

        after_ann = collect_annotations(self.images, self.output_dir, allow_empty=True)
        after_mm = collect_source_mismatches(self.images, self.output_dir, allow_empty=True)
        after_content = collect_content_documents(
            [self.images[0], self.images[1]], self.output_dir, allow_empty=True
        )
        self.assertEqual(before_ann, after_ann)
        self.assertEqual(before_mm, after_mm)
        self.assertEqual(before_content, after_content)

        reopened = self.engine.open_image(self.images[0])
        self.assertEqual(reopened['annotations'], {'1': '永', '2': '樂'})

    def test_restore_from_export_archive_and_backup_creation(self):
        self._populate_sample_progress()
        annotations = collect_annotations(self.images, self.output_dir, allow_empty=True)
        content = collect_content_documents(
            [self.images[0], self.images[1]], self.output_dir, allow_empty=True
        )
        mismatches = collect_source_mismatches(self.images, self.output_dir, allow_empty=True)
        export_zip = save_export_archive(annotations, content, self.output_dir, mismatches)
        copied_export_zip = self.root / 'annotations.zip'
        shutil.copy2(export_zip, copied_export_zip)

        # Restore over existing output_dir -> must create backup in .backup/
        summary = restore_progress_archive(copied_export_zip, self.output_dir, self.images)
        self.assertEqual(summary['mode'], 'export')
        self.assertIsNotNone(summary['backup_dir'])
        self.assertTrue(Path(summary['backup_dir']).exists())

        after_ann = collect_annotations(self.images, self.output_dir, allow_empty=True)
        after_mm = collect_source_mismatches(self.images, self.output_dir, allow_empty=True)
        self.assertEqual(annotations, after_ann)
        self.assertEqual(mismatches, after_mm)

    def test_reject_zip_slip_path_traversal(self):
        evil_zip = self.root / 'evil.zip'
        with zipfile.ZipFile(evil_zip, 'w') as zf:
            zf.writestr('../evil.json', json.dumps({'image': '12305.jpg'}))
        with self.assertRaises(ValueError):
            restore_progress_archive(evil_zip, self.output_dir, self.images)

    def test_create_app_initializes_cleanly(self):
        from app import create_app, parser
        args = parser().parse_args([
            '--image-dir', str(self.image_dir),
            '--source-json', str(self.source_json),
            '--output-dir', str(self.output_dir),
            '--skip-detection',
        ])
        self.assertEqual(args.detection_source, 'json')
        blocks = create_app(args)
        self.assertIsNotNone(blocks)

    def test_app_save_and_load_progress_callbacks_replace_current_session(self):
        import base64
        from app import create_app, parser
        self._populate_sample_progress()
        args = parser().parse_args([
            '--image-dir', str(self.image_dir),
            '--source-json', str(self.source_json),
            '--output-dir', str(self.output_dir),
            '--skip-detection',
        ])
        blocks = create_app(args)
        fns = {fn.fn.__name__: fn.fn for fn in blocks.fns.values() if getattr(fn, 'fn', None)}
        save_progress_fn = fns['save_progress']
        load_progress_fn = fns['load_progress']

        # Open 12307 in Step 3 with 1 box ordered=1 (mid-progress draft, not yet saved at Step 7)
        s_draft = self.engine.open_image(self.images[2])
        s_draft = self.engine.apply(s_draft, 'next')
        s_draft = self.engine.apply(s_draft, 'detect')
        s_draft = self.engine.apply(s_draft, 'sort_boxes')
        self.assertEqual(s_draft['current_step'], 3)
        ctx_at_save = {
            'active': s_draft,
            'drafts': {s_draft['image_path']: s_draft},
        }
        selection_snapshot = json.dumps({
            'boxes': s_draft['regions'],
            'orders': {uid: box.get('order') for uid, box in s_draft['regions'].items()},
        })
        saved_ctx, payload_json = save_progress_fn(
            ctx_at_save, None, None, '', selection_snapshot, 'intact', None
        )
        payload = json.loads(payload_json)
        zip_path = self.root / payload['name']
        zip_path.write_bytes(base64.b64decode(payload['content']))

        # Now simulate the user continuing to modify the session in memory (e.g., advancing to Step 4)
        s_later = self.engine.apply(saved_ctx['active'], 'next')
        self.assertEqual(s_later['current_step'], 4)
        s_later['revision'] = 99
        current_ctx = {
            'active': s_later,
            'drafts': {s_later['image_path']: s_later},
        }

        # Load the older progress zip -> must revert active session back to Step 3 and increment revision
        rendered = load_progress_fn(current_ctx, str(zip_path))
        restored_ctx = rendered[0]
        board_update = rendered[8]
        self.assertIn('drafts', restored_ctx)
        self.assertEqual(restored_ctx['active']['image'], '12307.jpg')
        self.assertEqual(restored_ctx['active']['current_step'], 3)
        self.assertEqual(restored_ctx['active']['revision'], 100)
        self.assertTrue(board_update['value']['resetLocalState'])


if __name__ == '__main__':
    unittest.main()
