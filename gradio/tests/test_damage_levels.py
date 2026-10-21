"""Damage classification and real Apply Changes callback regression tests."""
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from app import create_app, parser
from annotation.state import new_state, set_verified_content, initialize_alignment
from annotation.bbox import add_bbox
from annotation.status import update_status, confirm_status, replace_statuses
from annotation.io import atomic_write, load_annotation


class DamageLevels(unittest.TestCase):
    def state(self):
        state = new_state()
        state.update(image='1.png', image_size=[100, 100], current_step=4, code='1', draft_content={'noi_dung': [{'ky_hieu': '1', 'chuyen_muc': [{'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '永寺'}]}]})
        set_verified_content(state, {}, '永寺')
        for x in (10, 40):
            add_bbox(state, [x, 10, x + 20, 30])
        initialize_alignment(state)
        return state

    def test_levels_validation_and_transitions(self):
        s = self.state()
        uid = s['region_uid_by_box_id']['1']
        update_status(s, uid, 'damaged')
        with self.assertRaisesRegex(ValueError, 'boxes: 1'):
            confirm_status(s)
        for level in (1, 2, 3):
            update_status(s, uid, 'damaged', damage_level=level)
            confirm_status(s)
            self.assertEqual(s['bounding_boxes']['1']['damage_level'], level)
        for level in (True, False, 0, 4, '2', 2.0):
            before = deepcopy(s)
            with self.assertRaises(ValueError):
                replace_statuses(s, {u: b['status'] for u, b in s['regions'].items()},
                                 damage_levels={u: level for u in s['regions']})
            self.assertEqual(s, before)
        update_status(s, uid, 'intact')
        update_status(s, uid, 'damaged', expert_prediction=True)
        self.assertIsNone(s['regions'][uid]['damage_level'])
        initialize_alignment(s, list(reversed(s['regions'])))
        self.assertIsNone(s['regions'][uid]['damage_level'])

    def test_apply_commits_draft_and_order_atomically_without_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / '1.png'
            Image.new('RGB', (100, 100)).save(image)
            source = root / 'source.json'
            atomic_write(source, [{'noi_dung': [{'ky_hieu': '1', 'chuyen_muc': [
                {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '永寺'}]}]}])
            app = create_app(parser().parse_args([
                '--image-dir', str(root), '--source-json', str(source),
                '--output-dir', str(root / 'out'), '--skip-detection']))
            apply = next(f.fn for f in app.fns.values()
                         if f.fn and f.fn.__name__ == 'apply_status_changes')
            state = self.state()
            state.update(image_path=str(image), image_url='image.jpg', draft_content={'noi_dung': [{'ky_hieu': '1', 'chuyen_muc': [{'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '永寺'}]}]})
            ctx = dict(active=state, drafts={})
            payload = {'statuses': {'1': 'damaged', '2': 'intact'},
                       'damage_levels': {'1': None, '2': None},
                       'unknowns': {'1': False, '2': False},
                       'textSequence': ['寺', '永']}
            result = apply(ctx, json.dumps(payload), 'intact')
            self.assertEqual(result[0], ctx)
            self.assertEqual(result[8], {'__type__': 'update'})
            self.assertFalse((root / 'out' / '1.json').exists())
            self.assertEqual(ctx['active']['bounding_boxes']['1']['status'], 'intact')
            payload['damage_levels']['1'] = 2
            result = apply(ctx, json.dumps(payload), 'intact')
            committed = result[0]['active']
            self.assertEqual(committed['current_step'], 4)
            self.assertEqual(committed['bounding_boxes']['1']['status'], 'damaged')
            self.assertEqual(committed['annotations'], {'1': '寺', '2': '永'})
            self.assertEqual(committed['bounding_boxes']['1']['damage_level'], 2)
            self.assertFalse((root / 'out' / '1.json').exists())
            before = deepcopy(result[0])
            payload['textSequence'] = ['永']
            failed = apply(result[0], json.dumps(payload), 'intact')
            self.assertEqual(failed[0], before)
            self.assertEqual(failed[8], {'__type__': 'update'})

    def test_legacy_json_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            from crop.crop import image_resize, crop_document
            s = self.state()
            boxes = deepcopy(s['bounding_boxes'])
            boxes['1']['status'] = 'damaged'
            for box in boxes.values():
                box.pop('damage_level', None)
            doc = dict(image='1.png', bounding_boxes=boxes, annotations=s['annotations'],
                       image_resize=image_resize([100, 100], [100, 100]), crop=crop_document('1.png', [0, 0, 100, 100], [100, 100])['crop'])
            path = Path(folder) / '1.json'
            atomic_write(path, doc)
            original = path.read_bytes()
            self.assertEqual(load_annotation(path, '1.png', [100, 100]), doc)
            self.assertEqual(path.read_bytes(), original)

    def test_apply_missing_token_moves_mark_and_preserves_rejected_draft(self):
        import xml.etree.ElementTree as ET
        from ui.editor import snapshot

        def marked_boxes(state):
            markup = snapshot(state)['markup']
            start = markup.index('<svg class="annotation-canvas"')
            svg = ET.fromstring(markup[start:markup.index('</svg>', start) + 6])
            for node in svg.iter():
                node.tag = node.tag.rsplit('}', 1)[-1]
            return [group.attrib['data-box-id'] for group in svg.iter('g')
                    if 'data-box-id' in group.attrib
                    and group.find("g[@data-miss-mark='1']") is not None]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / '1.png'
            Image.new('RGB', (100, 100)).save(image)
            source = root / 'source.json'
            atomic_write(source, [{'noi_dung': [{'ky_hieu': '1', 'chuyen_muc': [
                {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '永寺'}]}]}])
            app = create_app(parser().parse_args([
                '--image-dir', str(root), '--source-json', str(source),
                '--output-dir', str(root / 'out'), '--skip-detection']))
            apply = next(f.fn for f in app.fns.values()
                         if f.fn and f.fn.__name__ == 'apply_status_changes')
            state = self.state()
            add_bbox(state, [70, 10, 90, 30])
            state['source_mismatch'] = dict(
                source_text='永寺', source_character_count=2,
                bounding_box_count=3, issue_type='missing_text', note='')
            initialize_alignment(state, list(state['regions']))
            state.update(image_path=str(image), image_url='image.jpg',
                         draft_content=json.loads(source.read_text())[0])
            ctx = dict(active=state, drafts={})
            self.assertEqual(marked_boxes(state), ['3'])
            for sequence, tokens, box_id in (
                    (['MISS', '永', '寺'], ['3', '1', '2'], '1'),
                    (['永', 'MISS', '寺'], ['1', '3', '2'], '2')):
                result = apply(ctx, json.dumps(dict(textSequence=sequence, tokenOrder=tokens)), 'intact')
                ctx = result[0]
                self.assertEqual(list(ctx['active']['annotations'].values()), sequence)
                self.assertEqual(marked_boxes(ctx['active']), [box_id])
            before = deepcopy(ctx)
            result = apply(ctx, json.dumps(dict(
                textSequence=['永', '寺', 'MISS'], tokenOrder=['1', '2', '3'],
                statuses={'1': 'damaged'}, damage_levels={'1': None})), 'intact')
            self.assertEqual(result[0], before)
            # gr.skip() leaves the browser draft and its live X untouched.
            self.assertEqual(result[8], {'__type__': 'update'})
            self.assertEqual(marked_boxes(result[0]['active']), ['2'])
            self.assertFalse((root / 'out' / '1.json').exists())
