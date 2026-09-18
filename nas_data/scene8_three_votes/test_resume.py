"""Regression coverage for the real tuple/list blocker and safe 0701108 reuse."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from common import (stage_lock, digest, dump, read, sha, current_code_hashes,
                    approved_legacy_resume, RecordStore, LABELS)
from merge import compatible_completed_signature
from prompts import PAIRS
from legacy_prompts import LEGACY_PROMPTS


class ResumeTests(unittest.TestCase):
    def meta(self):
        return {'pairs': PAIRS, 'legacy_prompts': LEGACY_PROMPTS,
                'selection': 'fixed-selection', 'checkpoint_sha256': 'fixed-weights',
                'margin': .02, 'packages': {'torch': '2.4.1+cu118'}}

    def legacy(self):
        return read(Path(__file__).with_name('resume_compatibility.json'))['legacy_code_hashes']

    def test_real_prompt_tuple_json_roundtrip_reuses_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, sig = stage_lock(root, 'mobile', self.meta())
            h = sha(directory/'metadata.json')
            directory2, sig2 = stage_lock(root, 'mobile', self.meta())
            self.assertEqual(directory, directory2)
            self.assertEqual(sig, sig2)
            self.assertEqual(h, sha(directory/'metadata.json'))
            self.assertEqual(sig, digest(read(directory/'metadata.json')))

    def test_old_sqlite_rows_and_metadata_not_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'mobile'; folder.mkdir()
            saved = dict(self.meta(), code_hashes=self.legacy())
            dump(folder/'metadata.json', saved)
            old_sig = digest(saved)
            store = RecordStore(folder/'results.sqlite', old_sig)
            item = {'id':'0000000', 'image_sha256':'image-hash'}
            store.put(item, {'labels':{k:-1 for k in LABELS}}); store.close()
            before = {n:sha(folder/n) for n in ('metadata.json','results.sqlite')}
            _, sig = stage_lock(root, 'mobile', self.meta())
            self.assertEqual(sig, old_sig)
            store = RecordStore(folder/'results.sqlite', sig, readonly=True)
            self.assertIsNotNone(store.get(item)); store.close()
            self.assertEqual(before, {n:sha(folder/n) for n in before})
            self.assertFalse(read(folder/'resume_compatibility_audit.json')['metadata_and_existing_rows_rewritten'])
            # New rows retain the persisted identity and coexist with unchanged old rows.
            store = RecordStore(folder/'results.sqlite', sig)
            store.put(dict(item,id='0000200'), {'labels':{k:0 for k in LABELS}})
            self.assertIsNotNone(store.get(item)); store.close()
            self.assertEqual(stage_lock(root,'mobile',self.meta())[1],old_sig)

    def test_changed_weights_prompts_selection_packages_still_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'mobile';folder.mkdir()
            dump(folder/'metadata.json',dict(self.meta(),code_hashes=self.legacy()))
            for key,new in [('checkpoint_sha256','other'),('selection','other'),('margin',.03),
                            ('pairs',{}),('packages',{'torch':'different'})]:
                with self.subTest(key=key):
                    meta=self.meta();meta[key]=new
                    with self.assertRaises(ValueError):stage_lock(root,'mobile',meta)

    def test_unapproved_local_source_change_still_blocks(self):
        code=current_code_hashes()
        self.assertTrue(approved_legacy_resume(self.legacy(),code))
        altered=dict(code, **{'qwen.py':'unapproved'})
        self.assertFalse(approved_legacy_resume(self.legacy(),altered))
        self.assertFalse(approved_legacy_resume(dict(self.legacy(), **{'mobile.py':'unapproved'}),code))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'mobile';folder.mkdir()
            dump(folder/'metadata.json',dict(self.meta(),code_hashes=self.legacy()))
            with patch('common.current_code_hashes',return_value=altered):
                with self.assertRaises(ValueError):stage_lock(root,'mobile',self.meta())

    def test_exact_legacy_preview_policy_reuse_not_other_policy(self):
        legacy=self.legacy();plan={'selection':'unchanged'}
        metadata=[dict(self.meta(),code_hashes=legacy)]*3
        completed={'signature':digest({'plan':plan,'teacher_metadata':metadata,
                   'merge_code':legacy['merge.py'],'common_code':legacy['common.py'],'preview_only':True})}
        self.assertTrue(compatible_completed_signature(completed,'new',plan,metadata,True))
        self.assertFalse(compatible_completed_signature(completed,'new',{'selection':'changed'},metadata,True))
        self.assertFalse(compatible_completed_signature(completed,'new',plan,metadata,False))
        self.assertFalse(compatible_completed_signature({'signature':'wrong'},'new',plan,metadata,True))


if __name__=='__main__':unittest.main()
