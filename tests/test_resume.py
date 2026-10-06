"""Resume helpers and the runner wiring. No torch."""
import hashlib, json, os, py_compile, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'src'))
import harden

HASH_LINE = "prev = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(); row['sha256'] = prev"
WRITE_LINE = "fo.write(json.dumps(row, sort_keys=True) + '\\n'); fo.flush(); rows += 1"
WSCALE = "wscale=float(np.exp(rng.uniform(np.log(0.05), np.log(1.5))) / R['W_syn_mV'])"
EXCLUDED = (
    '_torch_coo_from_canonical_csr', 'resolve_device', '--allow-accelerator',
    'FLY_VERIFICATION', 'D1_WSCALE_COMPAT', 'wscale-shim',
)


def runner_text():
    return (ROOT / 'src' / 'runner.py').read_text()


def row(**kw):
    base = {
        'item': 1, 'condition': 'real', 'trial': 0, 'stimulus': 'a',
        'step': None, 'jitter': None, 'sha256': 'aaa',
    }
    base.update(kw)
    return base


class ResumeTests(unittest.TestCase):
    def test_missing_and_empty(self):
        r = harden.apply_resume('/nonexistent/out.jsonl', None, None)
        self.assertIsNone(r.chain_prev)
        self.assertEqual(r.trial_start, 0)
        self.assertEqual(r.completed_keys, set())
        self.assertTrue(any('missing/empty' in line for line in r.logs))
        with tempfile.TemporaryDirectory() as td:
            empty = os.path.join(td, 'out.jsonl')
            open(empty, 'w').close()
            r = harden.apply_resume(empty, None, None)
            self.assertIsNone(r.chain_prev)
            self.assertEqual(r.trial_start, 0)
            self.assertEqual(r.completed_keys, set())
            self.assertTrue(any('missing/empty' in line for line in r.logs))

    def test_last_sha_keys_and_suggested_trial(self):
        rows = [
            row(trial=0, stimulus='a', sha256='aaa'),
            row(trial=1, stimulus='a', sha256='bbb'),
            row(trial=1, stimulus='b', step=0.5, jitter=2.0, sha256='ccc'),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.jsonl')
            with open(path, 'w') as f:
                f.write('not-json\n')
                for item in rows:
                    f.write(json.dumps(item) + '\n')
                f.write('[]\n')
            r = harden.apply_resume(path, None, None)
            self.assertEqual(r.chain_prev, 'ccc')
            self.assertEqual(r.trial_start, 2)
            self.assertEqual(len(r.completed_keys), 3)
            self.assertIn(harden.row_key(1, 'real', 0, 'a', None, None), r.completed_keys)
            self.assertIn(harden.row_key(1, 'real', 1, 'b', 0.5, 2.0), r.completed_keys)
            # the stored sha is wired as-is; it is not recomputed from the row body
            self.assertNotEqual(r.chain_prev, hashlib.sha256(
                json.dumps({k: v for k, v in rows[-1].items() if k != 'sha256'}, sort_keys=True).encode()
            ).hexdigest())

    def test_explicit_chain_must_match_last_sha_and_trial_pin_stays(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.jsonl')
            with open(path, 'w') as f:
                f.write(json.dumps(row(trial=4, sha256='ddd')) + '\n')
            r = harden.apply_resume(path, 'ddd', 0)
            self.assertEqual(r.chain_prev, 'ddd')
            self.assertEqual(r.trial_start, 0)
            self.assertIn(harden.row_key(1, 'real', 4, 'a', None, None), r.completed_keys)
            with self.assertRaises(harden.ResumeRefused) as caught:
                harden.apply_resume(path, 'pinned', 0)
            msg = str(caught.exception)
            self.assertIn('pinned', msg)
            self.assertIn('ddd', msg)
            self.assertEqual(Path(path).read_text(), json.dumps(row(trial=4, sha256='ddd')) + '\n')

    def test_chain_prev_without_resume_is_refused(self):
        with self.assertRaises(harden.ResumeRefused) as caught:
            harden.reject_bare_chain_prev('anything', False)
        self.assertIn('--chain-prev is valid only with --resume', str(caught.exception))
        harden.reject_bare_chain_prev(None, False)
        harden.reject_bare_chain_prev('anything', True)

    def test_last_row_without_sha_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.jsonl')
            bare = row(trial=1)
            del bare['sha256']
            with open(path, 'w') as f:
                f.write(json.dumps(row(trial=0, sha256='aaa')) + '\n')
                f.write(json.dumps(bare) + '\n')
            with self.assertRaises(harden.ResumeRefused) as caught:
                harden.apply_resume(path, None, None)
            self.assertIn('no sha256', str(caught.exception))
            numbered = row(trial=2, sha256=123)
            with open(path, 'w') as f:
                f.write(json.dumps(numbered) + '\n')
            with self.assertRaises(harden.ResumeRefused) as caught:
                harden.apply_resume(path, 'aaa', None)
            self.assertIn('no sha256', str(caught.exception))
            blank = row(trial=3, sha256='')
            with open(path, 'w') as f:
                f.write(json.dumps(blank) + '\n')
            with self.assertRaises(harden.ResumeRefused) as caught:
                harden.apply_resume(path, None, None)
            self.assertIn('no sha256', str(caught.exception))

    def test_earlier_row_without_sha_is_ok_when_last_has_one(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.jsonl')
            bare = row(trial=0)
            del bare['sha256']
            with open(path, 'w') as f:
                f.write(json.dumps(bare) + '\n')
                f.write(json.dumps(row(trial=1, sha256='bbb')) + '\n')
            r = harden.apply_resume(path, None, None)
            self.assertEqual(r.chain_prev, 'bbb')
            self.assertEqual(r.trial_start, 2)

    def test_torn_last_line_is_truncated(self):
        good = json.dumps(row(trial=3, sha256='aaa')) + '\n'
        torn = '{"item": 1, "sha256": "bbb", "trial":'
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.jsonl')
            with open(path, 'wb') as f:
                f.write(good.encode() + torn.encode())
            original = good.encode() + torn.encode()
            r = harden.apply_resume(path, None, None)
            data = Path(path).read_bytes()
            self.assertNotIn(torn.encode(), data)
            self.assertEqual(data, good.encode())
            self.assertEqual(r.chain_prev, 'aaa')
            self.assertEqual(r.trial_start, 4)
            sides = [n for n in os.listdir(td) if n.startswith('out.jsonl.torn-')]
            self.assertEqual(len(sides), 1)
            side_path = os.path.join(td, sides[0])
            self.assertEqual(Path(side_path).read_bytes(), torn.encode())
            self.assertEqual(data + Path(side_path).read_bytes(), original)
            self.assertTrue(any('truncated torn last line' in line and side_path in line for line in r.logs))

    def test_torn_line_with_newline_is_truncated(self):
        good = json.dumps(row(sha256='aaa')) + '\n'
        torn = '{"item":1,"trial":\n'
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.jsonl')
            with open(path, 'wb') as f:
                f.write(good.encode() + torn.encode())
            original = good.encode() + torn.encode()
            r = harden.apply_resume(path, None, None)
            data = Path(path).read_bytes()
            self.assertEqual(data, good.encode())
            self.assertEqual(r.chain_prev, 'aaa')
            sides = [n for n in os.listdir(td) if n.startswith('out.jsonl.torn-')]
            self.assertEqual(len(sides), 1)
            side_path = os.path.join(td, sides[0])
            self.assertEqual(data + Path(side_path).read_bytes(), original)
            self.assertTrue(any('truncated torn' in line and side_path in line for line in r.logs))

    def test_middle_garbage_is_not_truncated(self):
        lines = 'not-json\n' + json.dumps(row(sha256='zzz')) + '\n'
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.jsonl')
            with open(path, 'w') as f:
                f.write(lines)
            before = Path(path).read_bytes()
            r = harden.apply_resume(path, None, None)
            self.assertEqual(Path(path).read_bytes(), before)
            self.assertEqual(r.chain_prev, 'zzz')
            self.assertFalse(any('truncated' in line for line in r.logs))

    def test_valid_json_without_newline_gets_newline(self):
        payload = json.dumps(row(trial=4, sha256='abc'))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.jsonl')
            with open(path, 'wb') as f:
                f.write(payload.encode())
            r = harden.apply_resume(path, None, None)
            self.assertTrue(Path(path).read_bytes().endswith(b'\n'))
            self.assertEqual(r.chain_prev, 'abc')
            self.assertEqual(r.trial_start, 5)
            self.assertTrue(any('newline' in line for line in r.logs))
            self.assertFalse(any(n.startswith('out.jsonl.torn-') for n in os.listdir(td)))

    def test_chain_recipe_excludes_sha256_and_uses_default_separators(self):
        battery = b'{"seal":"example"}\n'
        prev = hashlib.sha256(battery).hexdigest()
        body = {
            'condition': 'real', 'item': 1, 'jitter': None, 'prev': prev,
            'step': None, 'stimulus': 'a', 'trial': 0,
        }
        encoded = json.dumps(body, sort_keys=True)
        self.assertEqual(encoded, json.dumps(body, sort_keys=True, separators=(', ', ': ')))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with_sha = dict(body)
        with_sha['sha256'] = digest
        self.assertNotEqual(
            digest,
            hashlib.sha256(json.dumps(with_sha, sort_keys=True).encode()).hexdigest())
        # the next row's prev is that digest
        nxt = dict(body)
        nxt['prev'] = digest
        nxt['trial'] = 1
        self.assertEqual(nxt['prev'], digest)

    def test_runner_wires_resume_and_keeps_upstream_row_bytes(self):
        src = runner_text()
        self.assertIn('import harden\n', src)
        self.assertIn('harden.apply_resume', src)
        self.assertIn('harden.reject_bare_chain_prev', src)
        self.assertIn('except harden.ResumeRefused as err:', src)
        self.assertIn('if rk in completed_keys:', src)
        self.assertIn('skip duplicate', src)
        self.assertIn("if a.trial_start is None:\n        a.trial_start = 0\n", src)
        self.assertIn(
            "prev = a.chain_prev or hashlib.sha256(open(BATTERY_FILE, 'rb').read()).hexdigest()",
            src)
        self.assertIn(HASH_LINE, src)
        self.assertIn(WRITE_LINE, src)
        self.assertIn(WSCALE, src)
        self.assertNotIn("np.exp(rng.uniform(np.log(0.05), np.log(1.5)) / R['W_syn_mV'])", src)
        self.assertNotIn('separators=', src)
        for bad in EXCLUDED:
            self.assertNotIn(bad, src)
        py_compile.compile(str(ROOT / 'src' / 'runner.py'), doraise=True)
        py_compile.compile(str(ROOT / 'src' / 'harden.py'), doraise=True)


if __name__ == '__main__':
    unittest.main()
