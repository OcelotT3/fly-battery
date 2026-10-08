"""Heartbeat helpers and the runner wiring. No torch."""
import ast, contextlib, io, json, os, py_compile, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'src'))
import harden


def runner_heartbeat_block():
    """The `if harden.heartbeat_due(...)` statement from runner.main, compiled on its own (runner imports torch)."""
    tree = ast.parse((ROOT / 'src' / 'runner.py').read_text())
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.If) and 'harden.heartbeat_due(' in ast.unparse(n.test)]
    assert len(hits) == 1, len(hits)
    after = ast.parse('reached_progress_print = True').body
    mod = ast.Module(body=[hits[0]] + after, type_ignores=[])
    return compile(ast.fix_missing_locations(mod), 'runner.py:heartbeat', 'exec')

HASH_LINE = "prev = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(); row['sha256'] = prev"
WRITE_LINE = "fo.write(json.dumps(row, sort_keys=True) + '\\n'); fo.flush(); rows += 1"
WSCALE = "wscale=float(np.exp(rng.uniform(np.log(0.05), np.log(1.5))) / R['W_syn_mV'])"
EXCLUDED = (
    '_torch_coo_from_canonical_csr', 'resolve_device', '--allow-accelerator',
    'FLY_VERIFICATION', 'D1_WSCALE_COMPAT', 'wscale-shim',
)


class HeartbeatTests(unittest.TestCase):
    def test_due_gate(self):
        self.assertFalse(harden.heartbeat_due(0, 1))
        self.assertFalse(harden.heartbeat_due(None, 1))
        self.assertFalse(harden.heartbeat_due(-1, 1))
        self.assertTrue(harden.heartbeat_due(1, 1))
        self.assertFalse(harden.heartbeat_due(2, 1))
        self.assertTrue(harden.heartbeat_due(2, 2))

    def test_payload_and_atomic_replace(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'nested', 'status-heartbeat.json')
            payload = harden.write_heartbeat(
                path, 3, [1, 'real', 0, 'a', None, None], 1.23456, ts='2026-10-06T00:00:00')
            self.assertFalse(os.path.exists(path + '.tmp'))
            data = json.loads(Path(path).read_text())
            self.assertEqual(data['rows'], 3)
            self.assertEqual(data['last_key'], [1, 'real', 0, 'a', None, None])
            self.assertEqual(data['rss_gb'], round(1.23456, 4))
            self.assertEqual(data['ts'], '2026-10-06T00:00:00')
            self.assertEqual(payload, data)
            text = Path(path).read_text()
            self.assertTrue(text.endswith('\n'))
            self.assertLess(text.index('"last_key"'), text.index('"rows"'))  # sort_keys

    def test_replace_failure_keeps_prior_heartbeat(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'status-heartbeat.json')
            harden.write_heartbeat(path, 1, ['a'], 0.1, ts='t1')
            real = os.replace

            def boom(src, dst):
                raise OSError('disk')

            os.replace = boom
            try:
                with self.assertRaises(OSError):
                    harden.write_heartbeat(path, 2, ['b'], 0.2, ts='t2')
            finally:
                os.replace = real
            kept = json.loads(Path(path).read_text())
            self.assertEqual(kept['rows'], 1)
            self.assertEqual(kept['ts'], 't1')

    def test_runner_call_site_survives_heartbeat_oserror(self):
        code = runner_heartbeat_block()
        with tempfile.TemporaryDirectory() as td:
            hb_path = os.path.join(td, 'status-heartbeat.json')
            harden.write_heartbeat(hb_path, 1, ['a'], 0.1, ts='t1')
            ns = {
                'harden': harden, 'hb_path': hb_path, 'rows': 2,
                'a': type('A', (), {'heartbeat_every': 1})(),
                'row': {'item': 1, 'condition': 'real', 'trial': 1, 'stimulus': 'a', 'step': None, 'jitter': None},
            }
            real = os.replace

            def boom(src, dst):
                raise OSError('disk full')

            out = io.StringIO()
            os.replace = boom
            try:
                with contextlib.redirect_stdout(out):
                    exec(code, ns)
            finally:
                os.replace = real
            self.assertTrue(ns.get('reached_progress_print'))
            lines = out.getvalue().splitlines()
            self.assertEqual(len(lines), 1, lines)
            self.assertIn(hb_path, lines[0])
            self.assertIn('disk full', lines[0])
            self.assertEqual(json.loads(Path(hb_path).read_text())['rows'], 1)

    def test_rss_gb_is_a_nonnegative_float(self):
        rss = harden.rss_gb()
        self.assertIsInstance(rss, float)
        self.assertGreaterEqual(rss, 0.0)

    def test_runner_wires_heartbeat_and_keeps_upstream_row_bytes(self):
        src = (ROOT / 'src' / 'runner.py').read_text()
        self.assertIn('import harden\n', src)
        self.assertIn("'--heartbeat-every', type=int, default=0", src)
        self.assertIn('harden.heartbeat_due(a.heartbeat_every, rows)', src)
        self.assertIn('harden.write_heartbeat(', src)
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
