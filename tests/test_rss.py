"""RSS gate helpers and the runner wiring. No torch."""
import io, contextlib, py_compile, unittest
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


class RssTests(unittest.TestCase):
    def setUp(self):
        self._rss = harden.rss_gb
        harden.rss_gb = lambda: 3.5

    def tearDown(self):
        harden.rss_gb = self._rss

    def test_unset_cap_is_silent(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = harden.guard_rss(None, 'mid-run')
        self.assertIsNone(got)
        self.assertEqual(buf.getvalue(), '')

    def test_under_cap_logs_and_returns(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = harden.guard_rss(10.0, 'build_weights(real)')
        self.assertEqual(got, 3.5)
        self.assertEqual(buf.getvalue(), 'rss_gb 3.500 after build_weights(real)\n')

    def test_abort_when_over_cap(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                harden.guard_rss(1.0, 'mid-run')
        self.assertEqual(caught.exception.code,
                         'abort: RSS 3.500 GB exceeds --max-rss-gb 1.000 (mid-run)')
        self.assertIn('rss_gb 3.500 after mid-run', buf.getvalue())

    def test_equal_to_cap_does_not_abort(self):
        harden.rss_gb = lambda: 1.0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = harden.guard_rss(1.0, 'mid-run')
        self.assertEqual(got, 1.0)

    def test_rss_gb_is_a_nonnegative_float(self):
        harden.rss_gb = self._rss
        rss = harden.rss_gb()
        self.assertIsInstance(rss, float)
        self.assertGreaterEqual(rss, 0.0)

    def test_runner_wires_gate_and_keeps_upstream_row_bytes(self):
        src = (ROOT / 'src' / 'runner.py').read_text()
        self.assertIn('import harden\n', src)
        self.assertIn("'--max-rss-gb', type=float, default=None", src)
        for where in ('build_weights(real)', 'build_weights(shuffled)', 'sign_permuted_weights', 'mid-run'):
            self.assertIn("harden.guard_rss(a.max_rss_gb, '%s')" % where, src)
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
