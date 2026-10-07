"""Signal-aware wrapper. No torch. The runner file is not modified."""
import json, os, subprocess, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'run-with-status.sh'
EXCLUDED = (
    'runner_atlas', '_torch_coo_from_canonical_csr', 'resolve_device',
    '--allow-accelerator', 'D1_WSCALE_COMPAT', 'wscale-shim',
)


def run(rc_or_cmd, env):
    if isinstance(rc_or_cmd, int):
        cmd = ['bash', str(SCRIPT), 'sh', '-c', 'exit %d' % rc_or_cmd]
    else:
        cmd = ['bash', str(SCRIPT), *rc_or_cmd]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


class RunStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = os.environ.copy()
        self.env['FLY_STATUS_DIR'] = self.tmp.name
        self.env['FLY_STATUS_FILE'] = os.path.join(self.tmp.name, 'run-status.json')

    def tearDown(self):
        self.tmp.cleanup()

    def status(self):
        with open(self.env['FLY_STATUS_FILE']) as f:
            return json.load(f)

    def test_exit_0_is_done(self):
        r = run(['true'], self.env)
        self.assertEqual(r.returncode, 0)
        st = self.status()
        self.assertEqual(st['status'], 'done')
        self.assertEqual(st['rc'], 0)
        self.assertNotIn('signal', st)
        self.assertIn('true', st['cmd'])
        self.assertFalse(os.path.exists(self.env['FLY_STATUS_FILE'] + '.tmp'))

    def test_exit_137_is_killed_oom(self):
        r = run(137, self.env)
        self.assertEqual(r.returncode, 137)
        st = self.status()
        self.assertEqual(st['status'], 'killed-oom')
        self.assertEqual(st['rc'], 137)
        self.assertEqual(st['signal'], 9)

    def test_exit_139_is_killed_segfault(self):
        r = run(139, self.env)
        self.assertEqual(r.returncode, 139)
        st = self.status()
        self.assertEqual(st['status'], 'killed-segfault')
        self.assertEqual(st['rc'], 139)
        self.assertEqual(st['signal'], 11)

    def test_exit_9_is_failed(self):
        r = run(9, self.env)
        self.assertEqual(r.returncode, 9)
        st = self.status()
        self.assertEqual(st['status'], 'failed')
        self.assertEqual(st['rc'], 9)
        self.assertNotIn('signal', st)

    def test_exit_11_is_failed(self):
        r = run(11, self.env)
        self.assertEqual(r.returncode, 11)
        st = self.status()
        self.assertEqual(st['status'], 'failed')
        self.assertEqual(st['rc'], 11)
        self.assertNotIn('signal', st)

    def test_exit_143_is_killed_signal(self):
        r = run(143, self.env)
        self.assertEqual(r.returncode, 143)
        st = self.status()
        self.assertEqual(st['status'], 'killed-signal')
        self.assertEqual(st['rc'], 143)
        self.assertEqual(st['signal'], 15)

    def test_exit_2_is_failed(self):
        r = run(2, self.env)
        self.assertEqual(r.returncode, 2)
        st = self.status()
        self.assertEqual(st['status'], 'failed')
        self.assertEqual(st['rc'], 2)
        self.assertNotIn('signal', st)

    def test_script_targets_upstream_runner_and_leaves_it_unchanged(self):
        text = SCRIPT.read_text()
        self.assertIn('src/runner.py', text)
        for bad in EXCLUDED:
            self.assertNotIn(bad, text)
        # Unchanged relative to whatever base this commit is applied on,
        # not a historical blob of runner.py.
        diff = subprocess.check_output(
            ['git', 'diff', 'HEAD^', '--', 'src/runner.py'], cwd=ROOT)
        self.assertEqual(diff, b'')


if __name__ == '__main__':
    unittest.main()
