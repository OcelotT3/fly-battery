"""Pure resume and heartbeat helpers for the runner. No torch and no connectome data."""
import json, os, time
from collections import namedtuple

Resume = namedtuple('Resume', 'chain_prev trial_start completed_keys logs')


class ResumeRefused(Exception):
    """The run must stop. `logs` are the resume notes already produced."""

    def __init__(self, message, logs=None):
        super().__init__(message)
        self.logs = list(logs or [])


def reject_bare_chain_prev(chain_prev, resume):
    """`--chain-prev` is a check against `--out`, not a chain root of its own.

    Without `--resume` an explicit value would start the file from an arbitrary
    digest. That is refused here; the runner calls this before the battery-hash
    fallback.
    """
    if chain_prev is not None and not resume:
        raise ResumeRefused('--chain-prev is valid only with --resume')


def row_key(item, condition, trial, stimulus, step=None, jitter=None):
    """Identity of one JSONL row. Matches the fields the runner writes."""
    return (item, condition, trial, stimulus, step, jitter)


def _last_line(data):
    """Return (prefix, last_line). prefix is kept on truncation and ends at a newline, or is empty.

    A file that ends in a newline still has a last line: the segment before that newline.
    An empty segment means the file already ends on a line break.
    """
    body = data[:-1] if data.endswith(b'\n') else data
    nl = body.rfind(b'\n')
    if nl == -1:
        return b'', body
    return body[:nl + 1], body[nl + 1:]


def _commit_truncation(path, data):
    """Replace `path` with `data` (the file minus a torn tail, or plus a newline)."""
    with open(path, 'wb') as f:
        f.write(data)


def truncate_torn_tail(path):
    """Drop a trailing JSONL line that is not valid JSON.

    Returns a log string when the file changed, else None. Only the last line
    is eligible; a bad line earlier in the file is left in place. A last line
    that is valid JSON but has no trailing newline gets a newline, so the next
    append does not glue two objects onto one line.

    Before the in-place rewrite, the removed tail is written beside the file
    as `<path>.torn-<unix time>` and that path is included in the log line.
    """
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        data = f.read()
    if not data:
        return None
    prefix, last = _last_line(data)
    if last.strip() == b'':
        return None
    try:
        json.loads(last.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        removed = data[len(prefix):]
        side = '%s.torn-%d' % (path, int(time.time()))
        with open(side, 'wb') as f:
            f.write(removed)
        _commit_truncation(path, prefix)
        return 'resume: truncated torn last line (%d bytes); wrote %s' % (len(removed), side)
    if not data.endswith(b'\n'):
        _commit_truncation(path, data + b'\n')
        return 'resume: appended missing newline after the last complete row'
    return None


def load_resume_state(path):
    """Parse a JSONL file into (prev_sha, completed_keys, suggested_trial_start).

    prev_sha is the last valid row's sha256, or None when that row has none
    (missing, empty, or not a string). suggested_trial_start is max(trial)+1,
    or 0 when no integer trial was seen. Missing, empty, and non-object lines
    contribute nothing. A non-last line that is not JSON is skipped; the caller
    truncates a torn last line first. When rows were seen and prev_sha is None,
    the caller refuses rather than restarting the chain from the battery hash.
    """
    completed = set()
    prev_sha = None
    max_trial = -1
    seen = False
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None, completed, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            seen = True
            completed.add(row_key(
                row.get('item'), row.get('condition'), row.get('trial'),
                row.get('stimulus'), row.get('step'), row.get('jitter')))
            trial = row.get('trial')
            if isinstance(trial, int) and not isinstance(trial, bool):
                max_trial = max(max_trial, trial)
            sha = row.get('sha256')
            prev_sha = sha if isinstance(sha, str) and sha else None
    if not seen:
        return None, completed, 0
    suggested = (max_trial + 1) if max_trial >= 0 else 0
    return prev_sha, completed, suggested


def apply_resume(path, chain_prev, trial_start):
    """Truncate a torn tail, then wire chain prev and trial start when still unset.

    An explicit trial_start is left alone, so a pinned --trial-start 0 still
    skips duplicates. An explicit chain_prev must equal the last row's sha256
    when that digest is present; a mismatch refuses and the message carries
    both values. When any valid row exists and the last one has no sha256,
    this refuses so the runner cannot fall back to the battery-file hash.
    """
    logs = []
    note = truncate_torn_tail(path)
    if note:
        logs.append(note)
    prev_sha, keys, suggested = load_resume_state(path)
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        logs.append('resume: --out missing/empty; proceeding as fresh run')
    else:
        shown = (prev_sha[:12] + '...') if prev_sha else None
        logs.append('resume: %d completed keys, suggested_trial_start=%d, last_sha=%s' % (
            len(keys), suggested, shown))
    if keys and not prev_sha:
        raise ResumeRefused(
            'resume: refusing; rows exist but the last valid row has no sha256', logs)
    if chain_prev is not None and prev_sha is not None and chain_prev != prev_sha:
        raise ResumeRefused(
            'resume: refusing; --chain-prev %s != last sha256 %s' % (chain_prev, prev_sha),
            logs)
    if chain_prev is None and prev_sha:
        chain_prev = prev_sha
    if trial_start is None:
        trial_start = suggested
    return Resume(chain_prev, trial_start, keys, logs)


def rss_gb():
    """Current RSS in GB. VmRSS from /proc (Linux, kB) when present, else peak ru_maxrss."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return 0.0


def heartbeat_due(every, rows):
    """True when this completed row should publish a heartbeat. 0 and None never do."""
    if not every or every < 0:
        return False
    return rows % every == 0


def write_heartbeat(path, rows, last_key, rss, ts=None):
    """Atomically write status-heartbeat.json (temp file, then os.replace)."""
    if ts is None:
        ts = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())
    payload = {
        'rows': int(rows),
        'last_key': list(last_key) if last_key is not None else None,
        'rss_gb': round(float(rss), 4),
        'ts': ts,
    }
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)
    return payload
