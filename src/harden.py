"""Pure resume helpers for the runner. No torch and no connectome data."""
import json, os
from collections import namedtuple

Resume = namedtuple('Resume', 'chain_prev trial_start completed_keys logs')


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
        _commit_truncation(path, prefix)
        return 'resume: truncated torn last line (%d bytes)' % len(last)
    if not data.endswith(b'\n'):
        _commit_truncation(path, data + b'\n')
        return 'resume: appended missing newline after the last complete row'
    return None


def load_resume_state(path):
    """Parse a JSONL file into (prev_sha, completed_keys, suggested_trial_start).

    prev_sha is the last valid row's sha256, or None when that row has none.
    suggested_trial_start is max(trial)+1, or 0 when no integer trial was seen.
    Missing, empty, and non-object lines contribute nothing. A non-last line
    that is not JSON is skipped; the caller truncates a torn last line first.
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
            prev_sha = sha if isinstance(sha, str) else None
    if not seen:
        return None, completed, 0
    suggested = (max_trial + 1) if max_trial >= 0 else 0
    return prev_sha, completed, suggested


def apply_resume(path, chain_prev, trial_start):
    """Truncate a torn tail, then wire chain prev and trial start when still unset.

    An explicit chain_prev or trial_start is left alone. completed_keys is
    returned either way, so a pinned --trial-start 0 still skips duplicates.
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
    if chain_prev is None and prev_sha:
        chain_prev = prev_sha
    if trial_start is None:
        trial_start = suggested
    return Resume(chain_prev, trial_start, keys, logs)
