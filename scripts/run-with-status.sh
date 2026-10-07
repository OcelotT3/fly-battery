#!/usr/bin/env bash
# Signal-aware launcher for src/runner.py.
# 0 → done; 137 (also 165) → killed-oom; 139 (also 155) → killed-segfault;
# any other 128+signal → killed-signal; otherwise → failed.
# A plain exit 9 or 11 was not killed by a signal. Bash reports a signal
# death as 128+N, and the runner already exits that way, so 9 and 11 stay
# failed. Adapted from the sealed harden overlay's run-with-status.sh to
# this repo's scripts/ + src/runner.py layout. This wrapper only names the
# code. No runner change.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
STATUS_DIR="${FLY_STATUS_DIR:-$ROOT/results}"
STATUS_FILE="${FLY_STATUS_FILE:-$STATUS_DIR/run-status.json}"
mkdir -p "$STATUS_DIR" "$(dirname "$STATUS_FILE")"

# A leading command that is not a flag and not a .py file runs as itself, so
# the map can be tested without torch. Anything else is the runner.
if [[ $# -gt 0 && "$1" != -* && "$1" != *.py && ! -f "$DIR/$1" ]]; then
  CMD=("$@")
else
  CMD=(python3 "$ROOT/src/runner.py" "$@")
fi

set +e
"${CMD[@]}"
rc=$?
set -e

status="failed"
case "$rc" in
  0) status="done" ;;
  137|165) status="killed-oom" ;;
  139|155) status="killed-segfault" ;;
esac
if [[ "$rc" -ge 128 && "$status" == failed ]]; then status="killed-signal"; fi

signal=""
if [[ "$rc" -ge 128 ]]; then signal=$((rc - 128)); fi

ts="$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')"
cmd_json="$(printf '%s\n' "${CMD[@]}" | python3 -c 'import json,sys; print(json.dumps([l.rstrip(chr(10)) for l in sys.stdin]))')"
sig_field=""
if [[ -n "$signal" ]]; then sig_field="$(printf ',"signal":%s' "$signal")"; fi
tmp="$STATUS_FILE.tmp"
printf '{"status":"%s","rc":%d,"ts":"%s","cmd":%s%s}\n' \
  "$status" "$rc" "$ts" "$cmd_json" "$sig_field" \
  > "$tmp"
mv -f "$tmp" "$STATUS_FILE"
echo "run-with-status: status=$status rc=$rc file=$STATUS_FILE"
exit "$rc"
