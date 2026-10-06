"""Pure RSS helpers for the runner. No torch and no connectome data."""
import sys


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


def guard_rss(max_rss_gb, where):
    """Log RSS after `where` when a cap is set. Abort cleanly when RSS exceeds it.

    No-op when max_rss_gb is None, so a run that did not pass --max-rss-gb
    prints nothing and never exits here.
    """
    if max_rss_gb is None:
        return None
    r = rss_gb()
    print('rss_gb %.3f after %s' % (r, where), flush=True)
    if r > float(max_rss_gb):
        sys.exit('abort: RSS %.3f GB exceeds --max-rss-gb %.3f (%s)' % (
            r, float(max_rss_gb), where))
    return r
