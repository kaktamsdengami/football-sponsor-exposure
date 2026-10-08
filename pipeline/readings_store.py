"""
Append-only store for brand readings, so reading a whole match survives anything.

THE PROBLEM THIS SOLVES. Reading a 113-minute match by eye is ~540 montages.
No single context holds that: a session compacts, a window is closed, a
subagent finishes, a model is swapped mid-way. Any of those loses whatever was
only ever held in a conversation.

So nothing is held in a conversation. Each batch of readings is appended to
disk the moment it is made, keyed by group id, and the next reader asks the
store what is still missing. That makes the job resumable by construction and
parallel for free -- several readers can work different slices at once, because
appends do not conflict and the reducer is last-wins per group.

An UNREAD group and a group read as EMPTY are different facts and are stored
differently. "Nobody was legible here" is a measurement; "nobody has looked
here yet" is a gap. Collapsing them would silently turn unread footage into
zero exposure, which is the failure this whole project exists to prevent.

    from pipeline.readings_store import append, load, pending
    append(path, [{"gid": 12, "brands": ["BRAND_A"]}], by="reader-1")
    done = load(path)                       # {gid: [brands]}
    todo = pending(index_path, path)        # montages still needing a reader
"""
import json
import os
from datetime import datetime


def append(path, entries, by="unknown"):
    """
    Add readings. `entries` are {"gid": int, "brands": [str], "note": str?}.

    Append-only and flushed per call: a reader that dies mid-match loses only
    the batch it was holding, never the ones before it.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for e in entries:
            if "gid" not in e:
                raise ValueError(f"reading has no gid: {e!r}")
            rec = {"gid": int(e["gid"]),
                   "brands": list(e.get("brands") or []),
                   "by": by, "at": stamp}
            if e.get("note"):
                rec["note"] = e["note"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    return n


def load(path):
    """{gid -> [brands]}, last write wins. Missing file is an empty store."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue            # a half-written last line from a kill
            out[int(r["gid"])] = list(r.get("brands") or [])
    return out


def load_meta(path):
    """{gid -> record} including who read it and when, for auditing a match."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(r["gid"])] = r
    return out


def _index_gids(index_path):
    """{montage file -> [gid]} from the export's index.json."""
    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)
    return {name: [g["gid"] for g in groups]
            for name, groups in idx["montages"].items()}


def pending(index_path, readings_path):
    """
    [(montage, [unread gids])] in file order.

    A montage is listed while ANY of its groups is unread, and the caller is
    told which -- a reader that was interrupted half way through an image
    should not have to redo the panels it already recorded.
    """
    done = set(load(readings_path))
    out = []
    for name, gids in sorted(_index_gids(index_path).items()):
        missing = [g for g in gids if g not in done]
        if missing:
            out.append((name, missing))
    return out


def progress(index_path, readings_path):
    """Counts for a one-line status: (read groups, total groups, montages left)."""
    idx = _index_gids(index_path)
    total = sum(len(v) for v in idx.values())
    done = len(set(load(readings_path)) & {g for v in idx.values() for g in v})
    return done, total, len(pending(index_path, readings_path))


def slice_for(index_path, readings_path, worker, n_workers):
    """
    The share of the outstanding montages this worker should read.

    Sliced round-robin rather than in blocks so that if some workers stop
    early, the ones still going have covered the match evenly rather than
    leaving the last third untouched.
    """
    todo = pending(index_path, readings_path)
    return [t for i, t in enumerate(todo) if i % n_workers == worker]
