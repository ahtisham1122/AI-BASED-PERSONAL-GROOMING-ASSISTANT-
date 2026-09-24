"""
Lightweight ratings feedback store: lets the recommender learn from
1-5 star ratings the user gives to hairstyle/glasses suggestions,
persisted to feedback.json. Every public function catches its own
exceptions and returns a safe default (empty stats / 0 / no-op) so a
disk error or corrupt file can never crash the camera loop that calls
into this module.

Off by default: nothing in here does anything (no file, no reordering
effect, no "Learned from" line) until add_rating() is called for the
first time.
"""
import json
import os
from datetime import datetime, timezone
from collections import defaultdict

FEEDBACK_PATH = 'feedback.json'

# In-memory cache keyed on the file's mtime, so recommendations.py can
# call the lookup functions every frame without re-reading/parsing the
# JSON file 30 times a second — only re-read when it actually changes.
_cache = {'mtime': None, 'records': []}


def _load():
    try:
        if not os.path.exists(FEEDBACK_PATH):
            _cache['mtime'] = None
            _cache['records'] = []
            return _cache['records']
        mtime = os.path.getmtime(FEEDBACK_PATH)
        if mtime != _cache['mtime']:
            with open(FEEDBACK_PATH, encoding='utf-8') as f:
                _cache['records'] = json.load(f)
            _cache['mtime'] = mtime
        return _cache['records']
    except Exception as e:
        print(f"[feedback] Couldn't read {FEEDBACK_PATH} ({e}); treating as empty.")
        return []


def _save(records):
    try:
        with open(FEEDBACK_PATH, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        _cache['records'] = records
        _cache['mtime'] = os.path.getmtime(FEEDBACK_PATH)
    except Exception as e:
        print(f"[feedback] Couldn't save {FEEDBACK_PATH}: {e}")


def add_rating(face_shape, gender, skin_tone, category, suggestion, rating):
    """category is 'hairstyle' or 'glasses'. rating is an int 1-5."""
    try:
        records = list(_load())
        records.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'face_shape': face_shape,
            'gender': gender,
            'skin_tone': skin_tone,
            'category': category,
            'suggestion': suggestion,
            'rating': int(rating),
        })
        _save(records)
    except Exception as e:
        print(f"[feedback] add_rating failed: {e}")


def reset():
    """Clears all stored feedback (Shift+R in the app) — for a clean demo."""
    try:
        _save([])
        print("[feedback] Cleared feedback.json")
    except Exception as e:
        print(f"[feedback] reset failed: {e}")


def _stats_for(records):
    """records -> {suggestion_text: (average_rating, count)}"""
    sums   = defaultdict(float)
    counts = defaultdict(int)
    for r in records:
        try:
            sums[r['suggestion']]   += r['rating']
            counts[r['suggestion']] += 1
        except (KeyError, TypeError):
            continue  # skip a malformed record rather than fail the whole lookup
    return {name: (sums[name] / counts[name], counts[name]) for name in counts}


def get_hair_rating_stats(face_shape, gender):
    """{suggestion_name: (avg, count)} for hairstyle ratings matching this face shape + gender."""
    try:
        records = [r for r in _load()
                   if r.get('category') == 'hairstyle'
                   and r.get('face_shape') == face_shape
                   and r.get('gender') == gender]
        return _stats_for(records)
    except Exception as e:
        print(f"[feedback] get_hair_rating_stats failed: {e}")
        return {}


def get_glasses_rating_stats(face_shape):
    """{suggestion_name: (avg, count)} for glasses ratings matching this face shape (glasses recs aren't gender-split)."""
    try:
        records = [r for r in _load()
                   if r.get('category') == 'glasses'
                   and r.get('face_shape') == face_shape]
        return _stats_for(records)
    except Exception as e:
        print(f"[feedback] get_glasses_rating_stats failed: {e}")
        return {}


def count_ratings_for(face_shape, gender):
    """Total ratings (both categories) tied to this face shape + gender, for the 'Learned from N ratings' line."""
    try:
        return sum(1 for r in _load()
                   if r.get('face_shape') == face_shape and r.get('gender') == gender)
    except Exception as e:
        print(f"[feedback] count_ratings_for failed: {e}")
        return 0
