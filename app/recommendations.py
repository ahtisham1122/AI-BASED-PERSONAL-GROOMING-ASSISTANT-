"""Style recommendation tables (hairstyle, grooming, glasses, outfit colors) and lookups."""

import feedback

# Each face shape maps to 3 suggestions with a one-line reason, so the
# UI can show "why" under each option instead of just a bare label.
HAIR_REC_MALE = {
    "Oval": [
        {"name": "Textured crop",  "why": "Keeps proportions balanced, works with any hairline"},
        {"name": "Quiff",          "why": "Adds height on top without overwhelming your face"},
        {"name": "Side part",      "why": "Clean and classic, flatters even proportions"},
    ],
    "Round": [
        {"name": "High fade",          "why": "Removes width at the sides, elongates the face"},
        {"name": "Pompadour",          "why": "Vertical volume counters the roundness"},
        {"name": "Angular fringe",     "why": "Sharp lines break up soft, curved features"},
    ],
    "Square": [
        {"name": "Crew cut",              "why": "Complements a strong jaw without competing with it"},
        {"name": "Textured pompadour",    "why": "Height softens the squareness at the temples"},
        {"name": "Side-swept fringe",     "why": "Softens a strong forehead line"},
    ],
    "Heart": [
        {"name": "Side part",           "why": "Balances a wider forehead with a narrower chin"},
        {"name": "Short textured crop", "why": "Keeps volume low, away from the forehead"},
        {"name": "Fringe / bangs",      "why": "Shortens the forehead visually"},
    ],
    "Oblong": [
        {"name": "Fringe / bangs",        "why": "Shortens the apparent length of the face"},
        {"name": "Crew cut with texture", "why": "Short sides avoid stretching the face further"},
        {"name": "Side part, low volume", "why": "Avoids extra height that lengthens the face"},
    ],
}

HAIR_REC_FEMALE = {
    "Oval": [
        {"name": "Curtain bangs",     "why": "Frames the face without hiding its natural balance"},
        {"name": "Long layers",       "why": "Adds movement while keeping the oval shape visible"},
        {"name": "Sleek middle part", "why": "Classic and low-maintenance for balanced features"},
    ],
    "Round": [
        {"name": "Long layers",     "why": "Elongates the face with vertical lines"},
        {"name": "Side-swept style","why": "Breaks up the roundness of the cheeks"},
        {"name": "High ponytail",   "why": "Draws the eye upward, away from width"},
    ],
    "Square": [
        {"name": "Soft waves",           "why": "Softens a strong, angular jawline"},
        {"name": "Side part with layers","why": "Reduces visual width at the jaw"},
        {"name": "Long bob (lob)",       "why": "Rounds out sharp angles at chin level"},
    ],
    "Heart": [
        {"name": "Chin-length bob",   "why": "Balances a narrower chin with width at the jaw"},
        {"name": "Side part",         "why": "Shifts volume away from a wider forehead"},
        {"name": "Soft curtain bangs","why": "Softens the forehead without adding width"},
    ],
    "Oblong": [
        {"name": "Voluminous curls", "why": "Adds width to balance a longer face"},
        {"name": "Curtain bangs",    "why": "Shortens the visual length of the face"},
        {"name": "Blunt bob",        "why": "A horizontal line breaks up the length"},
    ],
}

# Beard shaping tips per face shape — at least 2 each.
GROOMING_REC_MALE = {
    "Oval": [
        "Most beard styles work — keep edges naturally rounded",
        "Light stubble adds definition without upsetting your balance",
    ],
    "Round": [
        "Angular beard, longer at the chin, to add length",
        "Keep cheek lines high and sharp to avoid adding width",
    ],
    "Square": [
        "Rounded beard edges soften a strong jawline",
        "A little extra length at the corners eases sharp angles",
    ],
    "Heart": [
        "Fuller beard at the chin, trimmed close at the cheeks",
        "Avoid heavy cheek volume — it widens an already broad forehead",
    ],
    "Oblong": [
        "Fuller at the sides, shorter at the chin, to add width",
        "Keep overall length short so the face doesn't look longer",
    ],
}

# Grooming/framing tips (brows, contour) per face shape — at least 2 each.
GROOMING_REC_FEMALE = {
    "Oval": [
        "Soft, rounded brows keep your naturally balanced proportions",
        "Face-framing layers work with almost any styling",
    ],
    "Round": [
        "Angled, structured brows add definition to soft features",
        "Contour along the cheekbones to add subtle length",
    ],
    "Square": [
        "Softly rounded brows ease a strong jawline",
        "Contour the corners of the jaw to soften angles",
    ],
    "Heart": [
        "Fuller, rounded brows balance a narrower chin",
        "Light contour at the jaw adds width where it's needed",
    ],
    "Oblong": [
        "Straighter, horizontal brows visually shorten the face",
        "Blush swept horizontally across cheeks adds width",
    ],
}

# 2-3 suitable frame TYPES per face shape, each with a one-line reason.
# Independent of which glasses PNGs we actually have assets for — see
# FACE_GLASSES below for which of the 4 loaded PNGs gets the REC badge.
GLASSES_REC = {
    "Oval": [
        {"name": "Rectangle frames", "why": "Adds structure to soft, balanced proportions"},
        {"name": "Round frames",     "why": "Complements natural symmetry without harsh lines"},
        {"name": "Aviator",          "why": "Follows the face's natural balance for a classic look"},
    ],
    "Round": [
        {"name": "Rectangle frames", "why": "Adds angles and length to counter roundness"},
        {"name": "Square frames",    "why": "Sharp edges bring definition soft features lack"},
        {"name": "Browline frames",  "why": "A strong top line adds structure above the cheeks"},
    ],
    "Square": [
        {"name": "Round frames",       "why": "Softens a strong, angular jawline"},
        {"name": "Oval frames",        "why": "Rounded edges balance a squared-off face"},
        {"name": "Thin/rimless frames","why": "Reduces visual weight so the jaw doesn't dominate"},
    ],
    "Heart": [
        {"name": "Aviator",          "why": "A bottom-heavy shape balances a wider forehead"},
        {"name": "Round frames",     "why": "Softens a narrower, more pointed chin"},
        {"name": "Rimless frames",   "why": "Keeps the focus off a wider forehead"},
    ],
    "Oblong": [
        {"name": "Round frames",      "why": "Adds width and softens a longer face"},
        {"name": "Square frames",     "why": "A wide shape breaks up the face's length"},
        {"name": "Oversized frames",  "why": "Wide frames add horizontal balance"},
    ],
}

COLOR_REC_MALE = {
    "Very Light" : "Navy, Forest Green, Burgundy, Charcoal, Slate Blue",
    "Light"      : "Teal, Olive, Warm Brown, Denim Blue, Burgundy",
    "Medium"     : "Jewel tones, Mustard, White, Olive, Rust",
    "Tan"        : "White, Royal Blue, Charcoal, Emerald, Terracotta",
    "Brown"      : "White, Sky Blue, Emerald, Mustard, Coral",
    "Deep"       : "White, Red, Gold, Royal Blue, Mint Green",
}

COLOR_REC_FEMALE = {
    "Very Light" : "Dusty Rose, Burgundy, Navy, Lavender, Charcoal",
    "Light"      : "Coral, Teal, Warm Brown, Blush Pink, Denim Blue",
    "Medium"     : "Jewel tones, Mustard, Terracotta, Coral, White",
    "Tan"        : "White, Hot Pink, Orange, Coral, Turquoise",
    "Brown"      : "Fuchsia, Yellow, Sky Blue, Mint Green, White",
    "Deep"       : "White, Electric Blue, Gold, Fuchsia, Sand",
}

# Colors that tend to clash with each skin tone, shown as a dimmer row.
# Checked against both gender wear lists above so nothing contradicts.
AVOID_COLOR_REC = {
    "Very Light" : "Pale Yellow, Beige, Ivory",
    "Light"      : "Beige, Pale Pastels, Cream",
    "Medium"     : "Muted Brown, Beige, Pale Pastels",
    "Tan"        : "Warm Brown, Rust, Olive",
    "Brown"      : "Charcoal, Dark Brown, Rust",
    "Deep"       : "Black, Navy, Dark Brown",
}

# Approximate RGB swatch for every color name used above. "Jewel tones"
# and "Pale Pastels" are vibes rather than single colors, so these pick
# one representative shade for the swatch.
COLOR_SWATCHES = {
    "Navy"          : (26,  35,  74),
    "Forest Green"  : (34,  85,  51),
    "Burgundy"      : (128, 25,  45),
    "Teal"          : (0,   128, 128),
    "Olive"         : (110, 117, 45),
    "Warm Brown"    : (139, 90,  43),
    "Jewel tones"   : (106, 13,  173),
    "Mustard"       : (204, 164, 26),
    "White"         : (245, 245, 245),
    "Royal Blue"    : (65,  105, 225),
    "Charcoal"      : (54,  58,  63),
    "Sky Blue"      : (135, 206, 235),
    "Emerald"       : (0,   168, 107),
    "Red"           : (196, 30,  30),
    "Gold"          : (212, 175, 55),
    "Dusty Rose"    : (196, 143, 151),
    "Coral"         : (255, 127, 80),
    "Terracotta"    : (204, 78,  52),
    "Hot Pink"      : (255, 105, 180),
    "Orange"        : (230, 126, 34),
    "Fuchsia"       : (204, 0,   153),
    "Yellow"        : (240, 210, 30),
    "Electric Blue" : (60,  110, 240),
    "Pale Yellow"   : (245, 230, 160),
    "Beige"         : (222, 202, 168),
    "Pale Pastels"  : (230, 200, 220),
    "Muted Brown"   : (120, 95,  70),
    "Rust"          : (150, 70,  35),
    "Dark Brown"    : (60,  40,  25),
    "Black"         : (25,  25,  25),
    "Slate Blue"    : (106, 122, 158),
    "Lavender"      : (181, 166, 214),
    "Denim Blue"    : (69,  108, 153),
    "Blush Pink"    : (222, 178, 180),
    "Cream"         : (240, 232, 210),
    "Turquoise"     : (48,  181, 176),
    "Sand"          : (210, 184, 145),
    "Mint Green"    : (152, 216, 190),
    "Ivory"         : (238, 233, 215),
}

# Which loaded glasses PNG (by index) gets the REC badge for each face
# shape — the single best match among the 4 actual assets we have.
FACE_GLASSES = {
    "Oval"   : [0, 1, 2, 3],
    "Round"  : [0, 2],
    "Square" : [1, 3],
    "Heart"  : [0, 3],
    "Oblong" : [1, 2],
}


# A suggestion needs at least this many ratings before a low average
# is trusted enough to demote it — one bad rating shouldn't bury an
# option, but a consistent pattern should.
DEMOTE_MIN_RATINGS = 3
DEMOTE_BELOW_AVG   = 2.5


def _rerank_by_feedback(base_items, stats):
    """
    Reorders base_items (list of {"name", "why", ...}) using per-name
    (avg, count) feedback stats: rated items sort to the front by
    average rating (highest first); items with no ratings keep their
    original relative order (Python's sort is stable, so giving them
    all the same sort key preserves that); items with
    DEMOTE_MIN_RATINGS+ ratings averaging below DEMOTE_BELOW_AVG drop
    to the bottom, marked 'less_popular'. Never raises — a bad/missing
    stats dict just means no reordering happens.
    """
    try:
        def sort_key(pair):
            _, item = pair
            avg, count = stats.get(item['name'], (None, 0))
            if avg is not None and count >= DEMOTE_MIN_RATINGS and avg < DEMOTE_BELOW_AVG:
                return (2, -avg)
            elif avg is not None and count >= 1:
                return (0, -avg)
            else:
                return (1, 0)

        ranked = sorted(enumerate(base_items), key=sort_key)
        result = []
        for _, item in ranked:
            avg, count = stats.get(item['name'], (None, 0))
            new_item = dict(item)
            new_item['less_popular'] = bool(
                avg is not None and count >= DEMOTE_MIN_RATINGS and avg < DEMOTE_BELOW_AVG
            )
            result.append(new_item)
        return result
    except Exception as e:
        print(f"[recommendations] re-ranking failed, using default order: {e}")
        return list(base_items)


def get_hair_rec(gender, face_shape):
    """Returns the list of {"name", "why", "less_popular"} suggestions for this gender/face shape, re-ranked by feedback (empty list if unknown)."""
    table = HAIR_REC_MALE if gender == 'male' else HAIR_REC_FEMALE
    base = table.get(face_shape, [])
    if not base:
        return []
    stats = feedback.get_hair_rating_stats(face_shape, gender)
    return _rerank_by_feedback(base, stats)


def get_hair_rec_summary(gender, face_shape):
    """One-line summary (just the top suggestion's name), for simple/fallback displays."""
    recs = get_hair_rec(gender, face_shape)
    return recs[0]['name'] if recs else "---"


def get_grooming_rec(gender, face_shape):
    """Returns the list of grooming tip strings for this gender/face shape (empty list if unknown)."""
    table = GROOMING_REC_MALE if gender == 'male' else GROOMING_REC_FEMALE
    return table.get(face_shape, [])


def get_glasses_rec(face_shape):
    """Returns the list of {"name", "why", "less_popular"} suitable frame types for this face shape, re-ranked by feedback."""
    base = GLASSES_REC.get(face_shape, [])
    if not base:
        return []
    stats = feedback.get_glasses_rating_stats(face_shape)
    return _rerank_by_feedback(base, stats)


def get_glasses_rec_summary(face_shape):
    """One-line summary (just the top frame type's name), for simple/fallback displays."""
    recs = get_glasses_rec(face_shape)
    return recs[0]['name'] if recs else "---"


def get_color_rec(gender, skin_tone):
    table = COLOR_REC_MALE if gender == 'male' else COLOR_REC_FEMALE
    return table.get(skin_tone, "---")


def get_avoid_colors(skin_tone):
    return AVOID_COLOR_REC.get(skin_tone, "---")


def get_color_swatch(color_name):
    """RGB tuple for a color name, or mid-gray if not in the table."""
    return COLOR_SWATCHES.get(color_name, (128, 128, 128))


def get_suggested_glasses(face_shape, num_glasses):
    return FACE_GLASSES.get(face_shape, list(range(num_glasses)))
