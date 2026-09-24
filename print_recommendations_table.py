"""
One-off: prints every recommendation cell for review, and writes the
same content to recommendations_table.md. Reads directly from
recommendations.py so the doc can never drift from the real data.
"""
import recommendations as rec

SHAPES = ["Oval", "Round", "Square", "Heart", "Oblong"]
TONES  = ["Very Light", "Light", "Medium", "Tan", "Brown", "Deep"]

lines = []


def out(text=""):
    print(text)
    lines.append(text)


out("# Recommendations Table")
out()
out("Generated from recommendations.py — regenerate with "
    "`python print_recommendations_table.py` after editing that file.")
out()

# ── Hairstyle ──
out("## Hairstyle (3 options per face shape, per gender)")
for gender, table in [("Male", rec.HAIR_REC_MALE), ("Female", rec.HAIR_REC_FEMALE)]:
    out()
    out(f"### {gender}")
    for shape in SHAPES:
        items = table.get(shape, [])
        status = "OK" if len(items) >= 3 else f"ONLY {len(items)}"
        out(f"**{shape}** ({status})")
        for item in items:
            out(f"- {item['name']} — {item['why']}")
        out()

# ── Grooming ──
out("## Grooming / Beard Tips (2+ per face shape, per gender)")
for gender, table in [("Male (beard)", rec.GROOMING_REC_MALE), ("Female (grooming)", rec.GROOMING_REC_FEMALE)]:
    out()
    out(f"### {gender}")
    for shape in SHAPES:
        tips = table.get(shape, [])
        status = "OK" if len(tips) >= 2 else f"ONLY {len(tips)}"
        out(f"**{shape}** ({status})")
        for tip in tips:
            out(f"- {tip}")
        out()

# ── Glasses ──
out("## Glasses (2-3 frame types per face shape, plus which PNG gets the REC badge)")
out()
for shape in SHAPES:
    items = rec.GLASSES_REC.get(shape, [])
    status = "OK" if len(items) >= 2 else f"ONLY {len(items)}"
    rec_indices = rec.FACE_GLASSES.get(shape, [])
    out(f"**{shape}** ({status}) — REC badge on PNG index(es): {rec_indices}")
    for item in items:
        out(f"- {item['name']} — {item['why']}")
    out()

# ── Outfit colors ──
out("## Outfit Colors (4-5 wear + 2-3 avoid per skin tone)")
for gender, table in [("Male", rec.COLOR_REC_MALE), ("Female", rec.COLOR_REC_FEMALE)]:
    out()
    out(f"### {gender} — wear colors")
    for tone in TONES:
        colors = [c.strip() for c in table.get(tone, "").split(",") if c.strip()]
        status = "OK" if len(colors) >= 4 else f"ONLY {len(colors)}"
        out(f"- **{tone}** ({status}): {', '.join(colors)}")
out()
out("### Avoid colors (shared across genders, per skin tone)")
for tone in TONES:
    colors = [c.strip() for c in rec.AVOID_COLOR_REC.get(tone, "").split(",") if c.strip()]
    status = "OK" if len(colors) >= 2 else f"ONLY {len(colors)}"
    out(f"- **{tone}** ({status}): {', '.join(colors)}")

with open('recommendations_table.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print("\n\n=== Saved to recommendations_table.md ===")
