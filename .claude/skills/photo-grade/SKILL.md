---
name: photo-grade
description: Professional colour grading of the user's photos from the CLI. Read the photo, set an intent, then grade with looks, AI masks, reference matching and skin-safe retouch. Use when asked to grade, edit, colour or restyle photos.
---

# photo-grade

Grade like a colourist, not a filter. Every photo gets a decision about what it should feel like and where the eye goes, and masked moves that serve that decision. The engine is `pgrade.py` from https://github.com/0x6Ason/photo-grade.

## Setup (once per session)

```bash
git clone -q --depth 1 https://github.com/0x6Ason/photo-grade ~/photo-grade 2>/dev/null || git -C ~/photo-grade pull -q
pip install -q -r ~/photo-grade/requirements.txt   # add --break-system-packages if pip refuses
PG="python3 $HOME/photo-grade/pgrade.py"
```
Work on copies of uploads and write results to an output folder. If the repo is private, attach it with the add_repo tool first.

## Workflow (never use `--look auto`)

1. **Read the photo.** Look at it yourself. `$PG analyze PHOTO` gives the facts, and `$PG masks PHOTO --subject` shows what the AI masks find. Note these:
   - the subject, and where the eye should land
   - the light: direction, quality, time of day
   - the mood it already has
   - distractions
   - skin and sky
2. **Set the intent in one line:** the feeling, where the eye goes, and what to calm. For example: "summer race energy; eye to the diver; calm the busy stands". If the user named a style or feeling, it wins. Ask only if the photo could plausibly go two very different ways and they gave you nothing.
3. **Choose a base:** a look from the table, a learned or matched reference look (see below), or `natural`.
4. **Write a recipe:** a small JSON file holding the base plus 2–5 local moves. Each move must serve the intent. Typical moves:
   - light the subject
   - calm or darken edges and busy areas
   - control the sky
   - separate colours, for example a warm subject against a cool background
   ```json
   {"base": "jp-summer", "intent": "summer race energy; eye to the diver; calm the stands",
    "local": [
      {"mask": "subject", "adjust": {"exposure": 0.12, "clarity": 12}},
      {"mask": "grad:top:0.55", "adjust": {"exposure": -0.2, "highlights": -25, "saturation": -12}},
      {"mask": "radial:0.45,0.62,0.45", "adjust": {"exposure": 0.08}}],
    "vignette": {"amount": -12}}
   ```
5. **Render:** `$PG grade PHOTO --look recipe.json --compare --show-masks -o graded/`. For faces, add `--retouch 0.5`–`1`.
6. **Critique, then iterate one to three times.**
   - **Check `*_masks.jpg` first:** each mask must hit exactly what you meant. If not, fix the tokens or positions.
   - **Then check the image:**
     - Does the eye land where intended?
     - Are highlights or shadows clipped?
     - Is skin natural?
     - Do the colours sit together?
     - Is there a halo at any mask edge?
     - Is it overcooked? If so, lower the values or use `--strength 0.7`–`0.85`.
7. **Deliver** the full-size files plus the compare images. Add one line covering the intent and what you did.

For a set from one shoot, write one series recipe and make small per-photo tweaks so the photos match.

## Masks

Tokens combine with `*` (intersection), and `!` inverts a token.
- **AI masks:** `sky`, `subject`, `background`, `person`, `face`, `skin`, `faceskin` (face and neck only; use for retouching)
- **Shapes and ranges:**
  - `lum:0.6-1`: a brightness range
  - `hue:185-235`: an Oklab hue range. Cyan is about 195, blue about 255, red about 25, orange about 58, green about 140.
  - `grad:top|bottom|left|right:EXTENT`: a gradient
  - `radial:CX,CY,SIZE`: an ellipse, positioned as fractions of the frame
  - `all`

Example: `"!sky*lum:0-0.3"` selects the shadows outside the sky.

A move's `adjust` accepts any look key: `exposure`, `contrast`, `highlights`, `shadows`, `whites`, `blacks`, `temperature`, `tint`, `vibrance`, `saturation`, `hsl`, `clarity`, `texture`, `dehaze`, `lut`.

Look-level keys also include:
- `curve`, `curve_r`, `curve_g`, `curve_b`
- `color_grade: {shadows|midtones|highlights|global: [hue°, sat, lum]}`
- `bloom`, `vignette`, `grain`, `sharpness`
- `protect_skin`, `bw` with `bw_mix`

## Bases

| look | for |
|---|---|
| `natural` | clean, true-to-life polish; the default when nothing stylised fits |
| `jp-airy` | portraits and selfies in light, airy settings (透明感) |
| `jp-film` | people in the street, everyday life; soft Fuji pastels |
| `jp-summer` | sunny days, sea, trains, blue skies (learned from the user's own edit; strong) |
| `film-warm` | portraits, indoor, warm nostalgic |
| `classic-chrome` | street, documentary, muted travel |
| `teal-orange` | cinematic daylight city and travel |
| `golden` | warm light, backlit people, beaches |
| `gta6` | sunsets, dusk streets, warm neon |
| `vice-night` | blue hour and night skylines (gta6 only makes a blue sky bluer) |
| `neon-noir` | night streets, neon, rain |
| `moody` | overcast, rain, architecture |
| `bw` | harsh light, overcast, character |

## Style from a reference

The final image is always the user's own pixels. A reference only supplies colour; never deliver a generated image.
- **The user edited a photo themselves:** run `$PG learn ORIGINAL EDIT --name N`. This is the most accurate option.
- **You have an image-generation tool:** generate a restyle of the user's own photo in the wanted style, keeping the same framing. Then run `$PG learn PHOTO RESTYLE --name N --loose`. Check `N_learn_check.jpg`, then grade the real photo with look `N`, which can be the base of a recipe.
- **A different scene** (film still, a photo the user likes): run `$PG match PHOTO REF --name N`. This works when the light and content are similar. Check `N_match_check.jpg`, and discard the result if it just tints the photo.

## Pro rules

- Fix exposure and white balance before style, and enhance the light that's there rather than fighting it.
- Strong styles (gta6, vice-night, jp-summer, neon-noir) need matching light. On the wrong photo they look like a filter.
- Skin must never turn orange, grey or whitened. The engine guards skin tone, but still check. Retouch faces only.
- Keep detail in highlights and shadows unless the look is deliberately faded or crushed.

## Other commands

- `$PG sheet PHOTO --looks a,b,c`: a labelled contact sheet, for when the user wants to choose.
- `$PG lut --look N out.cube`, then `ffmpeg -i in.mp4 -vf lut3d=out.cube ...`: applies the look to video (global colour only).

## Limits

- A 12 MP photo takes about 10–25 s and up to about 2.8 GB of RAM.
- Masks follow real edges at full resolution, but very thin structures and blonde hair next to faces can be partly included.
- `hue:` masks ignore near-neutral pixels, so slightly tinted whites are not selected.
