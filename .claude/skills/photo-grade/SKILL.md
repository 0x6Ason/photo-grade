---
name: photo-grade
description: Professional colour grading of the user's photos from the CLI. Read the photo, set an intent, then grade with looks, AI masks, reference matching and skin-safe retouch. Use when asked to grade, edit, colour or restyle photos.
---

# photo-grade

Grade like a colourist, not a filter. Every photo gets a decision about what it should feel like and where the eye goes, and masked moves that serve that decision. The engine is `pgrade.py` from https://github.com/0x6Ason/photo-grade (Python, numpy, OpenCV, ONNX; no Lightroom). It grades full-resolution float data. Its AI masks follow the photo's real edges, and skin tone is guarded.

## Setup (once per session)

If `pgrade.py` is not already present, run:
```bash
git clone -q --depth 1 https://github.com/0x6Ason/photo-grade ~/photo-grade 2>/dev/null || git -C ~/photo-grade pull -q
pip install -q -r ~/photo-grade/requirements.txt   # add --break-system-packages if pip refuses
PG="python3 $HOME/photo-grade/pgrade.py"
```
- If the clone is refused because the repo is private, attach it with the add_repo tool (owner `0x6Ason`, repo `photo-grade`) and clone again.
- Models download on first use to `~/.cache/photo-grade`: sky 2 MB, face 0.2 MB, person 6 MB. The 170 MB subject model is fetched only when the `subject` or `background` mask, or `--pop`, is used.
- Work on copies of uploads, and write results to a separate output folder.

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
   The recipe's local moves are added to the base look's own. Any top-level key overrides the base.
5. **Render:** `$PG grade PHOTO --look recipe.json --compare --show-masks -o graded/`
   - **Faces:** add `--retouch 1`, or `0.5` for men or subtle results. It softens skin and lights the face, on face skin only. `jp-airy` and `jp-film` already include it.
   - **Quick subject lift:** `--pop 1`. It is skipped automatically when there is no clear subject.
6. **Critique, then iterate one to three times.**
   - **Check `*_masks.jpg` first:** each mask must hit exactly what you meant. If not, fix the tokens or positions, or switch off a bad AI mask with `--no-mask sky`.
   - **Then check the compare image:**
     - Does the eye land where intended?
     - Are highlights or shadows clipped?
     - Is skin natural?
     - Do the colours sit together?
     - Is there a halo at any mask edge?
     - Is it overcooked? If so, lower the values or use `--strength 0.7`–`0.85`.
7. **Deliver** the full-size files plus the compare images. Add one line covering the intent and what you did.

For a set from one shoot, write one series recipe and make small per-photo tweaks so the photos match. If the user wants to choose a look, run `$PG sheet PHOTO --looks a,b,c` (or `all`) and send the labelled sheet.

## Masks

Tokens combine with `*` (intersection), and `!` inverts a token.
- **AI masks:** `sky`, `subject`, `background`, `person`, `face`, `skin`, `faceskin` (face and neck only; use for retouching)
- **Shapes and ranges:**
  - `lum:0.6-1`: a brightness range
  - `hue:185-235`: an Oklab hue range. Cyan is about 195, blue about 255, red about 25, orange about 58, green about 140. Near-neutral pixels are ignored.
  - `grad:top|bottom|left|right:EXTENT`: a gradient
  - `radial:CX,CY,SIZE`: an ellipse, positioned as fractions of the frame (`radial:SIZE` is centred)
  - `all`

Example: `"!sky*lum:0-0.3"` selects the shadows outside the sky.

## Look and recipe keys (Lightroom scale)

- **Basic:** `exposure` (EV), `contrast`, `highlights`, `shadows`, `whites`, `blacks`, `temperature`, `tint`, `vibrance`, `saturation`, `clarity`, `texture`, `dehaze`, `sharpness`
- **HSL:** `hsl: {band: [hue, sat, lum]}`, where band is red, orange, yellow, green, aqua, blue, purple or magenta
- **Colour grading:** `color_grade: {shadows|midtones|highlights|global: [hue°, sat, lum], balance}`
- **Curves:** `curve`, `curve_r`, `curve_g`, `curve_b` as `[[in, out], ...]` with values 0–255
- **Black and white:** `bw: true` plus `bw_mix: {band: value}`
- **Effects:** `bloom: {amount, threshold, radius, tint}`, `vignette: {amount, midpoint, feather}`, `grain: {amount, size}`
- **Engine:** `protect_skin` (0–1), `color_denoise` (0–1), `auto: {exposure, wb}`, `lut` (a .cube path)
- **Local moves:** `local: [{mask, amount, adjust: {...}}]`. `adjust` accepts the basic keys, `hsl` and `lut`.

## Bases

| look | for |
|---|---|
| `natural` | clean, true-to-life polish; the default when nothing stylised fits |
| `jp-airy` | portraits and selfies in light, airy settings (透明感). For dark or busy backgrounds use `jp-film` or `natural` |
| `jp-film` | people in the street, everyday life; soft Fuji pastels |
| `jp-summer` | sunny days, sea, trains, blue skies (learned from the user's own Kamakura edit; strong) |
| `film-warm` | portraits, indoor, warm and nostalgic |
| `classic-chrome` | street, documentary, muted travel |
| `teal-orange` | cinematic daylight city and travel |
| `golden` | warm light, backlit people, beaches |
| `gta6` | sunsets, dusk streets, warm neon |
| `vice-night` | blue hour and night skylines (gta6 only makes a blue sky bluer) |
| `neon-noir` | night streets, neon, rain |
| `moody` | overcast, rain, architecture, ships (warm looks turn muddy on grey days) |
| `bw` | harsh light, overcast, character |

For unusual subjects (aurora, astro, underwater, products on white), start from `natural` and don't force a style.

## Style from a reference

The final image is always the user's own pixels. A reference only supplies colour; never deliver a generated image.
- **The user edited a photo themselves:** run `$PG learn ORIGINAL EDIT --name N`. This is the most accurate option. It fits colour, clarity and texture.
- **You have an image-generation tool:** generate a restyle of the user's own photo in the wanted style, keeping the same framing. Then run `$PG learn PHOTO RESTYLE --name N --loose`, and grade the real photo with look `N`, which can be the base of a recipe.
- **A different scene** (film still, a photo the user likes): run `$PG match PHOTO REF --name N`. This works when the light and content are similar. Discard the result if it just tints the photo.
- **Lightroom preset screenshots:** transcribe the sliders 1:1 into a look JSON. Values Lightroom itself doesn't allow are ignored, for example a colour-grading blending of 260.

Always check the check image (`N_learn_check.jpg` or `N_match_check.jpg`) before using the look. Learned and matched looks are saved to `~/.config/photo-grade/looks`.

## Pro rules

- Fix exposure and white balance before style, and enhance the light that's there rather than fighting it.
- Strong styles (gta6, vice-night, jp-summer, neon-noir) need matching light. On the wrong photo they look like a filter.
- Skin must never turn orange, grey or whitened. The engine limits how far any look can change skin lightness and colour strength, but still check. Retouch faces only.
- Keep detail in highlights and shadows unless the look is deliberately faded or crushed.

## Video

Run `$PG lut --look NAME look.cube`, then:
```bash
ffmpeg -i in.mp4 -vf "lut3d=look.cube" -c:v libx264 -crf 18 -preset veryfast -c:a copy out.mp4
```
The LUT holds the global colour only. Masks, clarity, bloom, grain and vignette are not included.

## Limits

- A 12 MP photo takes about 10–25 s and up to about 2.8 GB of RAM.
- Photos are graded and saved in sRGB. Display P3 originals are converted first.
- Masks follow real edges at full resolution, but very thin structures, such as a lit spire against the sky, and blonde hair next to faces can be partly included.
- `analyze` scene types are hints; your own reading of the photo decides.
