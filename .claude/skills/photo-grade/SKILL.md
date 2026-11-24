---
name: photo-grade
description: One-step colour grading of the user's photos (JPEG/HEIC/PNG/TIFF/RAW) with 10 looks (GTA VI, cinematic, film, moody, B&W...) and AI sky/skin/subject masks, run from the CLI. Use when asked to grade, edit or restyle photos.
---

# photo-grade

The engine is `pgrade.py` from https://github.com/0x6Ason/photo-grade (Python, numpy, OpenCV, ONNX; no Lightroom). It grades on full-resolution float data. Its AI masks are refined to the photo's real edges, and skin is protected from strong colour casts.

## Setup (once per session)

If `pgrade.py` is not already present (for example, you are not inside the repo), run:
```bash
git clone -q --depth 1 https://github.com/0x6Ason/photo-grade ~/photo-grade 2>/dev/null || git -C ~/photo-grade pull -q
pip install -q -r ~/photo-grade/requirements.txt   # add --break-system-packages if pip refuses
PG="python3 $HOME/photo-grade/pgrade.py"
```
- If the clone is refused because the repo is private, attach it first with the add_repo tool (owner `0x6Ason`, repo `photo-grade`), then clone again.
- Models download on first use to `~/.cache/photo-grade`: sky 2 MB, face 0.2 MB, person 6 MB. The 170 MB subject model is fetched only for `--pop`.
- Work on copies of the user's uploads, and write results to a separate output folder.

## One-step workflow

1. **Look at every photo yourself**, then run `$PG analyze PHOTOS`. It prints the scene type and 3 suggested looks for each photo.
2. **Pick the look.**
   - If the user named a look or style, use it.
   - Otherwise take the first suggestion, unless what you see contradicts it (use the guide below).
   - For a set from one shoot, use one look for every photo so they match. Use `--look auto` only when the photos are unrelated.
3. **Grade:** `$PG grade PHOTOS --look NAME -o graded/ --compare`
   - If there is one clear subject (person, car, pet, product), add `--pop 1`. The engine skips it on its own when there is no clear subject.
4. **Check every `*_compare.jpg` before sending.** Fix what you see:
   - Too strong or garish: rerun with `--strength 0.6`–`0.8`.
   - Halo or odd colour along the horizon or rooftops: run `$PG masks PHOTO` and look at the preview. If the sky mask is wrong, add `--no-mask sky`.
   - Skin looks off: check that the face was found (white box in the masks preview). Try `natural` or `film-warm`, or lower the strength.
   - Dark night or RAW file looks muddy: try `natural` first, then a stronger look.
5. **Deliver** the full-size graded files, plus the compare images when the difference matters. Add one line saying which look and strength you used.

If the user wants to choose, run `$PG sheet PHOTO --looks suggested` (or `all`) and send the sheet. It shows one photo in each look, labelled.

## Looks and when to use them

| look | for |
|---|---|
| `natural` | anything; the clean, safe default |
| `gta6` | sunsets, golden hour, warm or neon streets at dusk |
| `vice-night` | blue hour, night skylines. Use this, not gta6, for these: gta6 only makes a blue sky bluer |
| `neon-noir` | night streets with neon, rain reflections |
| `teal-orange` | daylight with blue sky: city, travel, landscape |
| `golden` | warm light, backlit people, beaches |
| `film-warm` | portraits, people, indoor, lifestyle |
| `classic-chrome` | street, documentary, muted travel |
| `moody` | overcast, grey, rain, architecture, ships. Warm looks turn muddy on grey days |
| `bw` | overcast, harsh light, architecture, portraits with character |

For unusual subjects (aurora, astro, underwater, products on white), use `natural` and don't force a style.

## Custom looks

A look is a JSON file in `looks/` or `~/.config/photo-grade/looks/`. You can also pass a file path to `--look`. Use `"base": "gta6"` to inherit from another look and override only some keys.

All keys use the Lightroom scale:
- **Basic:** `exposure` (EV), `contrast`, `highlights`, `shadows`, `whites`, `blacks`, `temperature`, `tint`, `vibrance`, `saturation`, `clarity`, `texture`, `dehaze`, `sharpness`
- **HSL:** `hsl: {band: [hue, sat, lum]}`, where band is red, orange, yellow, green, aqua, blue, purple or magenta
- **Colour grading:** `color_grade: {shadows|midtones|highlights|global: [hue°, sat, lum], balance}`
- **Curves:** `curve`, `curve_r`, `curve_g`, `curve_b` as `[[in, out], ...]` with values 0–255
- **Black and white:** `bw: true` plus `bw_mix: {band: value}`
- **Effects:** `bloom: {amount, threshold, radius, tint}`, `vignette: {amount, midpoint, feather}`, `grain: {amount, size}`
- **Engine:** `protect_skin` (0–1), `color_denoise` (0–1), `auto: {exposure, wb}`
- **Local adjustments:** `local: [{mask, amount, adjust: {...}}]`

Mask expressions combine tokens with `*` (intersection), and `!` inverts a token. Tokens:
- `sky`, `person`, `face`, `skin`, `subject`, `background`, `all`
- `lum:0.6-1`, a brightness range
- `hue:200-280`, an Oklab hue range
- `grad:top:0.5`, a gradient
- `radial:0.6`, a centred ellipse

For example, `"sky"` or `"!sky*lum:0-0.4"`.

**Copying a Lightroom preset from screenshots:** transcribe the sliders 1:1 into these keys and save the file as a new look. Values that Lightroom itself does not allow are ignored, for example a colour-grading blending of 260. Tell the user which sliders the engine lacks; tone curves, vignette and grain are supported.

## Video

Run `$PG lut --look NAME look.cube`, then:
```bash
ffmpeg -i in.mp4 -vf "lut3d=look.cube" -c:v libx264 -crf 18 -preset veryfast -c:a copy out.mp4
```
The LUT holds the global colour only. Masks, clarity, bloom, grain and vignette are not included.

## Limits

- A 12 MP photo grades in about 13–15 s and peaks at about 2.7 GB of RAM with `--pop` (2.2 GB without). The subject model adds about 3 s.
- Masks are refined at full resolution, but very thin structures can be partly included in a mask, for example a lit spire against the sky.
- Auto scene detection is a heuristic. Your own look at the photo wins.
