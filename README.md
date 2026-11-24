# photo-grade

One-step colour grading for your own photos, from the command line. It has 13 looks, AI masks for sky, people, skin and subject, and gentle skin retouching. It can also learn a new look from any before/after pair you edited yourself. No Lightroom needed.

```bash
pip install -r requirements.txt
python3 pgrade.py grade IMG_1234.jpg                 # auto: detects the scene, picks a look
python3 pgrade.py grade shoot/ --look gta6 --compare # whole folder, one look, before/after images
python3 pgrade.py sheet IMG_1234.jpg                 # one photo in every look, side by side
python3 pgrade.py learn raw.jpg my_edit.jpg --name mine   # teach it your own grade
```

## Looks

| look | what it does | best for |
|---|---|---|
| `natural` | clean polish: balanced exposure and white balance, gentle contrast and colour | anything; the safe default |
| `gta6` | GTA VI / Vice City: lavender wash, teal shadows, warm highlights, faded navy blacks | sunsets, golden hour, warm streets |
| `vice-night` | GTA VI after dark: violet-magenta skies over gold city lights | blue hour, night skylines |
| `teal-orange` | blockbuster cinematic: teal shadows and skies, warm skin | daylight city, travel, landscapes |
| `jp-airy` | Japanese airy portrait (透明感): bright, soft, pastel shadows, fair luminous skin, mint greens | portraits, selfies, cafes, window light |
| `jp-film` | Japanese film (Fuji 400H / Superia feel): pastel tones, green-cyan shadows, rosy highlights | people outdoors, street, everyday life |
| `jp-summer` | Japanese summer, learned from a real edit: cyan-teal skies, bright clouds, warm sunlit rails | sunny days, seaside, trains, travel |
| `film-warm` | Portra-style film: soft roll-off, lifted blacks, glowing skin, grain | portraits, people, indoor |
| `classic-chrome` | muted documentary film, firm shadows, cyan blues | street, documentary, overcast |
| `moody` | dark, cool, desaturated, heavy skies, vignette | grey days, rain, architecture |
| `golden` | golden-hour glow with soft highlight bloom | sunsets, backlit portraits, beaches |
| `neon-noir` | cyberpunk night: magenta/cyan neon, inky blacks, glow | night streets, neon, rain |
| `bw` | punchy red-filter black & white with grain | overcast, architecture, character portraits |

Add your own looks by dropping a JSON file in `looks/` or `~/.config/photo-grade/looks/`. A look can inherit from another with `"base": "gta6"`. All values use the Lightroom slider scale, so you can copy a Lightroom preset's sliders straight across.

Or **learn a look from your own edit.** Edit one photo in any app, then give `learn` the original and your edited version:

```bash
python3 pgrade.py learn DSC_2281.jpg DSC_2281_edit.jpg --name kamakura
python3 pgrade.py grade more_photos/ --look kamakura
```

It fits a smooth 3D colour LUT, filling in colours the example never contained from their neighbours. It also estimates the clarity and texture the LUT can't capture. Then it saves a check image (original | learned | your edit) so you can confirm the match. `jp-summer` was made this way from one Kamakura edit: the colour error dropped from 40 to 10 (out of 255).

## What makes it good

- **AI masks with refined edges.** A sky segmenter (2 MB), a face detector (YuNet), a person segmenter (PP-HumanSeg) and an optional subject model (ISNet) run at 1024 px. Each mask is then refined at full resolution with an edge-aware colour guided filter. That gives clean skylines, hair and branches, with no halos.
- **Skin protection.** Strong looks keep their tone, but skin hue and saturation are pulled back toward natural, so faces don't turn purple or orange.
- **Skin retouching.** `--retouch 1` softens skin texture and adds a little light to faces. The `jp-*` looks do this by default. It uses a face-only skin mask: a colour and lightness model sampled from each detected face, limited to the person's outline and the face area. Hair, lips, eyes and backgrounds stay sharp.
- **Local adjustments per look.** For example, `moody` darkens and dehazes only the sky, and `teal-orange` shifts only the sky's hue. Use `--pop` to lift the subject and quieten the background.
- **Clean colour maths.** Colour operations run in Oklab, with continuous gamut mapping and colour noise reduction. Strong grades don't create blotches or banding in skies, and output is dithered.
- **Scene-aware auto mode.** The engine sorts each photo into golden hour, blue hour, night, neon night, overcast, daylight or portrait, then picks the look.
- **Inputs and outputs.** It reads JPEG, PNG, TIFF (8/16-bit), HEIC and camera RAW. Wide-gamut photos, such as iPhone Display P3, are converted to sRGB first, so every look behaves the same on every camera. EXIF is kept, and it writes 16-bit TIFF from RAW if you ask for it.

## Commands

```
pgrade.py analyze IMG...                 scene report + 3 suggested looks (JSON)
pgrade.py grade   IMG|DIR... [options]   --look NAME|auto|file.json  --strength 0..1.5
                                         --pop 0..2  --retouch 0..2  --compare  --no-mask sky,skin  --format tif
pgrade.py sheet   IMG [--looks all|suggested|a,b,c]
pgrade.py masks   IMG [--subject]        preview the AI masks
pgrade.py looks                          list looks
pgrade.py learn   BEFORE AFTER --name N learn a look from your own edit of one photo
pgrade.py lut     --look NAME out.cube   global colour as a 3D LUT (ffmpeg, Resolve, Premiere, CapCut)
```

Models download on first use to `~/.cache/photo-grade`. The face, person and sky models total under 10 MB; the 170 MB subject model is only fetched for `--pop` or a look that uses the `subject` mask.

## Licence and credits

The code is released under the MIT licence; see `LICENSE`.

- The `gta6` slider values come from a preset shared by @lightrooms.id.
- Sky model: [xiongzhu666/Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing) (MIT).
- Face detection (YuNet) and person segmentation (PP-HumanSeg) come from [opencv_zoo](https://github.com/opencv/opencv_zoo).
- The subject model (ISNet) comes via [rembg](https://github.com/danielgatis/rembg).
