# Brand assets

| file | use |
|---|---|
| `logo.svg` | README header, light theme |
| `logo-dark.svg` | README header, dark theme (selected by `prefers-color-scheme`) |
| `icon.svg` | square mark, for favicons and avatars -- a tighter crop that still reads at 16px |
| `social-preview.svg` | source for the social preview |
| `social-preview.png` | 1280x640, uploaded via **Settings -> Social preview** |

The mark is a sealed bundle straddling a dashed boundary: the package, and the
air gap it crossed.

Colours: `#1B4965` and `#5FA8D3` on light, `#62B6CB` and `#123A52` on dark,
with `#94A3B8` / `#6E7B8B` for the boundary.

The SVGs are hand-written and should be edited as text. The PNG exists only
because GitHub's social preview will not accept an SVG; regenerate it from
`social-preview.svg` rather than editing it. On macOS, with no SVG rasteriser
installed, the source is laid out on a 1280x1280 canvas with the artwork in the
centre band so that the square thumbnail can be cropped back to size:

```bash
qlmanage -t -s 1280 -o /tmp/out assets/social-preview.svg
sips -c 640 1280 /tmp/out/social-preview.svg.png --out assets/social-preview.png
```
