# Discrete Image Runner

This repo is now set up around a simple Linux workflow:

1. Put raw `.jpg` or `.jpeg` files into [`input/`](/home/patrick-hutecker/Documents/working/code/input).
2. Run `make prepare` to convert those files into source `.png` files in [`output/`](/home/patrick-hutecker/Documents/working/code/output).
3. Edit [`runs.txt`](/home/patrick-hutecker/Documents/working/code/runs.txt).
4. Run `make run` to build the renderer and process every source PNG in `output/`.

## Install

```bash
sudo apt update
sudo apt install g++ make pkgconf libopencv-dev ffmpeg
```

## Commands

```bash
make prepare
make run
```

Or all at once:

```bash
make all
```

## `runs.txt` Format

Each non-empty, non-comment line is:

```txt
label intensity drift
```

Example:

```txt
basement_glow 1.05 0.45
ultraviolet_crush 1.35 1.90
```

## What The Two Numbers Mean

`intensity`

- Scales how hard the random band colors push into the shifted sample field.
- Also scales the number of shading passes.
- Lower values are gentler and faster.
- Higher values are denser, louder, and more layered.

`drift`

- Controls how quickly the discrete shift field slides across the image between passes.
- Lower values keep the shifted color field more stable and blocky.
- Higher values make the field roam faster and shear the sampled pixels more aggressively.

## Shading Profiles

`default`

- The balanced baseline shading.
- Keeps things readable while still breaking the surface with discrete texture.

Raw set:
- Passes: `8`
- Main patterns: `checker_ladder`, `scan_bars`, `corner_bloom`
- Main blends: `cross_stitch`, `north_pull`, `halo`, `scanline`, `stutter_step`

`moody`

- Darker and heavier.
- Feels more pooled and downward-pulled.

Raw set:
- Passes: `10`
- Main patterns: `checker_ladder`, `corner_bloom`, `scan_bars`
- Main blends: `north_pull`, `halo`

`soft`

- Lighter and more pastel.
- Better when you want the source image to stay readable.

Raw set:
- Passes: `6`
- Main patterns: `checker_ladder`, `corner_bloom`, `scan_bars`
- Main blends: `cross_stitch`, `halo`

`electric`

- The loudest preset.
- Best for stronger posterized or club-light energy.

Raw set:
- Passes: `9`
- Main patterns: `scan_bars`, `checker_ladder`, `corner_bloom`
- Main blends: `scanline`, `stutter_step`, `cross_stitch`

`wash`

- More airy and smeared.
- Less contrasty than `electric`, less brooding than `moody`.

Raw set:
- Passes: `7`
- Main patterns: `corner_bloom`, `checker_ladder`, `scan_bars`
- Main blends: `halo`, `cross_stitch`

## How It Works Now

Each line in `runs.txt` defines one overall run mood:

- how strong to push
- how fast the pattern field drifts

Inside each run, the renderer invents 3 to 4 awkward RMS bands on its own. Each of those bands gets:

- its own random RMS window
- its own random color
- its own discrete shift field
- its own discrete mix pattern

Inside a matching RMS band, pixels are always acted on. The renderer no longer relies on sparse probability gating to decide whether a pixel gets colored. Instead, a discrete shift field moves where the source sample comes from, and that shifted sample is what gets dragged toward the run color.

Then, for every run line, the renderer chooses two random shading profiles and applies them in sequence. That means the double-run behavior is now deliberate, while the color band itself stays anchored to the input you wrote.

## Output Naming

For a source image like:

```txt
output/computer.png
```

and a run line like:

```txt
basement_glow 1.05 0.45
```

the output becomes:

```txt
output/computer_basement_glow_mix_soft_electric_i105_d45.png
```
