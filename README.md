# Procedural Landscapes — Three.js Skill

A Claude Code skill for generating procedural 3D terrain in Three.js, emphasizing **WebGPU with WebGL fallback**.

Part of the **Teaching Three.js** skill series.

## What This Skill Does

When installed, this skill teaches Claude Code how to generate procedural landscapes including:

- **Heightmap generation** via FBM, ridged multifractal, and domain warping noise
- **WebGPU-first rendering** with automatic WebGL2 fallback
- **Chunked LOD terrain** with distance-based level of detail
- **Slope + height materials** using custom shaders (GLSL) and TSL node materials (WebGPU)
- **Procedural water** with animated wave displacement
- **Atmospheric sky dome** with gradient and sun disc
- **Instanced vegetation** scattered by height/slope/noise constraints
- **GPU compute shaders** (WGSL) for real-time heightmap generation and hydraulic erosion

## Installation

### Claude Code (CLI)

```bash
claude install-skill path/to/procedural-landscapes
```

Or copy the `procedural-landscapes/` folder into your Claude Code skills directory.

### Manual Usage

You can also reference the skill files directly in conversations with Claude. The key files are:

| File | Purpose |
|------|---------|
| `SKILL.md` | Main skill — terrain pipeline, materials, vegetation, full scene assembly |
| `references/wgsl-shaders.md` | WGSL compute shaders for GPU heightmap gen, erosion, normals |
| `references/noise-algorithms.md` | Advanced noise variants, composition recipes, seed management |

## Quick Start Prompts

Try these with Claude after installing:

> "Create a procedural island landscape with mountains and a water plane"

> "Build a chunked terrain system with LOD that loads as the camera moves"

> "Generate a terrain with ridged multifractal noise and slope-based texturing"

> "Create a WebGPU terrain with compute shader heightmap generation"

## Requirements

- **Three.js r170+** (for WebGPU support and TSL node materials)
- A modern browser with WebGPU support (Chrome 121+, Edge 121+) for GPU features
- WebGL2 fallback works in all modern browsers

## Skill Architecture

```
procedural-landscapes/
├── SKILL.md                          # Core skill (read first)
├── README.md                         # This file
└── references/
    ├── wgsl-shaders.md               # GPU compute shaders
    └── noise-algorithms.md           # Noise math & composition
```

Claude reads `SKILL.md` first, then loads reference files as needed based on the task complexity.

## Key Concepts

### WebGPU / WebGL Dual Backend

The skill always checks for WebGPU availability first:

```javascript
if (WebGPU.isAvailable()) {
  // WebGPURenderer + TSL node materials + GPU compute
} else {
  // WebGLRenderer + GLSL ShaderMaterial + CPU noise
}
```

This ensures your terrain works everywhere while taking advantage of GPU compute where available.

### Noise Layering

Complex terrain comes from combining noise types:

| Layer | Noise Type | Effect |
|-------|-----------|--------|
| Base elevation | FBM | Rolling hills |
| Mountains | Ridged Multifractal | Sharp peaks and valleys |
| Organic detail | Domain Warping | Twisted, natural features |
| Surface detail | High-frequency FBM | Fine ground texture |

### Performance Targets

| Platform | Max Vertices | Chunks |
|----------|-------------|--------|
| Mobile | 500K total | 4–8 visible |
| Desktop | 2M total | 16–32 visible |
| With LOD | Budget per ring | High-res near, low-res far |

## Series: Teaching Three.js

This skill is part of a series designed for learning and teaching Three.js with Claude Code. Each skill is independent and can be used standalone or combined with others.
[procedural-landscapes](../procedural-landscapes-threejs/) for terrain, [procedural-grass](../procedural-grass-threejs/) for ground cover, [procedural-clouds](../procedural-clouds-threejs/), [procedural-weather](../procedural-weather-threejs/).

## License

MIT — use freely in your projects.
