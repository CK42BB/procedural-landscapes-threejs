# Noise Algorithms Reference

Mathematical foundations and advanced noise variants for procedural terrain generation.

## Core Concepts

### Frequency, Amplitude, and Octaves

Procedural noise is built by layering frequencies:

- **Frequency**: How "zoomed in" the noise is. Higher = more detail, smaller features.
- **Amplitude**: How much influence each layer has on the final value.
- **Octave**: A single noise layer at a specific frequency/amplitude.
- **Lacunarity**: Frequency multiplier between octaves (typically 2.0).
- **Persistence/Gain**: Amplitude multiplier between octaves (typically 0.5).

Standard FBM stacks 4–8 octaves. More octaves = more fine detail but diminishing returns past 8.

### Value Range Normalization

Raw FBM output ranges approximately `[-1, 1]`. For terrain height, remap to `[0, 1]`:

```javascript
const normalized = fbm(...) * 0.5 + 0.5;
```

For ridged noise, output is `[0, ~2]` depending on octaves. Divide by peak theoretical value or clamp.

## Advanced Noise Variants

### Terraced Noise

Creates stepped, plateau-like terrain (mesa formations, rice paddies).

```javascript
function terraced(noise, x, y, steps = 8) {
  const n = fbm(noise, x, y, 6) * 0.5 + 0.5;
  return Math.round(n * steps) / steps;
}

// Smooth terrace edges with interpolation
function smoothTerraced(noise, x, y, steps = 8, sharpness = 4) {
  const n = fbm(noise, x, y, 6) * 0.5 + 0.5;
  const stepped = Math.round(n * steps) / steps;
  const frac = (n * steps) % 1;
  const smooth = 1 / (1 + Math.exp(-sharpness * (frac - 0.5)));
  return stepped + smooth / steps;
}
```

### Voronoi / Worley Noise

Cellular noise producing patterns resembling cracked earth, stone tiles, or organic cells.

```javascript
function voronoi(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  let minDist = Infinity;
  let secondDist = Infinity;

  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      // Deterministic random point per cell
      const cx = ix + dx, cy = iy + dy;
      const h = Math.sin(cx * 127.1 + cy * 311.7) * 43758.5453;
      const px = dx + (h - Math.floor(h)) - fx;
      const py = dy + (Math.sin(cx * 269.5 + cy * 183.3) * 43758.5453 % 1) - fy;

      const dist = px * px + py * py;
      if (dist < minDist) {
        secondDist = minDist;
        minDist = dist;
      } else if (dist < secondDist) {
        secondDist = dist;
      }
    }
  }

  return {
    f1: Math.sqrt(minDist),           // Distance to closest point
    f2: Math.sqrt(secondDist),        // Distance to second closest
    edge: Math.sqrt(secondDist) - Math.sqrt(minDist), // Edge detection
  };
}

// Use f2 - f1 for cracks/ridges, f1 for cells
function voronoiTerrain(noise, x, y, scale = 2) {
  const v = voronoi(x * scale, y * scale);
  const base = fbm(noise, x, y, 4) * 0.5 + 0.5;
  return base * 0.7 + v.f1 * 0.3; // Cellular rocky terrain
}
```

### Analytical Derivatives

Compute noise derivatives alongside values for slope-aware operations (erosion direction,
normal computation) without finite differences.

```javascript
// Returns { value, dx, dy } - the noise value and its partial derivatives
function fbmWithDerivatives(noise, x, y, octaves = 6, lac = 2.0, gain = 0.5) {
  let sum = 0, dx = 0, dy = 0;
  let amp = 1, freq = 1, maxAmp = 0;
  const eps = 0.001;

  for (let i = 0; i < octaves; i++) {
    const nx = x * freq, ny = y * freq;
    const v = noise(nx, ny);
    // Finite difference approximation of derivatives
    const vx = noise(nx + eps, ny);
    const vy = noise(nx, ny + eps);

    sum += v * amp;
    dx += ((vx - v) / eps) * amp * freq;
    dy += ((vy - v) / eps) * amp * freq;
    maxAmp += amp;
    amp *= gain;
    freq *= lac;
  }
  return { value: sum / maxAmp, dx: dx / maxAmp, dy: dy / maxAmp };
}
```

### Curl Noise (2D)

Divergence-free noise field useful for wind patterns, water flow visualization, and
particle advection over terrain.

```javascript
function curlNoise2D(noise, x, y, eps = 0.01) {
  // Curl of a 2D scalar field gives a 2D vector
  const dndx = (noise(x + eps, y) - noise(x - eps, y)) / (2 * eps);
  const dndy = (noise(x, y + eps) - noise(x, y - eps)) / (2 * eps);
  // Rotate 90 degrees for divergence-free field
  return { x: dndy, y: -dndx };
}

// Use for river path generation
function traceFlowPath(noise, startX, startY, steps = 200, stepSize = 0.5) {
  const path = [{ x: startX, y: startY }];
  let x = startX, y = startY;

  for (let i = 0; i < steps; i++) {
    const flow = curlNoise2D(noise, x * 0.1, y * 0.1);
    x += flow.x * stepSize;
    y += flow.y * stepSize;
    path.push({ x, y });
  }
  return path;
}
```

### Billowed Noise

Soft, cloud-like variant. Absolute value of FBM creates pillow-shaped undulations.

```javascript
function billowed(noise, x, y, octaves = 6) {
  let sum = 0, amp = 1, freq = 1, maxAmp = 0;
  for (let i = 0; i < octaves; i++) {
    sum += Math.abs(noise(x * freq, y * freq)) * amp;
    maxAmp += amp;
    amp *= 0.5;
    freq *= 2.0;
  }
  return sum / maxAmp;
}
```

## Terrain Composition Recipes

### Island with Ocean Falloff

```javascript
function islandTerrain(noise, x, y, islandRadius = 0.4) {
  const cx = x - 0.5, cy = y - 0.5;
  const dist = Math.sqrt(cx * cx + cy * cy);
  const falloff = Math.max(0, 1 - Math.pow(dist / islandRadius, 2));

  const height = domainWarp(noise, x * 3, y * 3, 0.4) * 0.5 + 0.5;
  return height * falloff * falloff; // Squared falloff for smooth coastline
}
```

### Mountain Range with Foothills

```javascript
function mountainRange(noise, x, y) {
  // Directional ridge along x-axis
  const ridgeLine = Math.exp(-y * y * 8); // Gaussian ridge mask
  const peaks = ridgedMultifractal(noise, x * 2, y * 2, 6) * ridgeLine;
  const foothills = fbm(noise, x, y, 4) * 0.5 + 0.5;
  return foothills * 0.3 + peaks * 0.7;
}
```

### Erosion-Like Channels (CPU approximation)

```javascript
function erodedTerrain(noise, x, y, iterations = 3) {
  let h = fbm(noise, x, y, 6) * 0.5 + 0.5;
  for (let i = 0; i < iterations; i++) {
    // Carve channels following steepest descent
    const { dx, dy } = fbmWithDerivatives(noise, x * (1 + i * 0.5), y * (1 + i * 0.5), 4);
    const slope = Math.sqrt(dx * dx + dy * dy);
    h -= slope * 0.05 * (1 / (i + 1)); // Diminishing erosion per pass
  }
  return Math.max(h, 0);
}
```

## Seed Management

Consistent seeding enables reproducible terrain for multiplayer or save/load.

```javascript
// Create deterministic noise instances for different layers
function createSeededLayers(worldSeed) {
  return {
    elevation: createNoise2D(worldSeed),
    moisture:  createNoise2D(worldSeed + 1),
    temperature: createNoise2D(worldSeed + 2),
    detail:    createNoise2D(worldSeed + 3),
    vegetation: createNoise2D(worldSeed + 4),
  };
}

// Each layer uses different frequency ranges for independent variation
function sampleBiomeData(layers, x, y) {
  return {
    height: fbm(layers.elevation, x, y, 6),
    moisture: fbm(layers.moisture, x * 0.7, y * 0.7, 4) * 0.5 + 0.5,
    temperature: fbm(layers.temperature, x * 0.3, y * 0.3, 3) * 0.5 + 0.5,
  };
}
```

## Performance Considerations

- **CPU noise at 256×256**: ~5–15ms. Acceptable for single chunks on load.
- **CPU noise at 1024×1024**: ~100–300ms. Use web workers or GPU compute.
- **GPU compute at 1024×1024**: ~1–5ms. Essential for real-time modification.
- **Octave count**: Diminishing visual returns past 6 octaves. Use 4 for distant LOD.
- **Cache heightmaps**: Store computed heights in `Float32Array` for physics/collision queries rather than recomputing per frame.
