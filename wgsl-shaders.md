# WGSL Shaders for Procedural Terrain

Complete WebGPU Shading Language (WGSL) implementations for GPU-accelerated terrain
generation. Use with Three.js r170+ WebGPURenderer and TSL compute nodes.

## Heightmap Compute Shader

Generate a heightmap texture entirely on the GPU using FBM noise in a compute shader.

### Three.js TSL Compute Setup

```javascript
import { compute, storageTexture, textureStore, uvec2, vec2, float as tslFloat,
         instanceIndex, textureSize } from 'three/tsl';

function createHeightmapCompute(resolution = 512) {
  const heightmap = new THREE.StorageTexture(resolution, resolution);
  heightmap.type = THREE.FloatType;

  // Compute kernel: one thread per texel
  const computeFn = Fn(() => {
    const idx = instanceIndex;
    const size = textureSize(storageTexture(heightmap));
    const x = idx.modInt(size.x);
    const y = idx.div(size.x);
    const uv = vec2(x, y).div(vec2(size));

    // FBM noise in TSL
    const height = tslFbm(uv, 6);
    textureStore(storageTexture(heightmap), uvec2(x, y), vec4(height, 0, 0, 1));
  });

  const computeNode = compute(computeFn, [resolution * resolution]);
  return { heightmap, computeNode };
}
```

### Raw WGSL Compute Shader

For direct WGSL usage or custom pipelines outside TSL:

```wgsl
// heightmap_compute.wgsl
@group(0) @binding(0) var output: texture_storage_2d<r32float, write>;

struct Params {
  resolution: u32,
  scale: f32,
  octaves: u32,
  lacunarity: f32,
  gain: f32,
  seed: f32,
  offset_x: f32,
  offset_z: f32,
}
@group(0) @binding(1) var<uniform> params: Params;

// Permutation-free gradient noise (no lookup table needed on GPU)
fn hash2(p: vec2<f32>) -> vec2<f32> {
  var q = vec2<f32>(
    dot(p, vec2<f32>(127.1, 311.7)),
    dot(p, vec2<f32>(269.5, 183.3))
  );
  return fract(sin(q) * 43758.5453) * 2.0 - 1.0;
}

fn gradientNoise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f); // Hermite interpolation

  let a = dot(hash2(i + vec2<f32>(0.0, 0.0)), f - vec2<f32>(0.0, 0.0));
  let b = dot(hash2(i + vec2<f32>(1.0, 0.0)), f - vec2<f32>(1.0, 0.0));
  let c = dot(hash2(i + vec2<f32>(0.0, 1.0)), f - vec2<f32>(0.0, 1.0));
  let d = dot(hash2(i + vec2<f32>(1.0, 1.0)), f - vec2<f32>(1.0, 1.0));

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, octaves: u32, lac: f32, gain: f32) -> f32 {
  var sum = 0.0;
  var amp = 1.0;
  var freq = 1.0;
  var maxAmp = 0.0;
  var pos = p;

  for (var i = 0u; i < octaves; i++) {
    sum += gradientNoise(pos * freq) * amp;
    maxAmp += amp;
    amp *= gain;
    freq *= lac;
  }
  return sum / maxAmp;
}

fn ridged(p: vec2<f32>, octaves: u32, lac: f32, gain: f32) -> f32 {
  var sum = 0.0;
  var amp = 1.0;
  var freq = 1.0;
  var prev = 1.0;

  for (var i = 0u; i < octaves; i++) {
    var n = 1.0 - abs(gradientNoise(p * freq));
    n = n * n * prev;
    sum += n * amp;
    prev = n;
    amp *= gain;
    freq *= lac;
  }
  return sum;
}

fn domainWarp(p: vec2<f32>, strength: f32) -> f32 {
  let qx = fbm(p, 4u, 2.0, 0.5);
  let qy = fbm(p + vec2<f32>(5.2, 1.3), 4u, 2.0, 0.5);
  return fbm(p + strength * vec2<f32>(qx, qy), 6u, 2.0, 0.5);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (gid.x >= res || gid.y >= res) { return; }

  let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / f32(res);
  let worldPos = (uv - 0.5) * params.scale + vec2<f32>(params.offset_x, params.offset_z);

  // Combine noise layers
  let base = fbm(worldPos * 0.5, params.octaves, params.lacunarity, params.gain) * 0.5 + 0.5;
  let mountains = ridged(worldPos, params.octaves, params.lacunarity, params.gain) * 0.4;
  let warp = domainWarp(worldPos, 0.4) * 0.1;

  let height = max(base * 0.5 + mountains * base + warp, 0.0);

  textureStore(output, vec2<u32>(gid.x, gid.y), vec4<f32>(height, 0.0, 0.0, 1.0));
}
```

### Binding the Compute Pipeline in Three.js

```javascript
// Manual WebGPU compute pipeline (when TSL abstractions aren't sufficient)
async function runHeightmapCompute(renderer, resolution, params) {
  const device = renderer.backend.device;

  // Create storage texture
  const texture = device.createTexture({
    size: [resolution, resolution],
    format: 'r32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  // Create uniform buffer
  const uniformData = new ArrayBuffer(32);
  const view = new DataView(uniformData);
  view.setUint32(0, resolution, true);
  view.setFloat32(4, params.scale ?? 4.0, true);
  view.setUint32(8, params.octaves ?? 6, true);
  view.setFloat32(12, params.lacunarity ?? 2.0, true);
  view.setFloat32(16, params.gain ?? 0.5, true);
  view.setFloat32(20, params.seed ?? 0.0, true);
  view.setFloat32(24, params.offsetX ?? 0.0, true);
  view.setFloat32(28, params.offsetZ ?? 0.0, true);

  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // Dispatch
  const workgroups = Math.ceil(resolution / 16);
  // ... create pipeline, bind group, dispatch, read back texture
  return texture;
}
```

## Normal Computation Shader

Compute normals from a heightmap texture on the GPU for smooth lighting.

```wgsl
// normals_compute.wgsl
@group(0) @binding(0) var heightmap: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8snorm, write>;

struct Params { resolution: u32, heightScale: f32, texelSize: f32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (gid.x >= res || gid.y >= res) { return; }

  let x = i32(gid.x);
  let y = i32(gid.y);

  // Sample neighbors (clamped at edges)
  let hL = textureLoad(heightmap, vec2<i32>(max(x - 1, 0), y), 0).r;
  let hR = textureLoad(heightmap, vec2<i32>(min(x + 1, i32(res) - 1), y), 0).r;
  let hD = textureLoad(heightmap, vec2<i32>(x, max(y - 1, 0)), 0).r;
  let hU = textureLoad(heightmap, vec2<i32>(x, min(y + 1, i32(res) - 1)), 0).r;

  let scale = params.heightScale * params.texelSize;
  let normal = normalize(vec3<f32>(
    (hL - hR) * scale,
    2.0,
    (hD - hU) * scale
  ));

  textureStore(output, vec2<u32>(gid.xy), vec4<f32>(normal, 1.0));
}
```

## Hydraulic Erosion Compute Shader

Simulate water erosion on the heightmap for realistic terrain features.

```wgsl
// erosion_compute.wgsl
@group(0) @binding(0) var heightmap: texture_storage_2d<r32float, read_write>;

struct ErosionParams {
  resolution: u32,
  droplets: u32,
  inertia: f32,
  capacity: f32,
  deposition: f32,
  erosion: f32,
  evaporation: f32,
  gravity: f32,
  maxSteps: u32,
  _pad: u32,
}
@group(0) @binding(1) var<uniform> params: ErosionParams;

fn sampleHeight(pos: vec2<f32>, res: u32) -> f32 {
  let p = clamp(pos, vec2<f32>(0.0), vec2<f32>(f32(res) - 1.0));
  let ip = vec2<u32>(u32(p.x), u32(p.y));
  return textureLoad(heightmap, ip).r;
}

fn calcGradient(pos: vec2<f32>, res: u32) -> vec2<f32> {
  let eps = 1.0;
  let hL = sampleHeight(pos - vec2<f32>(eps, 0.0), res);
  let hR = sampleHeight(pos + vec2<f32>(eps, 0.0), res);
  let hD = sampleHeight(pos - vec2<f32>(0.0, eps), res);
  let hU = sampleHeight(pos + vec2<f32>(0.0, eps), res);
  return vec2<f32>(hR - hL, hU - hD) * 0.5;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.droplets) { return; }

  let res = params.resolution;
  // Pseudo-random starting position using thread ID
  let seed = f32(gid.x) * 0.618033988749;
  var pos = vec2<f32>(fract(seed * 127.1) * f32(res), fract(seed * 311.7) * f32(res));
  var dir = vec2<f32>(0.0);
  var speed = 1.0;
  var water = 1.0;
  var sediment = 0.0;

  for (var step = 0u; step < params.maxSteps; step++) {
    let grad = calcGradient(pos, res);
    dir = dir * params.inertia - grad * (1.0 - params.inertia);

    if (length(dir) < 0.0001) { break; }
    dir = normalize(dir);

    let oldH = sampleHeight(pos, res);
    pos += dir;

    if (pos.x < 0.0 || pos.x >= f32(res) || pos.y < 0.0 || pos.y >= f32(res)) { break; }

    let newH = sampleHeight(pos, res);
    let dH = newH - oldH;

    let cap = max(-dH * speed * water * params.capacity, 0.01);

    if (sediment > cap || dH > 0.0) {
      let deposit = select((sediment - cap) * params.deposition, min(dH, sediment), dH > 0.0);
      sediment -= deposit;
      let ip = vec2<u32>(u32(pos.x), u32(pos.y));
      let h = textureLoad(heightmap, ip).r + deposit;
      textureStore(heightmap, ip, vec4<f32>(h, 0.0, 0.0, 1.0));
    } else {
      let erode = min((cap - sediment) * params.erosion, -dH);
      sediment += erode;
      let ip = vec2<u32>(u32(pos.x), u32(pos.y));
      let h = textureLoad(heightmap, ip).r - erode;
      textureStore(heightmap, ip, vec4<f32>(h, 0.0, 0.0, 1.0));
    }

    speed = sqrt(max(speed * speed + dH * params.gravity, 0.0));
    water *= (1.0 - params.evaporation);
  }
}
```

## Vertex Displacement Shader

Instead of modifying geometry on the CPU, displace vertices directly from the
heightmap texture in the vertex shader.

### GLSL (WebGL fallback)

```glsl
// terrain_displacement.vert
uniform sampler2D heightmap;
uniform float maxHeight;
uniform vec2 terrainOffset;
uniform float terrainScale;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec2 vUv;

void main() {
  vUv = uv;
  vec2 samplePos = uv * terrainScale + terrainOffset;
  float h = texture2D(heightmap, samplePos).r * maxHeight;

  vec3 pos = position;
  pos.y += h;

  // Compute normal from heightmap neighbors
  float texel = 1.0 / 512.0; // Match heightmap resolution
  float hL = texture2D(heightmap, samplePos + vec2(-texel, 0.0)).r * maxHeight;
  float hR = texture2D(heightmap, samplePos + vec2( texel, 0.0)).r * maxHeight;
  float hD = texture2D(heightmap, samplePos + vec2(0.0, -texel)).r * maxHeight;
  float hU = texture2D(heightmap, samplePos + vec2(0.0,  texel)).r * maxHeight;

  vec3 computedNormal = normalize(vec3(hL - hR, 2.0, hD - hU));

  vWorldPos = (modelMatrix * vec4(pos, 1.0)).xyz;
  vNormal = normalize(normalMatrix * computedNormal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
```

### TSL (WebGPU)

```javascript
import { texture, positionLocal, normalLocal, uniform, vec2, vec3,
         float as tslFloat, attribute } from 'three/tsl';

function createDisplacementMaterial(heightmapTex, maxHeight = 50) {
  const material = new MeshStandardNodeMaterial();
  const uv = attribute('uv');
  const h = texture(heightmapTex, uv).r.mul(tslFloat(maxHeight));

  // Displace Y
  material.positionNode = positionLocal.add(vec3(0, h, 0));

  // Compute normal from neighbors
  const texel = tslFloat(1.0 / 512.0);
  const hL = texture(heightmapTex, uv.add(vec2(texel.negate(), 0))).r.mul(tslFloat(maxHeight));
  const hR = texture(heightmapTex, uv.add(vec2(texel, 0))).r.mul(tslFloat(maxHeight));
  const hD = texture(heightmapTex, uv.add(vec2(0, texel.negate()))).r.mul(tslFloat(maxHeight));
  const hU = texture(heightmapTex, uv.add(vec2(0, texel))).r.mul(tslFloat(maxHeight));
  material.normalNode = normalize(vec3(hL.sub(hR), tslFloat(2), hD.sub(hU)));

  return material;
}
```

## Usage Tips

- **Dispatch workgroup sizes**: Use 16×16 for 2D texture ops, 64 or 256 for 1D arrays. Ensure dispatch count covers full resolution: `Math.ceil(resolution / 16)`.
- **Storage texture formats**: `r32float` for heightmaps, `rgba8snorm` for normals, `rgba8unorm` for color maps.
- **Read-back**: To read GPU heightmap data back to CPU (for collision), use `GPUBuffer` with `MAP_READ` usage and `copyTextureToBuffer`.
- **Erosion iterations**: Run 50K–200K droplets for visible results on 512×512. Multiple passes with `storageBarrier()` between them.
