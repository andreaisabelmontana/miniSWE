# miniSWE
**2D Shallow Water Equation Solver**

High-performance parallel computing implementation of the 2D shallow water equations using Lax-Friedrichs finite differences with OpenMP threading.

---

## 📋 Overview

**miniSWE** solves the 2D shallow water equations (SWE) — a fundamental system in computational fluid dynamics. The implementation emphasizes:

- ⚡ **High Performance**: OpenMP parallelization for multi-core systems
- 🎯 **Accuracy**: Conservative finite difference scheme
- 🔍 **Verification**: Mass conservation checking
- 📊 **Visualization**: PGM frame export for animation

**Course:** High Performance Computing (IE University BCSAI)  
**Language:** C++  
**Parallelization:** OpenMP  
**Performance Target:** 1000+ Mcell-updates/second

---

## 🌊 The Physics

### State Variables

State vector: `U = [h, hu, hv]`

| Variable | Meaning |
|----------|---------|
| `h` | Water height (depth) |
| `hu` | Horizontal momentum |
| `hv` | Vertical momentum |

### Conservation Laws

```
∂h /∂t + ∂(hu)/∂x        + ∂(hv)/∂y        = 0                    [Mass]
∂hu/∂t + ∂(hu² + ½gh²)/∂x + ∂(huv)/∂y       = 0                    [X-momentum]
∂hv/∂t + ∂(huv)/∂x        + ∂(hv² + ½gh²)/∂y = 0                    [Y-momentum]
```

### Parameters

- Gravity: `g = 9.81 m/s²`
- Wave speed: `c = √(gh)`

---

## 🔢 Numerical Scheme

### Lax-Friedrichs Method

Explicit finite difference update:

```
U_{i,j}^{n+1} = ¼(U_E + U_W + U_N + U_S) 
               − Δt/(2Δx)(F_E − F_W) 
               − Δt/(2Δy)(G_N − G_S)
```

Where:
- `U_E, U_W, U_N, U_S`: States at East, West, North, South neighbors
- `F, G`: Flux functions (momentum + pressure terms)

### Adaptive Time Stepping

CFL (Courant-Friedrichs-Lewy) condition ensures stability:

```
Δt = CFL · min(Δx, Δy) / max(|u| + c, |v| + c)
```

Where CFL ≈ 0.9 for stability.

### Boundary Conditions

Periodic boundary conditions (wrap-around):
- Left connects to right
- Top connects to bottom
- Simulates continuous periodic domain

---

## 🏗️ Build & Compilation

### Prerequisites

- C++ compiler with OpenMP support (`g++`, `clang`, `icc`)
- `make` build system
- `libpng` (optional, for visualization)

### Build

```bash
make                    # Compile with optimizations
make clean              # Remove build artifacts
```

The Makefile enables:
- `-O3` optimization
- `-fopenmp` OpenMP support
- `-march=native` CPU-specific optimizations

---

## 🚀 Running the Solver

### Basic Usage

```bash
./swe nx ny t_final n_frames
```

**Arguments:**
- `nx, ny`: Grid resolution (e.g., 1024 for 1024×1024)
- `t_final`: Total simulation time (seconds)
- `n_frames`: Number of output frames

### Example

```bash
# Small test
./swe 512 512 1.0 4

# Production run (8 threads)
OMP_NUM_THREADS=8 ./swe 1024 1024 2.0 8

# Large simulation
OMP_NUM_THREADS=16 ./swe 2048 2048 5.0 20
```

### Output

```
Computing 2048x2048 grid for 2.0s in 8 frames...
Frame 1: t=0.25s, Mcells/s=1234.5, mass_drift=2.1e-14
Frame 2: t=0.50s, Mcells/s=1245.3, mass_drift=1.9e-14
...
Total time: 12.34s
```

---

## 📊 Performance Analysis

### Throughput Metric

**Mcell-updates/second** = (nx × ny × n_timesteps) / wall_time / 1e6

Typical performance:
- **Single-core**: 200-400 Mcells/s
- **4-core**: 700-1200 Mcells/s  
- **8-core**: 1400-2000 Mcells/s
- **16-core**: 2500-3500 Mcells/s

### Scaling Analysis

Run strong-scaling benchmark:

```bash
./scale.sh                      # Default: 1024² at 1,2,4,8 threads
NX=2048 NY=2048 ./scale.sh      # Larger domain
```

This executes the same problem on different thread counts and plots efficiency.

---

## 🎨 Visualization

### Generate Frames

PGM (Portable Graymap) files are output for each frame:

```
frame_001.pgm   [1024×1024 uint8 image of height field]
frame_002.pgm
...
frame_008.pgm
```

### Create Animation

```bash
pip install pillow
python3 make_gif.py        # Converts frame_*.pgm → wave.gif
```

This produces `wave.gif` showing the wave evolution.

---

## ✅ Verification

### Mass Conservation Check

The final line prints:

```
mass: init=1000.0 final=1000.00000000001 rel-drift=1.2e-14
```

**Expected behavior:**
- ✅ Periodic BCs + conservative form → drift ≈ 1e-14 (machine epsilon)
- ⚠️ Drift > 1e-10 suggests numerical issues

### Trust Metrics

- `rel-drift < 1e-12`: Excellent
- `1e-12 < rel-drift < 1e-10`: Good
- `rel-drift > 1e-10`: Investigate stability

---

## 🔧 Customization

### Initial Conditions

Edit `swe.cpp` `initialize()` function to set different initial water profiles:

```cpp
// Bump in center
h[i*ny + j] = 1.0 + 0.1 * exp(-((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)) / 0.01);

// Wave
h[i*ny + j] = 1.0 + 0.1 * sin(2*M_PI*x) * cos(2*M_PI*y);

// Dam break
h[i*ny + j] = (x < 0.5) ? 2.0 : 1.0;
```

### Numerical Parameters

In `swe.cpp`:

```cpp
const double CFL = 0.9;      // Time step control (↑ faster, ↑ unstable)
const double GRAVITY = 9.81; // Earth's gravity
```

### Grid Resolution

Command-line arguments control resolution at runtime.

---

## 📈 Performance Tips

### Maximize Performance

1. **Use native CPU flags**: Already in Makefile
2. **Increase thread count**: `OMP_NUM_THREADS=16`
3. **Larger grid size**: Better load balancing (> 1024²)
4. **Longer simulation**: Amortizes I/O overhead
5. **Profile**: Use `perf` or `gprof` to identify bottlenecks

### Memory Usage

Memory per grid point: ~32 bytes (3 floats × 4 states)

```
1024²:  ~130 MB
2048²:  ~500 MB
4096²:  ~2 GB
```

---

## 🐛 Troubleshooting

### Compilation Issues

```bash
# Missing OpenMP?
g++ -fopenmp -O3 -o swe swe.cpp

# Intel compiler
icc -qopenmp -O3 -o swe swe.cpp
```

### Runtime Instability

- ↑ CFL → ↑ timestep → instability
- ↓ grid resolution with same time → instability
- Try: `CFL = 0.8` or reduce `t_final`

### Slow Performance

- Fewer threads than cores? Check `OMP_NUM_THREADS`
- Profiling: `perf stat ./swe 1024 1024 1.0 4`
- Use release mode: `-O3`

---

## 📚 References

- Godunov, S.K. (1959): "Difference scheme for numerical computation of discontinuous solutions"
- Lax, P.D., Wendroff, B. (1960): "Systems of conservation laws"
- OpenMP Official Docs: https://www.openmp.org/
- ShallowWater SWE Community: [Codes](https://github.com/topics/shallow-water)

---

## 🎓 Related Topics

- Finite Difference Methods
- Conservative Schemes
- OpenMP Parallelization
- CFD (Computational Fluid Dynamics)
- Numerical Stability Analysis
- High-Performance Computing

---

**Course:** High Performance Computing | **School:** IE University BCSAI  
**Type:** Educational HPC Project | **Language:** C++  
**Key Topics:** Parallelism · Numerical Methods · Physics Simulation
