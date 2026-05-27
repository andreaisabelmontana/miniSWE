# miniSWE

2D shallow-water solver. Lax-Friedrichs finite differences, periodic BCs, OpenMP threaded.
Outputs PGM frames of the height field, prints Mcell-updates/s and a mass-conservation check.

## Equations

State `U = [h, hu, hv]`, gravity `g = 9.81`. Conservative form:

```
∂h /∂t + ∂(hu)/∂x        + ∂(hv)/∂y        = 0
∂hu/∂t + ∂(hu² + ½gh²)/∂x + ∂(huv)/∂y       = 0
∂hv/∂t + ∂(huv)/∂x        + ∂(hv² + ½gh²)/∂y = 0
```

Lax-Friedrichs update:
`U_{i,j}^{n+1} = ¼(U_E+U_W+U_N+U_S) − Δt/(2Δx)(F_E−F_W) − Δt/(2Δy)(G_N−G_S)`.
CFL adaptive: `Δt = CFL · min(Δx,Δy) / max(|u|+c, |v|+c)`, `c = √(gh)`.

## Build & run

```sh
make
OMP_NUM_THREADS=8 ./swe 1024 1024 2.0 8   # nx ny t_final n_frames
```

## Strong-scaling sweep

```sh
./scale.sh                       # default 1024² at 1,2,4,8 threads
NX=2048 NY=2048 ./scale.sh       # bigger
```

## Animate the height field

```sh
pip install pillow
python3 make_gif.py              # writes wave.gif from frame_*.pgm
```

## Sanity check

The final line prints `mass: init=… final=… rel-drift=…`.
Periodic BCs + conservative form → drift should be ~1e-14 (round-off).
