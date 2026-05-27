// miniSWE — 2D shallow-water solver, Lax-Friedrichs scheme, OpenMP parallel.
// Independent implementation; not derived from any existing mini-app.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>

struct Grid {
    int nx, ny;
    double dx, dy;
    std::vector<double> h, hu, hv;     // state
    std::vector<double> h2, hu2, hv2;  // scratch

    Grid(int nx_, int ny_, double Lx, double Ly)
        : nx(nx_), ny(ny_),
          dx(Lx / nx_), dy(Ly / ny_),
          h(nx_ * ny_), hu(nx_ * ny_), hv(nx_ * ny_),
          h2(nx_ * ny_), hu2(nx_ * ny_), hv2(nx_ * ny_) {}

    inline int idx(int i, int j) const { return j * nx + i; }
};

static constexpr double G_ACC = 9.81;

// Gaussian bump in the middle, calm elsewhere.
static void init_bump(Grid& g) {
    const double cx = 0.5 * g.nx * g.dx;
    const double cy = 0.5 * g.ny * g.dy;
    const double r2 = std::pow(0.08 * g.nx * g.dx, 2);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            double x = (i + 0.5) * g.dx;
            double y = (j + 0.5) * g.dy;
            double bump = std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / r2);
            g.h[g.idx(i, j)]  = 1.0 + 0.5 * bump;
            g.hu[g.idx(i, j)] = 0.0;
            g.hv[g.idx(i, j)] = 0.0;
        }
    }
}

// Periodic neighbor index.
static inline int wrap(int i, int n) { return (i + n) % n; }

// Lax-Friedrichs step: U^{n+1} = avg(neighbors) - dt/(2 dx) * (F_E - F_W) - dt/(2 dy) * (G_N - G_S).
static double step(Grid& g, double cfl) {
    const int nx = g.nx, ny = g.ny;
    const double dx = g.dx, dy = g.dy;

    // Compute max wave speed for dt.
    double maxc = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:maxc) schedule(static)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = g.idx(i, j);
            double h = g.h[k];
            double u = g.hu[k] / h;
            double v = g.hv[k] / h;
            double c = std::sqrt(G_ACC * h);
            double s = std::max(std::fabs(u), std::fabs(v)) + c;
            if (s > maxc) maxc = s;
        }
    }
    double dt = cfl * std::min(dx, dy) / maxc;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int iE = wrap(i + 1, nx), iW = wrap(i - 1, nx);
            int jN = wrap(j + 1, ny), jS = wrap(j - 1, ny);

            auto F = [&](int ii, int jj, double& f0, double& f1, double& f2) {
                int k = g.idx(ii, jj);
                double h = g.h[k], hu = g.hu[k], hv = g.hv[k];
                double u = hu / h;
                f0 = hu;
                f1 = hu * u + 0.5 * G_ACC * h * h;
                f2 = hv * u;
            };
            auto Gflux = [&](int ii, int jj, double& g0, double& g1, double& g2) {
                int k = g.idx(ii, jj);
                double h = g.h[k], hu = g.hu[k], hv = g.hv[k];
                double v = hv / h;
                g0 = hv;
                g1 = hu * v;
                g2 = hv * v + 0.5 * G_ACC * h * h;
            };

            double fE0, fE1, fE2, fW0, fW1, fW2;
            double gN0, gN1, gN2, gS0, gS1, gS2;
            F(iE, j, fE0, fE1, fE2);
            F(iW, j, fW0, fW1, fW2);
            Gflux(i, jN, gN0, gN1, gN2);
            Gflux(i, jS, gS0, gS1, gS2);

            int kE = g.idx(iE, j), kW = g.idx(iW, j);
            int kN = g.idx(i, jN), kS = g.idx(i, jS);

            double avg_h  = 0.25 * (g.h[kE]  + g.h[kW]  + g.h[kN]  + g.h[kS]);
            double avg_hu = 0.25 * (g.hu[kE] + g.hu[kW] + g.hu[kN] + g.hu[kS]);
            double avg_hv = 0.25 * (g.hv[kE] + g.hv[kW] + g.hv[kN] + g.hv[kS]);

            int k = g.idx(i, j);
            g.h2[k]  = avg_h  - 0.5 * dt / dx * (fE0 - fW0) - 0.5 * dt / dy * (gN0 - gS0);
            g.hu2[k] = avg_hu - 0.5 * dt / dx * (fE1 - fW1) - 0.5 * dt / dy * (gN1 - gS1);
            g.hv2[k] = avg_hv - 0.5 * dt / dx * (fE2 - fW2) - 0.5 * dt / dy * (gN2 - gS2);
        }
    }

    g.h.swap(g.h2);
    g.hu.swap(g.hu2);
    g.hv.swap(g.hv2);
    return dt;
}

// Dump height field as grayscale PPM (P5).
static void write_ppm(const Grid& g, const std::string& path) {
    double lo = g.h[0], hi = g.h[0];
    for (double v : g.h) { if (v < lo) lo = v; if (v > hi) hi = v; }
    double scale = (hi > lo) ? 255.0 / (hi - lo) : 0.0;

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P5\n%d %d\n255\n", g.nx, g.ny);
    std::vector<unsigned char> row(g.nx);
    for (int j = g.ny - 1; j >= 0; --j) {
        for (int i = 0; i < g.nx; ++i) {
            double v = (g.h[g.idx(i, j)] - lo) * scale;
            row[i] = (unsigned char)std::max(0.0, std::min(255.0, v));
        }
        std::fwrite(row.data(), 1, g.nx, f);
    }
    std::fclose(f);
}

int main(int argc, char** argv) {
    int    nx       = (argc > 1) ? std::atoi(argv[1]) : 512;
    int    ny       = (argc > 2) ? std::atoi(argv[2]) : 512;
    double t_final  = (argc > 3) ? std::atof(argv[3]) : 1.5;
    int    n_frames = (argc > 4) ? std::atoi(argv[4]) : 6;
    double cfl      = 0.4;

    Grid g(nx, ny, 10.0, 10.0);
    init_bump(g);

    std::printf("miniSWE  %dx%d  threads=%d  t_final=%g\n",
                nx, ny, omp_get_max_threads(), t_final);

    double cell_area = g.dx * g.dy;
    double mass0 = 0.0;
    #pragma omp parallel for reduction(+:mass0) schedule(static)
    for (int k = 0; k < nx * ny; ++k) mass0 += g.h[k] * cell_area;

    write_ppm(g, "frame_000.pgm");

    double t = 0.0;
    int step_count = 0;
    int next_frame = 1;
    double frame_dt = t_final / n_frames;

    auto t0 = std::chrono::steady_clock::now();
    while (t < t_final) {
        double dt = step(g, cfl);
        if (t + dt > t_final) dt = t_final - t;
        t += dt;
        ++step_count;

        if (t >= next_frame * frame_dt || t >= t_final) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%03d.pgm", next_frame);
            write_ppm(g, name);
            ++next_frame;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    double mass1 = 0.0;
    #pragma omp parallel for reduction(+:mass1) schedule(static)
    for (int k = 0; k < nx * ny; ++k) mass1 += g.h[k] * cell_area;

    double cells_per_sec = (double)nx * ny * step_count / secs;
    std::printf("steps=%d  wall=%.3fs  Mcell-updates/s=%.2f\n",
                step_count, secs, cells_per_sec / 1e6);
    std::printf("mass: init=%.6e  final=%.6e  rel-drift=%.2e\n",
                mass0, mass1, std::fabs(mass1 - mass0) / mass0);
    return 0;
}
