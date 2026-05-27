#!/usr/bin/env bash
# Strong-scaling sweep: same problem size, vary thread count.
set -e
NX=${NX:-1024}
NY=${NY:-1024}
T=${T:-1.5}
THREADS=${THREADS:-"1 2 4 8"}

make -s swe

printf "threads  wall(s)  Mcell/s  speedup\n"
base=""
for t in $THREADS; do
    out=$(OMP_NUM_THREADS=$t ./swe $NX $NY $T 1 | grep '^steps=')
    wall=$(echo "$out" | sed -E 's/.*wall=([0-9.]+)s.*/\1/')
    mcells=$(echo "$out" | sed -E 's/.*Mcell-updates\/s=([0-9.]+).*/\1/')
    if [ -z "$base" ]; then base=$wall; fi
    speedup=$(awk "BEGIN{printf \"%.2f\", $base / $wall}")
    printf "%7d  %7.3f  %7.2f  %6sx\n" $t $wall $mcells $speedup
done
