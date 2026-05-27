CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -Wall

swe: swe.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: swe
	OMP_NUM_THREADS=$${T:-4} ./swe 512 512 1.5 6

clean:
	rm -f swe frame_*.pgm
