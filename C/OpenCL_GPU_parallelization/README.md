# OpenCL GPU parallelization
Program that renders satellites orbiting around a blackhole

Optimized and parallelized as a part of parallelization course and the program went from average of 2.7FPS to 167FPS, so a 62x increase in performance

The optimizations come from using SIMD for smaller loops, optimizing data access and cache loading patterns, avoiding recomputations and using OpenCL for GPU computation of large paralellizable parts 

The project depends on SDL2 and requires OpenCL headers & OpenCL.lib statically linked library to compile

The default compilation is also set to use `AVX2`

Unoptized program: 370ms per frame
<div align="center">
  <img src="media/satellites_unoptimal.webp">
</div>


Optimized program: 6ms per frame
<div align="center">
  <img src="media/satellites_optimized.webp">
</div>
