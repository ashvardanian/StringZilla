// Fingerprint device TU: emits the single explicit `szs_fingerprints_warp` entry kernel into ONE PTX
// blob. Compiled with `nvcc -ptx`; the rolling hash uses only f64 FMA / fmod / floor so a low compute
// floor (compute_50) suffices and forward-JITs to all newer GPUs.
#include "stringzillas/fingerprints/cuda_device.cuh"
