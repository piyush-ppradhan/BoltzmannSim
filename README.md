# BoltzmannSim: High-Performance Lattice Boltzmann Simulation

BoltzmannSim is a high-performance 2D/3D Lattice Boltzmann Method (LBM) library that can be run on accelerated hardware (GPU, TPU). It is written using the [JAX](https://github.com/google/jax) library.

## Additional Features
1. Single Phase Flows
2. Multiphase Flows
3. Chemical Reactions 
4. Direct scaling between lattice and physical units
5. Generalized approach for applying external forces

## Installation Guide

To use JLBM, you must first install JAX and other dependencies using the following commands:

Please refer to https://github.com/google/jax for the latest installation documentation. The following table is taken from [JAX's Github page](https://github.com/google/jax).

| Hardware   | Instructions                                                                                                    |
|------------|-----------------------------------------------------------------------------------------------------------------|
| CPU        | `pip install -U "jax[cpu]"`                                                                                       |
| NVIDIA GPU on x86_64 | `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`        |
| Google TPU | `pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`                 |
| AMD GPU    | Use [Docker](https://hub.docker.com/r/rocm/jax) or [build from source](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-a-rocm-jaxlib-for-amd-gpus). |
| Apple GPU  | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |


Install dependencies:
```bash
pip install pyvista numpy matplotlib Rtree trimesh jmp orbax-checkpoint termcolor
```

Run an example:
```bash
git clone https://github.com/Autodesk/XLB
cd XLB
export PYTHONPATH=.
python3 examples/CFD/cavity2d.py
```