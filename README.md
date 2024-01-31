# JLBM: High-Performance Lattice Boltzmann Simulation

JLBM is a high-performance 2D/3D Lattice Boltzmann Method (LBM) library that can be run on accelerated hardware(GPU, TPU). It is written using the [JAX](https://github.com/google/jax) library.

## Features
    1. Single Phase Flows
    2. Multiphase Flows (Shan-Chen model)
    3. Chemical Reactions 

## Dependencies
    1. jax
    2. pyvista (Exporting the flow data).
    3. numpy
    4. functools

    To install the dependencies, use the following command: `pip install pyvista numpy functools`