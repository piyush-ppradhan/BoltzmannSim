JLBM is a high-performance, Lattice Boltzmann Method (LBM)-based simulation library using [JAX](https://github.com/google/jax). The library is designed to be an extension to [XLB](https://github.com/Autodesk/XLB).

#### Features
- **Integration with JAX Ecosystem:** The library can be easily integrated with JAX's robust ecosystem of machine learning libraries such as [Flax](https://github.com/google/flax), [Haiku](https://github.com/deepmind/dm-haiku), [Optax](https://github.com/deepmind/optax), and many more.
- **Differentiable LBM Kernels:** XLB provides differentiable LBM kernels that can be used in differentiable physics and deep learning applications.
- **Scalability:** XLB is capable of scaling on distributed multi-GPU systems, enabling the execution of large-scale simulations on hundreds of GPUs with billions of cells.
- **Support for Various LBM Boundary Conditions and Kernels:** XLB supports several LBM boundary conditions and collision kernels.
- **User-Friendly Interface:** Written entirely in Python, XLB emphasizes a highly accessible interface that allows users to extend the library with ease and quickly set up and run new simulations.
- **Leverages JAX Array and Shardmap:** The library incorporates the new JAX array unified array type and JAX shardmap, providing users with a numpy-like interface. This allows users to focus solely on the semantics, leaving performance optimizations to the compiler.
- **Platform Versatility:** The same XLB code can be executed on a variety of platforms including multi-core CPUs, single or multi-GPU systems, TPUs, and it also supports distributed runs on multi-GPU systems or TPU Pod slices.
- **Visualization:** XLB provides a variety of visualization options including in-situ on GPU rendering using [PhantomGaze](https://github.com/loliverhennigh/PhantomGaze).

#### Supported Problems
Currently JLBM can be used to model the following problems:
1. Isothermal flows:
    - **Single Phase** are modelled using the Bhatnagar-Gross-Krook (BGK) collision model.
    - **Multiphase Flows** are modelled using the BGK model, while the interaction forces between different phases is performed using the Shan-Chen model. The computation also incorporates Equation of State (EOS) to model high-density ratio collision fluids.
2. Non-isothermal flows:
