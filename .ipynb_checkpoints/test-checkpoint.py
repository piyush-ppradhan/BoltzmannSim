"""
import meshio
import numpy as np

# two triangles and one quad
points = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [2.0, 1.0],
]
cells = [
    ("triangle", [[0, 1, 2], [1, 3, 2]]),
    ("quad", [[1, 4, 5, 3]]),
]

data = np.asarray([1.0,2.0,3.0,4.0,5.0,6.0])

with meshio.xdmf.TimeSeriesWriter("foo.xdmf") as writer:
    writer.write_points_cells(points, cells)
    for t in [0.0, 0.1, 0.21]:
        writer.write_data(t, point_data={"phi": (t + 1.0)*data})
"""

import jax.numpy as jnp
import timeit
from jax import jit, random

def selu(x, alpha=1.67, lmbda=0.5):
    return lmbda * alpha * x - alpha

key = random.PRNGKey(0)
x = random.normal(key, (1000,))
selu_jit = jit(selu)

#selu_jit = jit(selu)
#timeit.timeit(lambda: selu_jit(x).block_until_ready())
