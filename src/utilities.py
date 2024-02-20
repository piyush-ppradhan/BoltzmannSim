# System libraries
import os
from time import time
from functools import partial
import __main__

# Custom libraries
from conversion_parameters import *

# Third-party libraries
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pyvista as pv
from termcolor import colored
import trimesh

# JAX specific libraries
import jax.numpy as jnp
from jax.image import resize
from jax import jit

def read_raw_file(filename,width,height,depth=0,data_type=np.uint8):
  """
  Read the given raw file and store it in a numpy array.

  Arguments:
    Filename: str
      Name of the file. Add location if the file is not in the same directory.
    width: int
      Image width
    height: int
      Image height 
    depth: int
      Image depth (only for 3D). Default: 0
    data_type: type
      Data type used for converting the binary values to decimal values. Default: np.uint8
  
  Returns:
    mask: array
      numpy array of dimension (nx,ny) in 2D or (nx,ny,nz) in 3D storing the solid and fluid nodes in the domain
  """
  with open(filename, 'rb') as file:
      raw_data = np.frombuffer(file.read(),dtype=data_type)
  
  if depth == 0:
    mask = raw_data.reshape((width,height))
  else:
    mask = raw_data.reshape((width,height,depth))
  return mask

def save_image(timestep, fld, prefix=None):
    """
    Save an image of a field at a given timestep.

    Parameters
    ----------
    timestep : int
        The timestep at which the field is being saved.
    fld : jax.numpy.ndarray
        The field to be saved. This should be a 2D or 3D JAX array. If the field is 3D, the magnitude of the field will be calculated and saved.
    prefix : str, optional
        A prefix to be added to the filename. The filename will be the name of the main script file by default.

    Returns
    -------
    None

    Notes
    -----
    This function saves the field as an image in the PNG format. The filename is based on the name of the main script file, the provided prefix, and the timestep number.
    If the field is 3D, the magnitude of the field is calculated and saved. The image is saved with the 'nipy_spectral' colormap and the origin set to 'lower'.
    """
    fname = os.path.basename(__main__.__file__)
    fname = os.path.splitext(fname)[0]
    if prefix is not None:
        fname = prefix + fname
    fname = fname + "_" + str(timestep).zfill(4)

    if len(fld.shape) > 3:
        raise ValueError("The input field should be 2D!")
    elif len(fld.shape) == 3:
        fld = np.sqrt(fld[..., 0] ** 2 + fld[..., 1] ** 2)

    plt.clf()
    plt.imsave(fname + '.png', fld.T, cmap=cm.nipy_spectral, origin='lower')

def read_tiff_file(filename):
  pass

def write_vtk(output_dir, prefix, time_step, fields, conv_param):
  """
  Write the macroscopic flow variables in a VTK file. The mesh is assumed to be structured and hence no coordinate information needs to be passed.

  Arguments:
    filename_prefix: str
      Filename prefix used for the VTK file.
    timestep: int
      Timestep of VTK export
    rho: array[float]
      Numpy data that stores the density at the grid points. Dimension: (nx,ny,nz) for 3D and (nx,ny) for 2D.
    conv_param: ConversionParameters
      Conversion parameters defined to convert the values in Lattice Units to SI units.
    lattice: Lattice
      Lattice used in the simulation. Used for accessing c_s2 attribute
    write_precision: type
      Precision of the output data.
  Returns:
    None
  """
  if not os.path.exists("./"+output_dir):
    print(colored("Directory does not exist, creating the directory " + output_dir))
    os.makedirs(output_dir)

  for key, val in fields.items():
    if key == list(fields.keys())[0]:
      dimensions = val.shape
    else:
       assert val.shape == dimensions, "All fields must have the same dimensions!"
    
  dimensions = tuple([dim + 1 for dim in dimensions])

  if val.ndim == 2:
    dimensions += (1,)
      
  grid = pv.ImageData(dimensions=dimensions)
  grid.origin = (0.0, 0.0, 0.0)
  # Scaling the lattice grid points using conversion parameters
  grid.spacing = (conv_param.C_l,conv_param.C_l,conv_param.C_l) 

  # Transfer the arrays to the memory from the host
  for key, val in fields.items():
    if key == "rho":
       grid[key] = conv_param.to_physical_units(val.flatten(order='F'), "density")
    elif key == "ux" or key == "uy" or key == "uz":
       grid[key] = conv_param.to_physical_units(val.flatten(order='F'), "velocity")
  
  filename = os.path.join(output_dir, prefix+"_"+f"{time_step:07}.vtk")
  grid.save(filename, binary=True)
@partial(jit, static_argnums=(1, 2))
def downsample_field(field, factor, method='bicubic'):
  """
  Downsample a JAX array by a factor of `factor` along each axis.

  Parameters:
    field : jax.numpy.ndarray
        The input vector field to be downsampled. This should be a 3D or 4D JAX array where the last dimension is 2 or 3 (vector components).
    factor : int
        The factor by which to downsample the field. The dimensions of the field will be divided by this factor.
    method : str, optional
        The method to use for downsampling. Default is 'bicubic'.

  Returns:
    jax.numpy.ndarray
        The downsampled field.
  """
  if factor == 1:
      return field
  else:
      new_shape = tuple(dim // factor for dim in field.shape[:-1])
      downsampled_components = []
      for i in range(field.shape[-1]):  # Iterate over the last dimension (vector components)
          resized = resize(field[..., i], new_shape, method=method)
          downsampled_components.append(resized)
      return jnp.stack(downsampled_components, axis=-1)