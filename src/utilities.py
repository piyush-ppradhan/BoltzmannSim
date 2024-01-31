import numpy as np
import pyvista as pv
from conversion_parameters import *

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
  
  mask = raw_data.reshape((depth,width,height))
  return mask

#TODO
def read_tiff_file(filename):
  pass

def write_vtk(filename_prefix,time_step,rho,u,conv_param,lattice,precision):
  """
    Write the macroscopic flow variables in a VTK file. The mesh is assumed to be structured and hence no coordinate information needs to be passed.

    Arguments:
      filename_prefix: str
        Filename prefix used for the VTK file.
      timestep: int
        Timestep of VTK export
      p: array[float]
        Numpy data that stores the pressure at the grid points. Dimension: (nx,ny,nz) for 3D and (nx,ny) for 2D.
      conv_param: ConversionParameters
        Conversion parameters defined to convert the values in Lattice Units to SI units.
      lattice: Lattice
        Lattice used in the simulation. Used for accessing c_s2 attribute
      precision: type
        Precision of the output data. This is the same as the precision of computation.

    Returns:
      None
  """
  grid = pv.ImageData()
  grid.dimension = np.array(np.shape(rho)[0:-1]) + 1
  grid.origin = (0.0, 0.0, 0.0)
  # Scaling the lattice grid points using conversion parameters
  grid.spacing = (conv_param.C_l,conv_param.C_l,conv_param.C_l) 

  # Transfer the arrays to the memory from the host
  p_cpu = jnp.asnumpy(p)
  u_cpu = jnp.asnumpy(u)

  grid["p"] = p_cpu.flatten(order='F')
  if lattice.d == 2:
    grid["ux"] = u_cpu[:,:,0].flatten(order='F')
    grid["uy"] = u_cpu[:,:,1].flatten(order='F')
  else:
    grid["ux"] = u_cpu[:,:,:,0].flatten(order='F')
    grid["uy"] = u_cpu[:,:,:,1].flatten(order='F')

  grid.save(filename_prefix+time_step+".vtk")