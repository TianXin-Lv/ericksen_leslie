"""
postprocessing class for FEM functions
To-Do:
- optional 3d Graphs
"""
from fenics import *
from my_plots import sliced_quiver, quiver_2d, energy_plot

class fem_postprocess:
    """
    class that initializes the postprocess for FEniCS functions. Allows to save 2d plots and xdmf files.
    """
    def __init__(self, dim, plots, path, id):
        """
        plots is a list of plots to be saved. Options currently are xdmf, quiver, e.g. plots= ["xdmf","quiver"]
        """
        self.dim = dim
        self.plots = plots
        self.id = id
        self.path = path
        self.quiver_scale = "auto"
        if "xdmf" in plots:
            self.xdmf_file = XDMFFile(MPI.comm_world, path+"/xdmf/"+id+".xdmf")
            self.xdmf_file.parameters["flush_output"] = True
            self.xdmf_file.parameters["functions_share_mesh"] = True
            self.xdmf_file.parameters["rewrite_function_mesh"] = False
    
    def save(self, f, mesh, t):
        for s in self.plots:
            if s=="xdmf":
                self.xdmf_file.write(f, t)
            if s=="quiver":
                if self.dim==2:
                    quiver_2d(mesh, f, self.id, t, self.path, scale = self.quiver_scale)
                elif self.dim==3:
                    sliced_quiver(mesh, f, self.id, t, self.path, scale = self.quiver_scale)
                else:
                    raise ValueError("Dimension not supported.")
            
