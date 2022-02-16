# my plots
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import os



def sliced_quiver(mesh, f,  id, t, path, z=0.0, tol= 1e-10, scale = "auto"):
    plt.close("all")
    os.makedirs(path+"/plots",exist_ok=True)
    #-
    vertices = mesh.coordinates()
    slice_vertices_x = []
    slice_vertices_y = []
    X = []
    Y = []
    colours = []
    test_list = []
    for x in vertices:
        test_list.append(x[2])
        test_list = list(set(test_list))
        if (x[2]-z)**2 <= tol:
            slice_vertices_x.append(x[0])
            slice_vertices_y.append(x[1])
            tmp = f(x)
            X.append(tmp[0])
            Y.append(tmp[1])
            colours.append(np.abs(tmp[2]))
    plt.clf()
    if colours == []:
        # this means there are no vertices in that plane
        raise Exception(" Error: no vertices in the plane z="+str(z))
    else:
        if scale == "auto":
            cnorm = mpl.colors.Normalize(vmin=0.0,vmax=np.max(colours))
        elif scale == "unit":
            cnorm = mpl.colors.Normalize(vmin=0.0,vmax=1.0)
        cmap = mpl.cm.winter
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=cnorm)
        plt.quiver(slice_vertices_x, slice_vertices_y, X,Y, color= cmap(colours))
        plt.colorbar(sm)
        plt.savefig(path+"/plots/"+id+"_"+str(t).replace(".","-")+".png", dpi=800)

def quiver_2d(mesh, f, id, t, path, scale="auto"):
    plt.close("all")
    os.makedirs(path+"/plots",exist_ok=True)
    #-
    vertices = mesh.coordinates()
    slice_vertices_x = []
    slice_vertices_y = []
    X = []
    Y = []
    colours = []
    for x in vertices:
        slice_vertices_x.append(x[0])
        slice_vertices_y.append(x[1])
        tmp = f(x)
        X.append(tmp[0])
        Y.append(tmp[1])  
        colours.append(np.sqrt(tmp[0]**2 + tmp[1]**2))
    plt.clf()
    if scale == "auto":
        cnorm = mpl.colors.Normalize(vmin=0.0,vmax=np.max(colours))
    elif scale == "unit":
        cnorm = mpl.colors.Normalize(vmin=0.0,vmax=1.0)
    cmap = mpl.cm.winter_r
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=cnorm)
    
    plt.quiver(slice_vertices_x, slice_vertices_y, X,Y, color= cmap(colours))
    plt.colorbar(sm)
    plt.savefig(path+"/plots/"+id+"_"+str(t).replace(".","-")+".png", dpi=800)

def energy_plot(Energy_data, path):
    plt.close("all")
    os.makedirs(path+"/plots",exist_ok=True)
    plt.clf()
    plt.ylabel("energy")
    plt.xlabel("time")
    plt.plot(Energy_data[0], Energy_data[1])
    plt.savefig(path+"/plots/energy_plot.png", dpi=800)
