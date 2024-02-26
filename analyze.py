import argparse
import os
import meshio
from utils import parse_xdmf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("--show", action="store_true", help="Show")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Tfile = os.path.join(args.folder, "T.xdmf")
    ufile = os.path.join(args.folder, "u.xdmf")

    dsets, topology_address, geometry_address = parse_xdmf(Tfile, get_mesh_address=True)
    dsets = dict(dsets)

    dsets_u = parse_xdmf(ufile, get_mesh_address=False)
    dsets_u = dict(dsets_u)

    with h5py.File(topology_address[0], "r") as h5f:
        elems = h5f[topology_address[1]][:]
    
    with h5py.File(geometry_address[0], "r") as h5f:
        nodes = h5f[geometry_address[1]][:]
    
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elems)

    t_ = np.array(sorted(dsets.keys()))
    it_ = list(range(len(t_)))

    T_ = np.zeros_like(nodes[:, 0])
    u_ = np.zeros((len(elems), 2))

    levels = np.linspace(0, 1, 11)

    xmax = dict([(level, np.zeros_like(t_)) for level in levels])
    xmin = dict([(level, np.zeros_like(t_)) for level in levels])
    umax = np.zeros_like(t_)

    for it in it_:
        t = t_[it]
        dset = dsets[t]
        with h5py.File(dset[0], "r") as h5f:
            T_[:] = h5f[dset[1]][:, 0]

        with h5py.File(dsets_u[t][0], "r") as h5f:
            u_[:, :] = h5f[dsets_u[t][1]][:, :2]

        fig, ax = plt.subplots(1, 1)
        ax.tripcolor(triang, T_)
        cs = ax.tricontour(triang, T_, colors="k", levels=levels)
        ax.set_aspect("equal")
        fig.set_tight_layout(True)
        if args.show:
            plt.show()
        plt.close(fig)

        if args.show:
            fig, ax = plt.subplots(1, 2)
            ax[0].tripcolor(triang, u_[:, 0])
            ax[1].tripcolor(triang, u_[:, 1])
            plt.show()


        paths = []
        for level, path in zip(cs.levels, cs.get_paths()):
            if len(path.vertices):
                paths.append((level, path.vertices))
        paths = dict(paths)

        for level, verts in paths.items():
            xmax[level][it] = verts[:, 0].max()
            xmin[level][it] = verts[:, 0].min()

        umax[it] = np.linalg.norm(u_, axis=0).max()

    fig, ax = plt.subplots(1, 4)
    for level in levels[1:-1]:
        ax[0].plot(t_, xmax[level], label=f"$T={level:1.2f}$")
        ax[1].plot(t_, xmin[level])
        ax[2].plot(t_, xmax[level]-xmin[level])

    ax[0].legend()
    [axi.set_xlabel("$t$") for axi in ax]
    ax[0].set_ylabel("$x_{max}$")
    ax[1].set_ylabel("$x_{min}$")
    ax[2].set_ylabel("$x_{max}-x_{min}$")
    ax[2].semilogy()
    ax[3].plot(t_, umax)
    ax[3].set_ylabel("$u_{max}$")

    plt.show()