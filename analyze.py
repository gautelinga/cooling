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
    pfile = os.path.join(args.folder, "p.xdmf")

    dsets, topology_address, geometry_address = parse_xdmf(Tfile, get_mesh_address=True)
    dsets = dict(dsets)

    dsets_u = parse_xdmf(ufile, get_mesh_address=False)
    dsets_u = dict(dsets_u)

    dsets_p = parse_xdmf(pfile, get_mesh_address=False)
    dsets_p = dict(dsets_p)

    with h5py.File(topology_address[0], "r") as h5f:
        elems = h5f[topology_address[1]][:]
    
    with h5py.File(geometry_address[0], "r") as h5f:
        nodes = h5f[geometry_address[1]][:]
    
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elems)

    t_ = np.array(sorted(dsets.keys()))
    it_ = list(range(len(t_)))

    T_ = np.zeros_like(nodes[:, 0])
    p_ = np.zeros_like(T_)
    u_ = np.zeros((len(elems), 2))

    levels = np.linspace(0, 1, 11)

    xmax = dict([(level, np.zeros_like(t_)) for level in levels])
    xmin = dict([(level, np.zeros_like(t_)) for level in levels])
    umax = np.zeros_like(t_)

    if True:
        beta = 0.001

        t = t_[it_[-1]]
        dset = dsets[t]
        with h5py.File(dset[0], "r") as h5f:
            T_[:] = h5f[dset[1]][:, 0]

        with h5py.File(dsets_p[t][0], "r") as h5f:
            p_[:] = h5f[dsets_p[t][1]][:, 0]

        T_intp = tri.CubicTriInterpolator(triang, T_)
        p_intp = tri.CubicTriInterpolator(triang, p_)

        Nx = 30
        Ny = 100

        x = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), Nx)
        y = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), Ny)

        X, Y = np.meshgrid(x, y)

        T_vals_ = T_intp(X, Y)
        p_vals_ = p_intp(X, Y)
        px_vals_, py_vals_ = p_intp.gradient(X, Y)

        ux_vals_ = -beta**-T_vals_ * px_vals_

        T_max_ = T_vals_.max(axis=0)
        ux_max_ = ux_vals_.max(axis=0)

        fig, ax = plt.subplots(1, 2)
        for i in range(Nx)[::3]:
            ax[0].plot(y, T_vals_[:, i], label=f"$x={x[i]:1.2f}$")
            ax[1].plot(y, ux_vals_[:, i])
        ax[0].set_ylabel("$T$")
        ax[1].set_ylabel("$u_x$")
        ax[0].legend()
        [axi.set_xlabel("$y$") for axi in ax]

        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].plot(x, T_max_)
        ax2[1].plot(x, ux_max_)
        ax2[0].set_ylabel("$T_{max}(x)$")
        ax2[1].set_ylabel("$u_{x,max}(x)$")
        [axi.set_xlabel("$x$") for axi in ax2]
        plt.show()

    for it in it_:
        t = t_[it]

        print(f"it={it} t={t}")

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