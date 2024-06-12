import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpi4py import MPI
mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()

if mpi_size > 1:
    if mpi_rank == 0:
        print("This script only works in serial. You are better off  \n"
              "simply running the parameter scan in parallel instead.")
    # exit()

def mpi_print(*args):
    if mpi_rank == 0:
        print(*args)

def mpi_max(a):
    local_max = np.max(a)
    global_max = np.zeros(1, dtype=float)
    MPI.COMM_WORLD.Allreduce(local_max, global_max, op=MPI.MAX)
    return global_max[0]

def mpi_stitch(a):
    stitched = MPI.COMM_WORLD.gather(a)
    if MPI.COMM_WORLD.rank == 0:
        return np.concatenate(stitched)

class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < df.DOLFIN_EPS_LARGE

class Right(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > df.DOLFIN_EPS_LARGE

def parse_args():
    parser = argparse.ArgumentParser(description="Solve the linearised model")
    parser.add_argument("-Pe", default=100.0, type=float, help="Peclet number")
    parser.add_argument("-lam", default=4.0, type=float, help="Wavelength")
    parser.add_argument("-Gamma", default=1.0, type=float, help="Heat conductivity")
    parser.add_argument("-beta", default=0.001, type=float, help="Viscosity ratio")
    parser.add_argument("-eps", default=1e-1, type=float, help="Perturbation amplide")
    parser.add_argument("-tpert", default=0.1, type=float, help="Perturbation duration")
    parser.add_argument("-dt", default=0.01, type=float, help="Timestep")
    parser.add_argument("-nx", default=1000, type=int, help="Number of mesh points")
    parser.add_argument("-Lx", default=50.0, type=float, help="System size")
    parser.add_argument("-tmax", default=10.0, type=float, help="Total time")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dt = args.dt
    nx = args.nx
    Lx = args.Lx
    tmax = args.tmax # 10.0

    Pe = args.Pe # 10**0.75 #100.0
    lam = args.lam # 2.0 # 4.0
    Gamma = args.Gamma #1.0
    beta = args.beta # 0.001
    eps = args.eps # 1e-1
    tpert = args.tpert # 0.1

    plot_intv = 100

    kappa = 1/Pe
    kappa_par = 2.0 / 105 * Pe
    k = 2*np.pi/lam

    kappa_eff = kappa + kappa_par
    psi = -np.log(beta)
    xi = (- 1 + np.sqrt(1 + 4*kappa_eff*Gamma))/(2*kappa_eff)

    mpi_print(kappa_eff, xi, 1.0/xi)

    mesh = df.UnitIntervalMesh(nx)
    mesh.coordinates()[:, 0] = mesh.coordinates()[:, 0]**4
    mesh.coordinates()[:] *= Lx

    S_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_el = df.MixedElement([S_el, S_el])

    W = df.FunctionSpace(mesh, W_el)
    S = W.sub(0).collapse()

    T, p = df.TrialFunctions(W)
    U, q = df.TestFunctions(W)

    w_ = df.Function(W)
    T_, p_ = df.split(w_)

    x = df.Function(S)
    x.interpolate(df.Expression("x[0]", degree=1))

    T0 = df.Function(S)
    T0.vector()[:] = np.exp(-xi*x.vector()[:])

    betamT0 = df.Function(S)
    betamT0.vector()[:] = beta**-T0.vector()[:]

    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    subd.set_all(0)
    left = Left()
    left.mark(subd, 1)
    right = Right()
    right.mark(subd, 2)

    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)

    onoff = df.Constant(1.0)

    FT = (T - T_) / dt * U * df.dx \
        + T.dx(0) * U * df.dx \
        + kappa_eff * T.dx(0) * U.dx(0) * df.dx \
        + (kappa * k**2 + Gamma - (2*kappa_par*xi + 1) * psi * xi * T0) * T * U * df.dx \
        - betamT0 * xi * T0 * ( kappa_par * k**2 * p - (2*kappa_par*xi + 1) * p.dx(0) ) * U * df.dx

    Fp = betamT0 * p.dx(0) * q.dx(0) * df.dx \
        - onoff * eps * q * ds(1) \
        + betamT0 * k**2 * p * q * df.dx \
        + psi * T.dx(0) * q * df.dx

    F = FT + Fp

    bc_T_l = df.DirichletBC(W.sub(0), 0., subd, 1)
    bc_T_r = df.DirichletBC(W.sub(0), 0., subd, 2)
    bc_p_l = df.DirichletBC(W.sub(1), 0, subd, 2)

    bcs = [bc_T_l, bc_T_r, bc_p_l]

    a, L = df.lhs(F), df.rhs(F)

    #xdmff_T = df.XDMFFile(mesh.mpi_comm(), "T.xdmf")
    #xdmff_p = df.XDMFFile(mesh.mpi_comm(), "p.xdmf")

    plot = False

    if mpi_rank == 0:
        data = []
        fig, ax = plt.subplots(1, 2)
    
    xx = mpi_stitch(x.vector()[:])
    if mpi_rank == 0:
        idx = np.argsort(xx)
        xx = xx[idx]

    t = 0.0
    it = 0
    while t < tmax:
        mpi_print(f"t = {t}")
        it += 1

        if t > tpert:
            onoff.assign(0.)

        df.solve(a == L, w_, bcs=bcs)
        t += dt

        T__, p__ = w_.split(deepcopy=True)

        #T__.rename("T", "T")
        #p__.rename("p", "p")
        #xdmff_T.write(T__, t)
        #xdmff_p.write(p__, t)

        Tmax = mpi_max(T__.vector()[:])
        pmax = mpi_max(p__.vector()[:])
        if mpi_rank == 0:
            data.append([t, Tmax, pmax])

        if it % plot_intv == 0:
            TT = mpi_stitch(T__.vector()[:])
            pp = mpi_stitch(p__.vector()[:])
            if mpi_rank == 0:
                ax[0].plot(xx, TT[idx]/Tmax, label=f"$t={t:1.2f}$")
                ax[1].plot(xx, pp[idx]/pmax)

    #xdmff_T.close()
    #xdmff_p.close()

    #ax[0].semilogx()
    #ax[1].semilogx()
    if mpi_rank == 0:
        ax[0].set_xlabel(r"$x$")
        ax[1].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$T/T_{\rm max}$")
        ax[1].set_ylabel(r"$p/p_{\rm max}$")
        ax[0].legend()

        data = np.array(data)

        istart = int(tmax/dt)//2

        popt = np.polyfit(data[istart:, 0], np.log(data[istart:, 1]), 1)
        gamma = popt[0]

        print(f"gamma = {gamma}")

        fig_, ax_ = plt.subplots(1, 1)

        ax_.plot(data[:, 0], data[:, 1], label=r"$T_{\rm max}$")
        ax_.plot(data[:, 0], data[:, 2], label=r"$p_{\rm max}$")
        ax_.plot(data[:, 0], np.exp(gamma*data[:, 0]), label=r"fit")
        ax_.semilogy()
        ax_.set_xlabel("$t$")
        ax_.set_ylabel("scalar")
        ax_.legend()
        
        # plt.show()

        import scipy.interpolate as intp

        T0 = np.exp(-xi*xx)

        T_intp = intp.InterpolatedUnivariateSpline(xx, TT[idx]/Tmax)
        p_intp = intp.InterpolatedUnivariateSpline(xx, pp[idx]/Tmax)
        pt_intp = intp.InterpolatedUnivariateSpline(xx, beta**(-T0) * pp[idx]/Tmax)

        x = np.linspace(1e-7, Lx, 1000)
        T = T_intp(x)
        Tx = T_intp.derivative(1)(x)
        Txx = T_intp.derivative(2)(x)

        p = p_intp(x)
        px = p_intp.derivative(1)(x)
        pxx = p_intp.derivative(2)(x)

        pt = pt_intp(x)
        ptx = pt_intp.derivative(1)(x)
        ptxx = pt_intp.derivative(2)(x)

        T0 = np.exp(-xi*x)

        fig, ax = plt.subplots(9, 1)
        ax[0].plot(x, T)
        ax[1].plot(x, Tx)
        ax[2].plot(x, Txx)
        
        ax[3].plot(x, p)        
        ax[4].plot(x, px)
        ax[5].plot(x, pxx)
        
        ax[6].plot(x, pt)
        ax[6].plot(x, -psi * Tx / (xi**2 * psi * T0 + k**2), label="est")
        #ax[6].plot(x, -psi * k**-2 * Tx, label="est2")
        ax[6].legend()

        ax[7].plot(x, ptx)
        ax[8].plot(x, ptxx)

        fig, ax = plt.subplots(1, 2)

        lam = (-1 + np.sqrt(1 + 4*kappa*(gamma + kappa*k**2 + Gamma)))/(2*kappa)

        ax[0].plot(x, T)
        ax[0].plot(x, np.exp(-lam*x))
        ax[0].semilogy()

        ax[1].plot(x, p)
        ax[1].plot(x, np.exp(-lam*x))
        ax[1].plot(x, np.exp(-k*x))
        ax[1].semilogy()
        ax[1].legend()

        fig, ax = plt.subplots(1, 2)

        T_terms = [Tx,
                   -kappa_eff*Txx,
                   (gamma + kappa*k**2 + Gamma - (2*kappa_par*xi+1)*psi*xi*T0) * T,
                   -xi * T0 * (kappa_par*k**2 - (2*kappa_par*xi+1)*psi*xi*T0)*pt,
                   xi * T0 * (2*kappa_par*xi+1)*ptx]

        ax[0].plot(x, T_terms[0], label="Tx")
        ax[0].plot(x, T_terms[1], label="2")
        ax[0].plot(x, T_terms[2], label="3")
        ax[0].plot(x, T_terms[3], label="4")
        ax[0].plot(x, T_terms[4], label="5")
        ax[0].plot(x, sum(T_terms), label="sum")
        ax[0].axvline(x=xi**-1, color="k", label=r"$1/\xi$")
        ax[0].legend()

        p_terms = [ptxx, 
                   psi*xi*T0*ptx,
                   -(xi**2*psi*T0 + k**2)*pt,
                   -psi * Tx]

        ax[1].plot(x, p_terms[0], label="1")
        ax[1].plot(x, p_terms[1], label="2")
        ax[1].plot(x, p_terms[2], label="3")
        ax[1].plot(x, p_terms[3], label="4")
        ax[1].plot(x, sum(p_terms), label="sum")
        ax[1].axvline(x=xi**-1, color="k", label=r"$1/\xi$")
        ax[1].legend()

        plt.show()
        
