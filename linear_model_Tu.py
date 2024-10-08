import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import argparse

import scipy.interpolate as intp
import scipy.optimize as opt

from linear_model import mpi_print, mpi_max, mpi_stitch, Left, Right, mpi_rank

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

    T, u = df.TrialFunctions(W)
    U, q = df.TestFunctions(W)

    w_ = df.Function(W)
    T_, u_ = df.split(w_)

    x = df.Function(S)
    x.interpolate(df.Expression("x[0]", degree=1))

    T0 = df.Function(S)
    T0.vector()[:] = np.exp(-xi*x.vector()[:])

    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    subd.set_all(0)
    left = Left()
    left.mark(subd, 1)
    right = Right()
    right.mark(subd, 2)

    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)

    onoff = df.Constant(eps)

    FT = (T - T_) / dt * U * df.dx \
        + T.dx(0) * U * df.dx \
        + kappa_eff * T.dx(0) * U.dx(0) * df.dx \
        + (kappa * k**2 + Gamma) * T * U * df.dx \
        - xi * T0 * ( - kappa_par * u.dx(0) + (2*kappa_par*xi + 1) * u ) * U * df.dx

    Fu = k**2 * psi * T * q * df.dx \
        - k**2 * u * q * df.dx \
        + psi * xi * T0 * u.dx(0) * q * df.dx \
        - u.dx(0) * q.dx(0) * df.dx

    F = FT + Fu

    bc_T_l = df.DirichletBC(W.sub(0), 0., subd, 1)
    bc_T_r = df.DirichletBC(W.sub(0), 0., subd, 2)
    bc_u_l = df.DirichletBC(W.sub(1), onoff, subd, 1)
    bc_u_r = df.DirichletBC(W.sub(1), 0, subd, 2)

    bcs = [bc_T_l, bc_T_r, bc_u_l]

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

        T__, u__ = w_.split(deepcopy=True)

        #T__.rename("T", "T")
        #p__.rename("p", "p")
        #xdmff_T.write(T__, t)
        #xdmff_p.write(p__, t)

        Tmax = mpi_max(T__.vector()[:])
        umax = mpi_max(u__.vector()[:])
        if mpi_rank == 0:
            data.append([t, Tmax, umax])

        if it % plot_intv == 0:
            TT = mpi_stitch(T__.vector()[:])
            uu = mpi_stitch(u__.vector()[:])
            if mpi_rank == 0:
                ax[0].plot(xx, TT[idx]/Tmax, label=f"$t={t:1.2f}$")
                ax[1].plot(xx, uu[idx]/umax)

    #xdmff_T.close()
    #xdmff_p.close()

    #ax[0].semilogx()
    #ax[1].semilogx()
    if mpi_rank == 0:
        ax[0].set_xlabel(r"$x$")
        ax[1].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$T/T_{\rm max}$")
        ax[1].set_ylabel(r"$u/u_{\rm max}$")
        ax[0].axvline(x=xi**-1, color="k", label=r"$1/\xi$")
        ax[1].axvline(x=xi**-1, color="k", label=r"$1/\xi$")
        ax[0].legend()

        def fitfunc(x, a, b, c):
            return a * x / (b + np.exp(c*x))

        data_save_quick = np.vstack((xx, uu[idx]/umax)).T
        #print(data_save_quick.shape)
        np.savetxt("profile.dat", data_save_quick)

        popt = [xi**-1, xi, xi]
        popt, pcov = opt.curve_fit(fitfunc, xx, uu[idx]/umax, p0=popt)
        ax[1].plot(xx, fitfunc(xx, *popt), 'k--')
        #ax[1].plot(xx, xi*xx*np.exp(-xx*xi), 'k--')

        data = np.array(data)

        istart = int(tmax/dt)//2

        popt = np.polyfit(data[istart:, 0], np.log(data[istart:, 1]), 1)
        gamma = popt[0]

        print(f"gamma = {gamma}")

        fig_, ax_ = plt.subplots(1, 1)

        ax_.plot(data[:, 0], data[:, 1], label=r"$T_{\rm max}$")
        ax_.plot(data[:, 0], data[:, 2], label=r"$u_{\rm max}$")
        ax_.plot(data[:, 0], np.exp(gamma*data[:, 0]), label=r"fit")
        ax_.semilogy()
        ax_.set_xlabel("$t$")
        ax_.set_ylabel("scalar")
        ax_.legend()
        
        # plt.show()


        T0 = np.exp(-xi*xx)

        T_intp = intp.InterpolatedUnivariateSpline(xx, TT[idx]/Tmax, k=4)
        u_intp = intp.InterpolatedUnivariateSpline(xx, uu[idx]/Tmax, k=4)

        x = np.linspace(1e-7, Lx, 1000)
        T = T_intp(x)
        Tx = T_intp.derivative(1)(x)
        Txx = T_intp.derivative(2)(x)

        u = u_intp(x)
        ux = u_intp.derivative(1)(x)
        uxx = u_intp.derivative(2)(x)
        uxxx = u_intp.derivative(3)(x)
        uxxxx = u_intp.derivative(4)(x)

        T0 = np.exp(-xi*x)

        fig, ax = plt.subplots(8, 1)
        ax[0].plot(x, T)
        ax[1].plot(x, Tx)
        ax[2].plot(x, Txx)
        
        ax[3].plot(x, u)
        ax[4].plot(x, ux)
        ax[5].plot(x, uxx)
        ax[6].plot(x, uxxx)
        ax[7].plot(x, uxxxx)

        for _ax in ax:
            _ax.set_xlabel("$x$")
        ax[0].set_ylabel(r"$T$")
        ax[1].set_ylabel(r"$T_x$")
        ax[2].set_ylabel(r"$T_{xx}$")

        ax[3].set_ylabel(r"$u$")
        ax[4].set_ylabel(r"$u_x$")
        ax[5].set_ylabel(r"$u_{xx}$")
        ax[6].set_ylabel(r"$u_{xxx}$")
        ax[7].set_ylabel(r"$u_{xxxx}$")

        G = gamma + kappa*k**2 + Gamma

        Lam = (- 1 + np.sqrt(1 + 4 *kappa_eff*G))/(2*kappa_eff)

        fig, ax = plt.subplots(1, 2)

        ax[0].plot(x, T)
        ax[0].plot(x, np.exp(-Lam*x), label=r"$\exp(-\Lambda x)$")
        ax[0].semilogy()
        ax[0].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$T$")

        ax[0].legend()

        ax[1].plot(x, u)
        ax[1].plot(x, np.exp(-Lam*x), label=r"$\exp(-\Lambda x)$")
        ax[1].plot(x, np.exp(-k*x), label=r"$\exp(-k x)$")
        ax[1].set_xlabel(r"$x$")
        ax[1].set_ylabel(r"$u$")

        ax[1].legend()

        #ixstart = len(x) // 2
        #ixstop = (3*len(x)) // 4
        #popt = np.polyfit(x[ixstart:ixstop], np.log(u[ixstart:ixstop]), 1)
        #lam = -popt[0]
        #ax[1].plot(x, np.exp(popt[1] - lam*x))

        ax[1].semilogy()
        ax[1].legend()

        fig, ax = plt.subplots(1, 1)
        ax.plot(x, u * np.exp(Lam*x))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u \exp(\Lambda x)$")

        fig, ax = plt.subplots(1, 1)

        Ts = psi*xi*T0

        u_terms = [(G - (2*kappa_par*xi+1)*Ts)*k**2*u,
                   (k**2 + (xi + xi**2*kappa_eff - G + kappa_par*k**2)*Ts)*ux,
                   -(G + k**2*kappa_eff + (1 + 2*kappa_eff*xi)*Ts) * uxx,
                   (-1 + kappa_eff*Ts)*uxxx,
                   kappa_eff * uxxxx]

        ax.plot(x, u_terms[0], label="0")
        ax.plot(x, u_terms[1], label="1")
        ax.plot(x, u_terms[2], label="2")
        ax.plot(x, u_terms[3], label="3")
        ax.plot(x, u_terms[4], label="4")
        ax.plot(x, sum(u_terms), label="sum")
        ax.axvline(x=xi**-1, color="k", label=r"$1/\xi$")
        ax.legend()
        ax.set_ylim(-1000, 1000)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"term # in $u$ eqn.")

        plt.show()
        
