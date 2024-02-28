from mpi4py import MPI
import os

import ufl
import dolfinx as dfx
from dolfinx_mpc import LinearProblem, MultiPointConstraint
import numpy as np
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from utils import get_next_subfolder, mpi_comm, mpi_rank, mpi_size

tol = 1e-7
Lx = 4.0
Ly = 1.0

def inlet_boundary(x):
    return np.isclose(x[0], 0, atol=tol)

def outlet_boundary(x):
    return np.isclose(x[0], Lx, atol=tol)

def periodic_boundary(x):
    return np.isclose(x[1], Ly, atol=tol)

def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = x[0]
    out_x[1] = x[1] - Ly
    return out_x

if __name__ == "__main__":
    Nx = 200
    Ny = 50

    kappa = 0.01
    gamma = 3.0*1.1*1.0
    beta = 0.001
    ueps = 0.1

    dt = 0.01
    t_end = 8.0
    dump_intv = 10

    rtol = 1e-10

    results_folder = "results_cooling"
    folder = get_next_subfolder(results_folder)

    mesh = create_unit_square(MPI.COMM_WORLD, Nx, Ny)
    mesh.geometry.x[:, 0] *= Lx
    mesh.geometry.x[:, 1] *= Ly
    S = dfx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    V = dfx.fem.FunctionSpace(mesh, ufl.VectorElement("DG", "triangle", 0))

    x = ufl.SpatialCoordinate(mesh)

    # Create Dirichlet boundary condition
    inlet_facets = locate_entities_boundary(mesh, 1, inlet_boundary)
    outlet_facets = locate_entities_boundary(mesh, 1, outlet_boundary)
    inlet_dofs = dfx.fem.locate_dofs_topological(S, 1, inlet_facets)
    outlet_dofs = dfx.fem.locate_dofs_topological(S, 1, outlet_facets)

    bc_T_inlet = dfx.fem.dirichletbc(1., inlet_dofs, S)
    bcs_T = [bc_T_inlet]

    bc_p_inlet = dfx.fem.dirichletbc(1., inlet_dofs, S)
    bc_p_outlet = dfx.fem.dirichletbc(0., outlet_dofs, S)
    #bcs_p = [bc_p_inlet, bc_p_outlet]
    bcs_p = [bc_p_outlet]

    mpc_T = MultiPointConstraint(S)
    mpc_T.create_periodic_constraint_geometrical(S, periodic_boundary, periodic_relation, bcs_T)
    mpc_T.finalize()

    mpc_p = MultiPointConstraint(S)
    mpc_p.create_periodic_constraint_geometrical(S, periodic_boundary, periodic_relation, bcs_p)
    mpc_p.finalize()

    mpc_u = MultiPointConstraint(V)
    mpc_u.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, [])
    mpc_u.finalize()

    facet_indices = dfx.mesh.locate_entities(mesh, mesh.topology.dim-1, inlet_boundary)
    facet_markers = np.full_like(facet_indices, 1)
    sorted_facets = np.argsort(facet_indices)

    facet_tag = dfx.mesh.meshtags(mesh, mesh.topology.dim-1, facet_indices[sorted_facets], facet_markers[sorted_facets])
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    T = ufl.TrialFunction(S)
    b = ufl.TestFunction(S)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    T_ = dfx.fem.Function(mpc_T.function_space, name="T")
    T_1 = dfx.fem.Function(mpc_T.function_space, name="T")

    factor_ = dfx.fem.Function(mpc_p.function_space, name="factor")
    factor_.x.array[:] = 1.0

    p_ = dfx.fem.Function(mpc_p.function_space, name="p") #(S, name="p")

    #u0 = ufl.as_vector((1.0 + 1.0*ufl.sin(2*ufl.pi*x[1]), 0.))
    ux0 = 1.0 + ueps*ufl.sin(2*ufl.pi*x[1])

    F_p = factor_ * ufl.dot(ufl.grad(T), ufl.grad(b)) * ufl.dx - ux0 * b * ds(1)
    a_p = ufl.lhs(F_p)
    L_p = ufl.rhs(F_p)

    problem_p = LinearProblem(a_p, L_p, mpc_p, u=p_, bcs=bcs_p,
                              petsc_options={"ksp_type": "cg", "ksp_rtol": rtol, "pc_type": "hypre", "pc_hypre_type": "boomeramg",
                                             "pc_hypre_boomeramg_max_iter": 1, "pc_hypre_boomeramg_cycle_type": "v",
                                             "pc_hypre_boomeramg_print_statistics": 0})

    u_ = - factor_ * ufl.grad(p_)

    F_T = (T - T_1)*b/dt * ufl.dx + ufl.dot(u_, ufl.grad(T)) * b * ufl.dx + kappa * ufl.inner(ufl.grad(T), ufl.grad(b)) * ufl.dx + gamma * T * b * ufl.dx
    a_T = ufl.lhs(F_T)
    L_T = ufl.rhs(F_T)

    problem_T = LinearProblem(a_T, L_T, mpc_T, bcs=bcs_T,
                              petsc_options={"ksp_type": "bcgs", "ksp_rtol": rtol, "pc_type": "jacobi"})

    # Project u for visualization (only used a few times)
    F_u = ufl.dot(u - u_, v) * ufl.dx
    a_u = ufl.lhs(F_u)
    L_u = ufl.rhs(F_u)
    problem_u = LinearProblem(a_u, L_u, mpc_u, bcs=[])

    xdmff_T = dfx.io.XDMFFile(mesh.comm, os.path.join(folder, "T.xdmf"), "w")
    xdmff_p = dfx.io.XDMFFile(mesh.comm, os.path.join(folder, "p.xdmf"), "w")
    xdmff_u = dfx.io.XDMFFile(mesh.comm, os.path.join(folder, "u.xdmf"), "w")

    xdmff_T.write_mesh(mesh)
    xdmff_p.write_mesh(mesh)
    xdmff_u.write_mesh(mesh)

    it = 0
    t = 0.
    while t < t_end:
        if mpi_rank == 0:
            print(f"t = {t}")

        # Update factor_ function
        factor_.x.array[:] = beta**-T_1.x.array

        p_h = problem_p.solve()

        p_.x.array[:] = p_h.x.array[:]
        p_.x.scatter_forward()

        T_h = problem_T.solve()

        T_1.x.array[:] = T_h.x.array
        T_1.x.scatter_forward()

        if it % dump_intv == 0:
            xdmff_T.write_function(T_1, t)
            xdmff_p.write_function(p_, t)

            # Project u for visualization
            u_h = problem_u.solve()
            u_h.name = "u"
            xdmff_u.write_function(u_h, t)

        t += dt
        it += 1

    xdmff_T.close()
    xdmff_p.close()
    xdmff_u.close()