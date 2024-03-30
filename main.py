"""
312

Author: Vighnesh Nayak
Date: 07 Mar 2024
Github: https://github.com/Kugelblitz25
"""
from mpi4py import MPI
import dolfinx
import dolfin
import pyvista
import dolfinx.fem as fem
import ufl
from mshr import Polygon, generate_mesh
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from dolfinx.geometry import bb_tree, compute_closest_entity, create_midpoint_tree

corners = [[0, 0],
           [10, 0],
           [10, 5],
           [5, 5],
           [5, 15],
           [0, 15]]

L = 15
W = 10
A = 2*10*5*1e-4
dA = A/(30*30)

g = dA*98

E0 = 190e5
Emin = 100
penal = 4


def createPolygonalMesh(corners, meshDensity=15):
    corners = list(map(dolfin.Point, corners))
    domain = Polygon(corners)
    domain = generate_mesh(domain, meshDensity)

    with dolfin.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf") as file:
        file.write(domain)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh()

    return domain


def plot(msh, uh, stresses):
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    p = pyvista.Plotter()

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(
        msh, msh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # p.add_mesh(grid, show_edges=True)

    grid["u"] = uh.x.array.reshape((geometry.shape[0], 2))
    z = np.zeros(len(grid['u']))
    grid['u'] = np.c_[grid["u"], z]
    warped = grid.warp_by_vector("u", factor=0)

    warped.cell_data["VonMises"] = stresses.vector.array
    warped.set_active_scalars("VonMises")
    plt = p.add_mesh(warped, show_edges=True)
    plt = p.add_camera_orientation_widget()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        print("Unable to show plot.")


domain = createPolygonalMesh(corners, 70)

V = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
T = fem.FunctionSpace(domain, ("DG", 0))
density = fem.Function(T, name="Density")
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Function(V, name="Body Force")

densityArr = 7750*(np.arange(domain.topology.index_map(2).size_local)>=1000)
density.interpolate(lambda _: densityArr)

fdim = domain.topology.dim - 1
boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 15))

u_D = np.array([0, 0], dtype=dolfinx.default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

def bodyForce(x):
    x0 = x.T
    tree = bb_tree(domain, domain.geometry.dim)
    entities = np.arange(domain.topology.index_map(2).size_local, dtype='int32')
    midTree = create_midpoint_tree(domain, 2, entities)
    cells = compute_closest_entity(tree, midTree, domain, x0)
    d = density.eval(x0, cells)[:, 0]
    def oncorner(x):
        return np.isclose(x[0], 10) & np.isclose(x[1], 0)
    c = oncorner(x)
    B = np.stack([0*d, -g*d], axis=0)
    T = np.stack([0*c, -1e6*c], axis=0)
    return T

f.interpolate(bodyForce)

# Update the traction vector
T = fem.Constant(domain, dolfinx.default_scalar_type((0, 0)))

E = Emin + density**penal*(E0-Emin)
nu = 0.3
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu = E/(2*(1+nu))

ds = ufl.Measure('ds', domain=domain)
dx = ufl.dx

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

print("Formulation Completed.")
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

s = sigma(uh) - 1. / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))

V_von_mises = fem.FunctionSpace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)

VOL = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(density*ufl.dx)),op=MPI.SUM)/7750
print(f"Volume fraction: {VOL}")

compliance = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(ufl.inner(sigma(uh), epsilon(uh)) * ufl.dx)),op=MPI.SUM)
print(f"Compliance: {compliance}")

plot(domain, uh, density)