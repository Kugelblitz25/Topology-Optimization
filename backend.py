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
from dolfinx.fem import (VectorFunctionSpace, 
                         FunctionSpace, 
                         Function, 
                         Constant,
                         locate_dofs_topological,
                         dirichletbc, Expression,
                         assemble_scalar, form)
from dolfinx.mesh import locate_entities_boundary
from ufl import (TrialFunction,
                 TestFunction, Measure, dot,
                 sym, grad, dx, inner, tr,
                 nabla_div, Identity, sqrt)
from mshr import Polygon, generate_mesh
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from dolfinx.geometry import (bb_tree, 
                              compute_closest_entity, 
                              create_midpoint_tree)

def plot(msh, displacements, stresses):
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    p = pyvista.Plotter()

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(msh, msh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # p.add_mesh(grid, show_edges=True)

    grid["u"] = displacements.x.array.reshape((geometry.shape[0], 2))
    z = np.zeros(len(grid['u']))
    grid['u'] = np.c_[grid["u"], z]
    warped = grid.warp_by_vector("u", factor=1)

    warped.cell_data["VonMises"] = stresses.vector.array
    warped.set_active_scalars("VonMises")
    p.add_mesh(warped, show_edges=True, clim=[0, max(stresses.vector.array)])
    p.add_camera_orientation_widget()

    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        print("Unable to show plot.")

class Simulation:
    def __init__(self, corners, meshDensity = 15) -> None:
        self.corners = corners
        self.domain = self.createPolygonalMesh(corners, meshDensity)
        self.V = VectorFunctionSpace(self.domain, ("Lagrange", 1))
        self.U = FunctionSpace(self.domain, ("DG", 0))
        self.bcs = []

    def createPolygonalMesh(self, corners, meshDensity=15):
        corners = list(map(dolfin.Point, corners))
        domain = Polygon(corners)
        domain = generate_mesh(domain, meshDensity)

        with dolfin.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf") as file:
            file.write(domain)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
            domain = xdmf.read_mesh()

        return domain
    
    def createFunctions(self):
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.B = Function(self.V, name="Body Force")
        self.T = Constant(self.domain, dolfinx.default_scalar_type((0, 0)))
        self.density = Function(self.U, name="Density")
        self.vm = Function(self.U, name="Von Mises Stress")

    def applyBodyForce(self, g, rho):
        def bodyForce(x):
            x0 = x.T
            tree = bb_tree(self.domain, self.domain.geometry.dim)
            entities = np.arange(self.domain.topology.index_map(2).size_local, dtype='int32')
            midTree = create_midpoint_tree(self.domain, 2, entities)
            cells = compute_closest_entity(tree, midTree, self.domain, x0)
            d = rho*self.density.eval(x0, cells)[:, 0]
            return np.stack([0*d, -g*d], axis=0)

        self.B.interpolate(bodyForce)

    def fixedBoundary(self, f):
        fdim = self.domain.topology.dim - 1
        boundary_facets = locate_entities_boundary(self.domain, fdim, f)

        u_D = np.array([0, 0], dtype=dolfinx.default_scalar_type)
        dofs = locate_dofs_topological(self.V, fdim, boundary_facets)
        self.bcs.append(dirichletbc(u_D, dofs, self.V))

    def constituentEqns(self, Emin, E0, penal, nu):
        E = Emin + self.density**penal*(E0-Emin)
        lambda_ = E*nu/(1+nu)/(1-2*nu)
        mu = E/(2*(1+nu))

        self.eps = lambda u: sym(grad(u))
        self.sigma = lambda u: lambda_ * nabla_div(u) * Identity(len(u)) + 2 * mu * self.eps(u)
        s = lambda u: self.sigma(u) - 1. / 3 * tr(self.sigma(u)) * Identity(len(u))
        self.von_Mises = lambda u: sqrt(3. / 2 * inner(s(u), s(u)))

    def volFrac(self):
        VOL = MPI.COMM_WORLD.allreduce(assemble_scalar(form(self.density*dx)),op=MPI.SUM)
        return VOL
    
    def compliance(self, u):
        return MPI.COMM_WORLD.allreduce(assemble_scalar(form(inner(self.sigma(u), self.eps(u)) * dx)),op=MPI.SUM)

    def createLP(self):
        ds = Measure('ds', domain=self.domain)
        a = inner(self.sigma(self.u), self.eps(self.v)) * dx
        L = dot(self.B, self.v) * dx + dot(self.T, self.v) * ds

        print("Formulation Completed.")
        self.problem = LinearProblem(a, L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    def updateStress(self, u):
        stress_expr = Expression(self.von_Mises(u), self.U.element.interpolation_points())
        self.vm.interpolate(stress_expr)
