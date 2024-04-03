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
                         Constant, assemble_vector,
                         locate_dofs_topological,
                         dirichletbc, Expression,
                         assemble_scalar, form)
from dolfinx.mesh import locate_entities_boundary
from ufl import (TrialFunction, CellVolume,
                 TestFunction, Measure, dot,
                 sym, grad, dx, inner, tr,
                 nabla_div, Identity, sqrt)
from mshr import Polygon, generate_mesh
from dolfinx.fem.petsc import LinearProblem
import numpy as np


def plot(msh, displacements, stresses, init=False):
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    p = pyvista.Plotter()

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(msh, msh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    if init:
        p.add_mesh(grid)

    grid["u"] = displacements.x.array.reshape((geometry.shape[0], 2))
    z = np.zeros(len(grid['u']))
    grid['u'] = np.c_[grid["u"], z]
    warped = grid.warp_by_vector("u", factor=1)

    warped.cell_data["VonMises"] = stresses.vector.array
    warped.set_active_scalars("VonMises")
    p.add_mesh(warped,clim=[0, 0.3*max(stresses.vector.array)], cmap='jet')
    p.add_camera_orientation_widget()

    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        print("Unable to show plot.")

def plotDensity(msh, density, lim=[0,1]):
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    p = pyvista.Plotter()

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(msh, msh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    grid.cell_data["VonMises"] = 1-density.vector.array
    # grid.set_active_scalars("VonMises")
    p.add_mesh(grid, cmap='gray', clim=lim)
    p.add_camera_orientation_widget()

    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        print("Unable to show plot.")

class Simulation:
    def __init__(self, corners, meshDensity = 15) -> None:
        self.corners = corners
        self.locs = []
        self.domain = self.createPolygonalMesh(corners, meshDensity)
        self.V = VectorFunctionSpace(self.domain, ("Lagrange", 1))
        self.U = FunctionSpace(self.domain, ("DG", 0))
        self.bcs = []

    def createPolygonalMesh(self, corners, meshDensity=15):
        corners = list(map(dolfin.Point, corners))
        domain = Polygon(corners)
        domain = generate_mesh(domain, meshDensity)

        for cell in dolfin.cells(domain,):
            self.locs.append((cell.midpoint().x(), cell.midpoint().y()))
        
        self.locs = np.array(self.locs)

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
        self.complianceArr = Function(self.U, name="Compliance")

    def applyForce(self, f, force):
        def bodyForce(x):
            c = f(x)
            T = np.stack([force[0]*c, force[1]*c], axis=0)
            return T

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
        comp = MPI.COMM_WORLD.allreduce(assemble_scalar(form(inner(self.eps(u), self.sigma(u)) * dx)),op=MPI.SUM)
        return comp

    def createLP(self):
        ds = Measure('ds', domain=self.domain)
        a = inner(self.sigma(self.u), self.eps(self.v)) * dx
        L = dot(self.B, self.v) * dx + dot(self.T, self.v) * ds

        self.problem = LinearProblem(a, L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    def updateStress(self, u):
        stress_expr = Expression(self.von_Mises(u), self.U.element.interpolation_points())
        comp_expr = Expression(inner(self.eps(u), self.sigma(u)), self.U.element.interpolation_points())
        self.complianceArr.interpolate(comp_expr)
        self.vm.interpolate(stress_expr)
