from backend import Simulation, plot
import numpy as np

corners = [[0, 0],
           [10, 0],
           [10, 5],
           [5, 5],
           [5, 15],
           [0, 15]]

L = 15
W = 10
A = 2*10*5
dA = A/(30*30)

g = dA*9.800

E0 = 190e9
Emin = 100
penal = 4
nu = 0.3
rho = 7750

maxVol = 50

sim = Simulation(corners, 70)
sim.createFunctions()
topBoundary = lambda x: np.isclose(x[1], 15)
corner = lambda x: np.isclose(x[0], 10) & np.isclose(x[1], 0)
x = np.ones(sim.domain.topology.index_map(2).size_local)
sim.fixedBoundary(topBoundary)
sim.applyForce(corner, [0, -1e9])

def obj(x):
    sim.density.interpolate(lambda _: x)
    sim.constituentEqns(Emin, E0, penal, nu)
    sim.createLP()
    uh = sim.problem.solve()
    sim.updateStress(uh)
    volFrac = sim.volFrac()
    comp = sim.compliance(uh)
    print(f"Volume Fraction: {volFrac}")
    print(f"Compliance: {comp}")
    plot(sim.domain, uh, sim.vm)

obj(x)

print(x.shape)