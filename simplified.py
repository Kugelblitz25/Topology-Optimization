from backend import Simulation, plot
import numpy as np
from scipy.optimize import minimize

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

E0 = 190e5
Emin = 100
penal = 4
nu = 0.3
rho = 7750

maxVol = 50

sim = Simulation(corners, 30)
sim.createFunctions()
topBoundary = lambda x: np.isclose(x[1], 15)
sideBoundary = lambda x: np.isclose(x[0], 10)
x = np.ones(sim.domain.topology.index_map(2).size_local)

def obj(x):
    sim.density.interpolate(lambda _: x)
    sim.applyBodyForce(g, rho)
    sim.fixedBoundary(topBoundary)
    sim.fixedBoundary(sideBoundary)
    sim.constituentEqns(Emin, E0, penal, nu)
    sim.createLP()
    uh = sim.problem.solve()
    sim.updateStress(uh)
    volFrac = sim.volFrac()
    comp = sim.compliance(uh)
    print(f"Volume Fraction: {volFrac}")
    print(f"Compliance: {comp}")
    dV = maxVol - volFrac
    if dV >= 0:
        penalty = 0
    else:
        penalty = 4 * dV**2

    return comp + penalty

bounds = [(0, 1)] * (sim.domain.topology.index_map(2).size_local**2)
print(obj(x))

# result = minimize(obj, x, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': lambda x: maxVol-sum(x)/len(x)})

# plot(sim.domain, uh, sim.vm)