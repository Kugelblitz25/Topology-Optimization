from backend import Simulation, plot, plotDensity
import numpy as np

corners = [[0, 0],
           [10, 0],
           [10, 5],
           [5, 5],
           [5, 15],
           [0, 15]]

# corners = [[0, 0],
#            [15, 15],
#            [0, 15]]

# corners = [[0, 0],
#            [20, 0],
#            [20, 5],
#            [0, 5]]

L = 15
W = 10

E0 = 190e9
Emin = 100
penal = 4
nu = 0.3
rho = 7750

sim = Simulation(corners, 100)
sim.createFunctions()
topBoundary = lambda x: np.isclose(x[1], 15)
# lefCorner = lambda x: np.isclose(x[0], 0) & (x[1]<=1)
# rightCorner = lambda x: np.isclose(x[0], 20) & (x[1]<=1)
corner = lambda x: np.isclose(x[0], 10) & np.isclose(x[1], 0)
x = np.ones(sim.domain.topology.index_map(2).size_local)
sim.fixedBoundary(topBoundary)
# sim.fixedBoundary(lefCorner)
# sim.fixedBoundary(rightCorner)
sim.applyForce(corner, [0, -1e9])
coord = sim.locs


def obj(x, plt=False):
    global uh
    sim.density.interpolate(lambda _: x)
    sim.constituentEqns(Emin, E0, penal, nu)
    sim.createLP()
    uh = sim.problem.solve()
    sim.updateStress(uh)
    volFrac = sim.volFrac()
    comp = sim.compliance(uh)
    if plt:
        plot(sim.domain, uh, sim.vm)
    return volFrac, comp

def gradCalc(sim: Simulation):
    C = sim.complianceArr.vector.array
    num = -penal*(E0-Emin)*x**(penal-1)*C 
    denom = Emin + x**penal*(E0-Emin)
    grad = num/denom 
    # grad = 0.01+1e-2*(grad-grad.min())/(grad.max()-grad.min())
    grad = (grad-grad.min())/(grad.max()-grad.min())
    # print(grad)
    return grad

n_iters = 50
for i in range(n_iters):
    vol, comp = obj(x)
    print(f"Iteration: {i+1}  Volume Fraction: {vol}, Compliance: {comp}")
    if vol <=50:
        break

    # for j in range(len(sim.density.vector.array)):
    #     if sim.density.vector.array[j] < 1:
    #         sim.density.vector.array[j] = 0
    if i > n_iters/4: 
        for j in range(len(x)):
            if x[j] < 0.7 and x[j] >0.5:
                x[j] = 0

    grad = gradCalc(sim)**1000
    x -= 0.1*grad
    mask = x>=0.4
    x = np.where(mask, x, 0)


vol, comp = obj(x, True)

plotDensity(sim.domain, sim.density)
print(type(sim.density), type(sim.domain))

# print(sim.density.vector.array)
# print(type(sim.density.vector.array))

# print(sim.density.vector.array.shape)

# with open('density_array.txt', 'w') as f:
#     for i in sim.density.vector.array:
#         f.write(str(i) + '\n')

# for i in range(len(sim.density.vector.array)):
#     if sim.density.vector.array[i] < 0.9:
#         sim.density.vector.array[i] = 0

# with open('density_array_updated.txt', 'w') as f:
#     for i in sim.density.vector.array:
#         f.write(str(i) + '\n')

# plotDensity(sim.domain, sim.density)
