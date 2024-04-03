from backend import Simulation, plot, plotDensity
import numpy as np
import matplotlib.pyplot as plt

corners = [[0, 0],
           [15, 0],
           [15, 5],
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
corner = lambda x: np.isclose(x[0], 15) & np.isclose(x[1], 5)
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

R = 0.5
sigma = R/3

def filter(i):
    mask = (coord[i,0]-R<=coord[:,0]) & (coord[:,0]<=coord[i,0]+R) & (coord[i,1]-R<=coord[:,1]) & (coord[:,1]<=coord[i,1]+R)
    y = np.where(mask, x, 0)
    weight = np.exp(-((coord[:,0]-coord[i,0])**2+(coord[:,1]-coord[i,1])**2)/(2*sigma**2))
    weight = weight/weight.sum()
    return (y*weight).sum()

filter = np.vectorize(filter)

n_iters = 50
for i in range(n_iters):
    vol, comp = obj(x)
    print(f"Iteration: {i+1}  Volume Fraction: {vol}, Compliance: {comp}")
    if vol <=50:
        break
    grad = gradCalc(sim)**200
    x -= 0.1*grad
    x = np.maximum(0, x)
    x = (x-x.min())/(x.max()-x.min())
    x = filter(np.arange(len(x)))

data = np.vstack([coord[:,0], coord[:,1], x])
np.save('data', data)
x = (x>=0.7).astype('float64')
vol, comp = obj(x, True)
plotDensity(sim.domain, sim.density, )

fig, axes = plt.subplots(10, 5, figsize=(20,40), sharex=True)
for p in range(1,6):
    for t in range(1,11):
        axes[t-1][p-1].scatter(coord[:,0], coord[:,1], c=(x**p)<t/10, marker='.', cmap='gray')
        axes[t-1][p-1].set_title(f'power = {p}, threshold = {t/10}')
plt.show()