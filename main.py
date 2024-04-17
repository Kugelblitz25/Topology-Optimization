"""
312

Author: Vighnesh Nayak
Date: 07 Mar 2024
Github: https://github.com/Kugelblitz25
"""

from backend import Simulation
from frontend import OptimizerPlot, setThreshold, saveAnimation
import numpy as np


class TopOpt:
    def __init__(self, corners: np.ndarray, 
                 meshDensity: int = 100,
                 E0: float = 190e9,
                 Emin:float = 100,
                 nu: float = 0.3,
                 penalty: int = 4) -> None:
        self.simulation = Simulation(corners, meshDensity)
        self.W = corners[:,0].max() - corners[:,0].min()
        self.L = corners[:,1].max() - corners[:,1].min()
        self.density = np.ones(self.simulation.domain.topology.index_map(2).size_local)
        self.elemLocs = self.simulation.locs
        self.numElems = len(self.density)
        self.simulation.createFunctions()
        self.E = E0
        self.Emin = Emin
        self.nu = nu
        self.penalty = penalty

    def createJoints(self, joints: dict):
        self.fixedJoints = joints['f']
        for locFunction in self.fixedJoints:
            self.simulation.fixedJoint(locFunction)
        
        self.rollingJoints = joints['r']
        for locFunction in self.rollingJoints:
            self.simulation.rollingJoint(locFunction)

    def applyForces(self, forces: dict[tuple, tuple]):
        self.forces = forces
        forceTuples = []
        for loc in forces:
            forceTuples.append((loc, forces[loc]))
        self.simulation.applyForce(forceTuples)

    def objectiveFunction(self):
        self.simulation.density.interpolate(lambda _: self.density)
        self.simulation.constituentEqns(self.Emin, self.E, self.penalty, self.nu)
        self.simulation.solve()
        self.simulation.updateStress()
        comp = self.simulation.compliance()
        return comp
    
    def normalize(self, vec: np.ndarray):
        return (vec-vec.min())/(vec.max()-vec.min())
    
    def percentileMask(self, vec: np.ndarray, p: float = 50, defaultVal: float = 0):
        mask = vec >= np.percentile(vec, p)
        return np.where(mask, vec, defaultVal)
    
    def gradient(self, p: int = 300):
        C = self.simulation.complianceArr.vector.array 
        C = C * self.simulation.domain.h(2,np.arange(len(C), dtype='int32'))
        num = -self.penalty*(self.E-self.Emin)*self.density**(self.penalty-1)*C 
        denom = self.Emin + self.density**self.penalty*(self.E-self.Emin)
        grad = self.normalize(num/denom)**p
        return self.percentileMask(grad)
    
    def gaussianFilter(self, vec: np.ndarray, R: float = 0.3):
        sigma = R/3
        def filter(i):
            mask = (self.elemLocs[i,0]-R<=self.elemLocs[:,0]) & (self.elemLocs[:,0]<=self.elemLocs[i,0]+R) & (self.elemLocs[i,1]-R<=self.elemLocs[:,1]) & (self.elemLocs[:,1]<=self.elemLocs[i,1]+R)
            y = np.where(mask, vec, 0)
            weight = np.exp(-((self.elemLocs[:,0]-self.elemLocs[i,0])**2+(self.elemLocs[:,1]-self.elemLocs[i,1])**2)/(2*sigma**2))
            weight = weight/weight.sum()
            return (y*weight).sum()
        filter = np.vectorize(filter)
        return filter(np.arange(self.numElems))
    
    def optimize(self, numIter: int = 50, 
                       targetVol: float = 0.5, 
                       lr: float = 0.1,
                       p: float = 300,
                       gr: float = 0.8,
                       saveResult: bool = True, 
                       animate: bool = False):
        vPrev = 2
        history = []
        plotter = OptimizerPlot(numIter, targetVol)
        plotter.init()        
        for i in range(numIter):
            comp = self.objectiveFunction()
            vol = self.density.mean()
            plotter.update(vol, comp)
            history.append(self.density)
            print(f"Iteration: {i+1}  Volume Fraction: {vol}, Compliance: {comp}")
            if vol <= targetVol or abs(vol-vPrev)<0.0001:
                break
            vPrev = vol
            grad = self.gradient(p)
            self.density = np.maximum(0.01, self.density-lr*grad)
            self.density = self.gaussianFilter(self.density, gr)
            self.density = self.normalize(self.density)
            self.density = self.percentileMask(self.density, 10, 0.01)

        plotter.stop()
        self.density = setThreshold(self.density, self.elemLocs)
        
        print(f'Optimization Completed. \nFinal Compliance = {self.objectiveFunction()}')
        self.simulation.displayDistribution()
        self.simulation.displayResult()

        if saveResult:
            result = np.empty((3, self.numElems))
            result[:2, :] = self.elemLocs[:, :2].T
            result[2, :] = self.density
            np.save('result', result)

        if animate:
            saveAnimation(np.array(history), self.elemLocs)

def optimLBrac():
    corners = np.array([[0, 0],
            [15, 0],
            [15, 5],
            [5, 5],
            [5, 15],
            [0, 15]])
    topBoundary = lambda x: np.isclose(x[1], 15)
    forces = {(15, 5): (0, -1e3)}
    opt = TopOpt(corners, meshDensity=70)
    opt.createFixedBoundaries([topBoundary])
    opt.applyForces(forces)
    opt.optimize(targetVol=0.3,animate=True)

def optimBridge():
    corners = np.array([[0, 0],
                        [36, 0],
                        [36, 6],
                        [0, 6]])
    leftBoundary = lambda x: np.isclose(x[0], 0, atol=1e-2) & (x[1]<=0.3)
    rightBoundary = lambda x: np.isclose(x[1], 0, rtol=1e-2) & np.isclose(x[0], 36, rtol=1e-2)
    forces = {(18, 6): (0, -1e4)}
    opt = TopOpt(corners, meshDensity=120)
    opt.createJoints({'f': [leftBoundary], 'r': [rightBoundary]})
    opt.applyForces(forces)
    opt.optimize(targetVol=0.3,animate=True,lr=0.1, p=300, gr=0.8)

if __name__ == "__main__":
    optimBridge()