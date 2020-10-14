from fenics import *
import mshr as ms
import ufl as uf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import *
import os
import json


def saveAsFile(namefile):
    def decorator(func):
        def draw(self):
            func(self)
            plt.savefig(namefile)
            plt.clf()
        return draw
    return decorator


class Formatter:
    def __init__(self):
        self.title = "mesh"
        self.namefile = "namefile"


class MeshedGeometry():

    ##############################
    # geometry patterns
    ##############################
    @staticmethod
    def simpleSymmetric(length=10, width=12, exitWidth=16):
        return {
            "geometry": [

                Point(length, 0.0),
                Point(length, width),
                Point(0.0, width),
                Point(0.0, 0.0)
            ],
            "exit": {
                "bottom": width/2.0-exitWidth/2.0,
                "left": length-0.1,
                "right": length+0.1,
                "top": width/2.0+exitWidth/2.0
            }
        }

    @staticmethod
    def simpleAsymmetric(length=10, width=12, exitWidth=16, shift=2):
        return {
            "geometry": [
                Point(0.0, 0.0),
                Point(0.0, length),
                Point(width, length),
                Point(width, 0.0)
            ],
            "exit": {
                "bottom": width/2.0-exitWidth/2.0+shift,
                "left": length-0.1,
                "right": length+0.1,
                "top": width/2.0+exitWidth/2.0-shift
            }
        }

    ##############################
    # obstackle patterns
    ##############################
    @staticmethod
    def rectangleObstackle(width=0.8, frontDistance=1, backDistance=2, middlePosition=0):
        return {
            "rect": [Point(frontDistance, middlePosition-width/2),
                     Point(frontDistance, middlePosition+width/2),
                     Point(backDistance, middlePosition+width/2),
                     Point(backDistance, middlePosition-width/2)],
            "TESTrect": [
                Point(3.0, 1.0),
                Point(3.0, 4.0),
                Point(1.0, 4.0),
                Point(1.0, 1.0),
            ]

        }

    @staticmethod
    def initialRectangle():
        return Expression("x[0] > 1.0 && x[0] < 2.0 && x[1] > 2.0 && x[1] < 1.0 ? 6.0 : 0.0", degree=2)

    def __init__(self, compSettings, denisityMesh):

        ######################################
        # TODO program each type of symmetric
        ######################################
        if compSettings["geometryType"] == "simpleSymmetric":
            _length = compSettings["calibration"]["length"]
            _width = compSettings["calibration"]["width"]
            _exitWidth = compSettings["calibration"]["exitWidth"]
            geometry = MeshedGeometry.simpleSymmetric(
                length=_length, width=_width, exitWidth=_exitWidth)
            domain = ms.Polygon(geometry["geometry"])
            self.geometry = geometry

        ######################################
        # TODO program each type of obstackle
        ######################################
        hasObstackle = False
        if compSettings["obstackleType"] == "rectangleObstackle":
            _obstackleWidth = compSettings["obstackleCalibration"]["width"]
            _frontDistance = compSettings["obstackleCalibration"]["frontDistance"]
            _backDistance = compSettings["obstackleCalibration"]["backDistance"]

            _frontDistance = compSettings["calibration"]["length"] - \
                _frontDistance
            _backDistance = compSettings["calibration"]["length"] - \
                _backDistance
            _middlePosition = compSettings["calibration"]["width"]/2.0
            obstackle = MeshedGeometry.rectangleObstackle(
                width=_obstackleWidth, frontDistance=_frontDistance, backDistance=_backDistance, middlePosition=_middlePosition)
            # TODO generalize this piece of code to set of parts of obstackle
            obstackle = ms.Polygon(obstackle["rect"])
            hasObstackle = True

        if hasObstackle:
            self.mesh = ms.generate_mesh(domain-obstackle, denisityMesh)
        else:
            self.mesh = ms.generate_mesh(domain, denisityMesh)

        self.initialCondition = None

        def isExitCond(p):
            xmin = self.geometry["exit"]["left"]
            xmax = self.geometry["exit"]["right"]
            ymin = self.geometry["exit"]["bottom"]
            ymax = self.geometry["exit"]["top"]
            return xmin < p[0] < xmax and ymin < p[1] < ymax

        self.isExitCond = isExitCond

    def generateMesh(self):
        V = FunctionSpace(self.mesh, "P", 2)
        self.V = V
        self.hmin = self.mesh.hmin()
        self.hmax = self.mesh.hmax()
        print("hmin::", self.hmin)

    @saveAsFile("mesh.jpg")
    def plotMesh(self):
        plot(self.mesh)


class PhysicalModel(MeshedGeometry):
    def __init__(self, geoSettings, meshDensity):
        super().__init__(geoSettings, meshDensity)

        def isExit(x, on_boundary):
            return on_boundary and self.isExitCond(x)

        def isWall(x, on_boundary):
            return on_boundary and not self.isExitCond(x)

        self.wall = isWall
        self.exit = isExit

    def isSetBoundaries(self):
        if hasattr(self, "wall") and hasattr(self, "exit"):
            return True

    def generateEikonalEqution(self):
        def norm(potentialField, theta):
            dx0 = project(potentialField.dx(0))+theta
            dx1 = project(potentialField.dx(1))
            unnormed_grad_phi = project(grad(potentialField))
            module = sqrt(dx0*dx0+dx1*dx1)
            return module

        u = TestFunction(self.V)
        v = TrialFunction(self.V)
        sigma = 10
        F = u*v*dx + sigma*sigma*dot(grad(u), grad(v))*dx
        a, L = lhs(F), rhs(F)

        bc = DirichletBC(self.V, Constant(1), self.exit)
        u = Function(self.V)
        solve(a == L, u, bc)
        phi = uf.ln(u)
        return phi

    def generateUnitFieldPhi(self, typeVectorField):
        if typeVectorField == "EIKONAL":
            UnitFieldPhi = self.generateEikonalEqution()
        self.unitFieldPhi = UnitFieldPhi
        return UnitFieldPhi

    @saveAsFile("unitFieldPhi.jpg")
    def plotUnitFieldPhi(self):
        plot(self.unitFieldPhi)
        plot(self.mesh)

    def velocityParam(self):
        self.velocity.maxv = 10

    def velocity(self, maxv):
        maxv = self.velocity.maxv

        def V(rho):
            return maxv*np.exp(rho**-2)
        return V


class NumericalModel:
    def __init__(self, PhysicalModel):
        self.PM = PhysicalModel
        self.rho = TrialFunction(PhysicalModel.V)
        self.rhoold = TrialFunction(PhysicalModel.V)
        self.test = TestFunction(PhysicalModel.V)
        self.normal = FacetNormal(mesh)

    def generate_V_rho():
        def V(rho, VMAX, RHOMAX):
            sigma = RHOMAX/3.0  # z zasady 3 sigm
            exp(rho)
            return

    def defineEquation(self, sigma=0.2, dt=0.1):
        rho = self.rho
        rhoold = self.rhoold
        test = self.test
        normed_phi = self.PM.unitFieldPhi
        V_rhoold =
        equation = {}
        equation["timeDer"] = (rho - rhoold)*test*dx/dt
        equation["adv"] = rho*V_rhoold*dot(normed_phi, grad(test))*dx
        equation["diff"] = sigma * dot(grad(rhoold), grad(test)) * dx
        equation["boundaryDiff"] = 0
        equation["boundaryAdv"] = 0


class Simulation:
    pass
