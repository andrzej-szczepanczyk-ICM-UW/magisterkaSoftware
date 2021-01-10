# from fenics import *
from dolfin import *
import mshr as ms
import ufl as uf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import *
import os
import shutil
import json

from matplotlib.cm import *
from matplotlib.colors import *
from matplotlib import rc

import multiprocessing


def saveAsFile(namefile):
    def decorator(func):
        def draw(self, *args):
            func(self, *args)
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
    def simpleSymmetric(length=10, width=12, exitWidth=16, vestibuleLength=0.5):
        return {
            "geometry": [

                Point(length+vestibuleLength, 0.0),
                Point(length+vestibuleLength, width),
                Point(0.0, width),
                Point(0.0, 0.0)
            ],
            "exit": {
                "bottom": width/2.0-exitWidth/2.0,
                "left": length+vestibuleLength-0.02,
                "right": length+vestibuleLength+0.02,
                "top": width/2.0+exitWidth/2.0
            }
        }

    @staticmethod
    def simpleAsymmetric(length=10, width=12, exitWidth=16, shift=2, vestibuleLength=0.5):
        return {
            "geometry": [
                Point(0.0, 0.0),
                Point(0.0, length+vestibuleLength),
                Point(width, length+vestibuleLength),
                Point(width, 0.0)
            ],
            "exit": {
                "bottom": width/2.0-exitWidth/2.0+shift,
                "left": length+vestibuleLength-0.05,
                "right": length+vestibuleLength+0.05,
                "top": width/2.0+exitWidth/2.0-shift
            }
        }

    ##############################
    # vestibule patterns
    ##############################
    @staticmethod
    def vestibule(length=10, width=12, exitWidth=16, shift=2, vestibuleLength=0.5):
        middlePosition = width/2 + shift

        print("vest length in func ", vestibuleLength)
        FL = Point(length+vestibuleLength, middlePosition-exitWidth/2)
        FR = Point(length+vestibuleLength, middlePosition+exitWidth/2)
        BL = Point(length, middlePosition-exitWidth/2)
        BR = Point(length, middlePosition+exitWidth/2)

        CFL = Point(length+vestibuleLength, 0)
        CFR = Point(length+vestibuleLength, width)
        CBL = Point(length, 0)
        CBR = Point(length, width)

        mainSpaceL = ms.Polygon([FR, CFR, CBR, BR])
        mainSpaceR = ms.Polygon([CFL, FL, BL, CBL])
        cornerL = ms.Circle(FL, vestibuleLength, segments=16)
        cornerR = ms.Circle(FR, vestibuleLength, segments=16)
        return mainSpaceL + mainSpaceR + cornerL + cornerR

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
    def anvilObstackle(length=10, width=12, thickness=0.5, middlePosition=0, frontDistance=2.0):
        return {
            "LFCircleMP": Point(frontDistance, middlePosition-width/2.0),
            "RFCircleMP": Point(frontDistance, middlePosition+width/2.0),
            "LBCircleMP": Point(frontDistance+length, middlePosition-width/2.0),
            "RBCircleMP": Point(frontDistance+length, middlePosition+width/2.0)
        }

    @staticmethod
    def initialRectangle(valueMax):
        return Expression("x[0] > 1.0 && x[0] < 10.0 && x[1] > 2.0 && x[1] < 12.0 ? {} : 0.0".format(valueMax), degree=2)

    @staticmethod
    def initialBellCurveExp(valueMax, middle=(0, 0)):
        px = middle[0]
        py = middle[1]
        return Expression("{}*exp(-((x[0]-{})*(x[0]-{})+(x[1]-{})*(x[1]-{})))".format(valueMax, px, px, py, py), degree=2)

    @staticmethod
    def initialBellCurveInvSquare(valueMax, middle=(0, 0)):
        px = middle[0]
        py = middle[1]
        return Expression("{}/((x[0]-{})*(x[0]-{})+(x[1]-{})*(x[1]-{})+2)".format(valueMax, px, px, py, py), degree=2)

    def __init__(self, compSettings, denisityMesh):

        ######################################
        # TODO program each type of symmetric
        ######################################
        if compSettings["geometryType"] == "simpleSymmetric":
            _length = compSettings["calibration"]["length"]
            _width = compSettings["calibration"]["width"]
            _exitWidth = compSettings["calibration"]["exitWidth"]
            _vestibuleLength = compSettings["calibration"]["vestibuleLength"]
            geometry = MeshedGeometry.simpleSymmetric(
                length=_length, width=_width, exitWidth=_exitWidth, vestibuleLength=_vestibuleLength)
            domain = ms.Polygon(geometry["geometry"])
            self.geometry = geometry

        ######################################
        # TODO program each type of obstackle
        ######################################
        hasObstackle = False
        if compSettings["obstackleType"] == "rectangleObstackle":
            _obstackleWidth = compSettings["obstackleCalibration"]["width"]
            # _frontDistance = compSettings["obstackleCalibration"]["frontDistance"]
            # _backDistance = compSettings["obstackleCalibration"]["backDistance"]

            # konwersja w ukłądzie współrzędnych dla SAMEJ przeszkody na układ właściwy dla pomieszczenia
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

        if compSettings["obstackleType"] == "anvilObstackle":
            length = compSettings["calibration"]["length"]
            width = compSettings["calibration"]["width"]

            _obstackleWidth = compSettings["obstackleCalibration"]["width"]
            _obstackleLength = compSettings["obstackleCalibration"]["length"]
            _frontDistance = compSettings["obstackleCalibration"]["frontDistance"]
            _thickness = compSettings["obstackleCalibration"]["thickness"]
            _middlePosition = compSettings["obstackleCalibration"]["middlePosition"]
            # length=10, width=12, thickness=0.5, middlePosition=0, frontDistance=2.0

            obstackle = MeshedGeometry.anvilObstackle(
                length=_obstackleLength, width=_obstackleWidth, thickness=_thickness, middlePosition=_middlePosition, frontDistance=_frontDistance)
            # konwersja w ukłądzie współrzędnych dla SAMEJ przeszkody na układ właściwy dla pomieszczenia
            cornerFL = ms.Circle(
                Point(
                    length - obstackle["LFCircleMP"].x(), width/2.0+obstackle["LFCircleMP"].y()),
                _thickness/2.0, segments=16)
            cornerFR = ms.Circle(
                Point(
                    length - obstackle["RFCircleMP"].x(), width/2.0+obstackle["RFCircleMP"].y()), _thickness/2.0, segments=16)
            cornerBL = ms.Circle(
                Point(
                    length - obstackle["LBCircleMP"].x(), width/2.0+obstackle["LBCircleMP"].y()), _thickness/2.0, segments=16)
            cornerBR = ms.Circle(
                Point(
                    length - obstackle["RBCircleMP"].x(), width/2.0+obstackle["RBCircleMP"].y()), _thickness/2.0, segments=16)

            obstackle = cornerFL + cornerFR + cornerBL + cornerBR

            hasObstackle = True

        print("vest length eqals ", _vestibuleLength)

        vestibule = MeshedGeometry.vestibule(
            length=_length, width=_width, exitWidth=_exitWidth, shift=0, vestibuleLength=_vestibuleLength)

        if hasObstackle:
            self.mesh = ms.generate_mesh(
                domain-vestibule-obstackle, denisityMesh)
        else:
            self.mesh = ms.generate_mesh(domain-vestibule, denisityMesh)

        if compSettings["initialType"] == "rectangleInitial":
            self.initialExpression = MeshedGeometry.initialRectangle(12)

        if compSettings["initialType"] == "bellCurveExp":
            px = py = compSettings["calibration"]["width"]/2
            self.initialExpression = MeshedGeometry.initialBellCurveExp(
                compSettings["initialParams"]["maxValue"], middle=(px, py))

        if compSettings["initialType"] == "bellCurveInvSquare":
            px = py = compSettings["calibration"]["width"]/2
            self.initialExpression = MeshedGeometry.initialBellCurveInvSquare(
                compSettings["initialParams"]["maxValue"], middle=(px, py))

        compSettings["calibration"]["width"]

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
        print("hmin::", self.hmin, "hmaxx::", self.hmax)


class PhysicalModel(MeshedGeometry):
    def __init__(self, geoSettings, meshDensity):
        super().__init__(geoSettings, meshDensity)

        def isExit(x, on_boundary):
            return on_boundary and self.isExitCond(x)

        def isWall(x, on_boundary):
            return on_boundary and not self.isExitCond(x)

        self.wall = isWall
        self.exit = isExit
        self.WALL_CODE = 1111
        self.EXIT_CODE = 3333

        self.generateCustomDS()

    def generateCustomDS(self):
        class Wall(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not self.isExitCond(x)

        class Exit(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and self.isExitCond(x)

        mf = MeshFunction("size_t", self.mesh, 2)
        w = Wall()
        e = Exit()
        w.mark(mf, self.WALL_CODE)
        e.mark(mf, self.EXIT_CODE)
        self.DS = Measure("ds")(subdomain_data=mf)

    def isSetBoundaries(self):
        if hasattr(self, "wall") and hasattr(self, "exit"):
            return True

    def generateEikonalEqution(self):
        def num2fs(x, V):
            return interpolate(Expression(str(x), degree=2), V)

        def norm(potentialField, theta):
            dx0 = project(potentialField.dx(0))  # +num2fs(theta, self.V)
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

        # TODO do sprawdzenia i naprawienia jeszcze tutaj
        return grad(phi)

    def generateUnitFieldPhi(self, typeVectorField):
        if typeVectorField == "EIKONAL":
            UnitFieldPhi = self.generateEikonalEqution()
        self.unitFieldPhi = UnitFieldPhi
        return UnitFieldPhi

    @saveAsFile("unitFieldPhi.jpg")
    def plotUnitFieldPhi(self):
        plot(self.unitFieldPhi)
        plot(self.mesh)


class VelocityFunction:

    def __init__(self, settings):
        self.accuracy = settings["TaylorChainAccuracy"]
        self.VMAX = settings["vmax"]
        self.RHOMAX = settings["rhomax"]

    def taylorRawExp(self, x, zerro, one):
        def pow(x, n, one):
            result = one
            for i in range(n):
                result = result*x
            return result

        def invSilnia(n, one):
            if n == zerro:
                return one

            result = one
            for i in range(1, n+1):
                # TODO przetestuj
                result = result*i*one

            # TODO przetestuj
            return one/result

        result = zerro
        for i in range(self.accuracy):
            sign = pow(-one, i, 1)
            power = pow(x, 2*i, one)
            invS = invSilnia(i, one)
            result += sign*invS*power
            # print("{}\t{}\t{}\t{}".format(i, sign, invS, power))

        return result

    def Vfield(self, x, V):
        def num2fs(x, V):
            return interpolate(Expression(str(x), degree=2), V)

        sigma = self.RHOMAX/2.0
        invSigma = num2fs(1.0/sigma, V)
        zerro = num2fs(0.0, V)
        one = num2fs(1.0, V)
        fx = self.taylorRawExp(x*invSigma, zerro, one)
        return fx*num2fs(self.VMAX, V)

    def Vfield2(self, xOrigin):
        x = xOrigin.copy(deepcopy=True)
        nodalValues = x.vector().vec().array
        newNodalValues = [self.V(x) for x in nodalValues]
        x.vector().vec().array = newNodalValues
        return x

    def V(self, x):
        sigma = self.RHOMAX/2.0
        zerro = 0.0
        one = 1.0
        fx = self.taylorRawExp(x/sigma, zerro, one)
        return fx*self.VMAX


class NumericalModel:

    @staticmethod
    def num2fs(expression, V):
        return interpolate(expression, V)

    def __init__(self, PhysicalModel, velocityFuncSettings):
        self.PM = PhysicalModel
        self.V = PhysicalModel.V
        self.initialExpression = PhysicalModel.initialExpression
        self.test = TestFunction(PhysicalModel.V)
        self.normal = FacetNormal(PhysicalModel.mesh)
        self.sigma = 0.0001
        self.dt = 0.001

        self.rho = TrialFunction(PhysicalModel.V)
        self.rhoold = NumericalModel.num2fs(
            PhysicalModel.initialExpression, PhysicalModel.V)
        print("initial definition", type(self.rho))
        # TODO - automatycznie tutaj parametry mają być takie same jak wszędzie !!!!!!!!
        self.V_rhoold = VelocityFunction(
            velocityFuncSettings).Vfield2(self.rhoold)

    def defineEquation(self):
        sigma = self.sigma
        dt = self.dt
        rho = self.rho
        self.rhoold
        rhoold = self.rhoold
        test = self.test
        normed_phi = self.PM.unitFieldPhi
        n = self.normal
        WALL = self.PM.WALL_CODE
        EXIT = self.PM.EXIT_CODE
        DS = self.PM.DS
        V_rhoold = self.V_rhoold
        equation = {}
        equation["timeDer"] = (rho - rhoold)/dt*test*dx
        equation["adv"] = -rho*V_rhoold*dot(normed_phi, grad(test))*dx
        equation["diff"] = 2*sigma * dot(grad(rho), grad(test))*dx
        equation["diff"] = equation["diff"] + sigma*rho*div(grad(test))*dx

        equation["boundaryAdv"] = rhoold*test * \
            V_rhoold*dot(normed_phi, n)*DS(EXIT)

        equation["boundaryDiff"] = - sigma*rhoold * dot(grad(test), n)*DS(EXIT) \
            - sigma * dot(grad(rhoold), n)*test*DS(WALL)

        # TODO napiać to samo w skróconej formie - wykorzystujące właściwości słownika
        F = equation["timeDer"]
        F = F + equation["adv"]
        F = F + equation["diff"]
        F = F + equation["boundaryDiff"]
        F = F + equation["boundaryAdv"]

        print("equation definition", type(rho))

        self.F = F
        return F


class Simulation:

    SIMULATION_FOLDER = "steps"

# https://stackoverflow.com/questions/15861875/custom-background-sections-for-matplotlib-figure

    def colorMappedPlot(self, field):
        # M = 0.2
        # b = 200
        # a, c = int(b*M)+2, int(b*M)+2
        # new_min = M/(1+2*M)
        # new_max = (1+M)/(1+2*M)

        a, b, c = 10, 50, 10

        rho_max = self.velocityFunction.RHOMAX

        low = mpl.colors.LinearSegmentedColormap.from_list(
            "my", ["yellow",  "black"])
        # low = mpl.cm.get_cmap("brg_r")
        mid = mpl.cm.get_cmap("cubehelix")
        high = mpl.colors.LinearSegmentedColormap.from_list(
            "my", ["white", "red"])
        # high = mpl.cm.get_cmap("ocean")

        l = low(np.linspace(0, 1, a))
        m = mid(np.linspace(0, 1, b))
        h = high(np.linspace(0, 1, c))

        lmh = np.vstack((l, m, h))

        one_level = rho_max/b

        vmin = -one_level*a
        vmax = one_level*c+rho_max
        newNorm = Normalize(vmin=vmin, vmax=vmax)
        newCMap = mpl.colors.ListedColormap(lmh)

        p = plot(field, levels=np.linspace(vmin, vmax, a+b+c),
                 norm=newNorm, cmap=newCMap, extend="both")
        plt.colorbar(p, cmap=newCMap, fraction=0.07, pad=0.1,
                     shrink=5.0, orientation='horizontal')

        return newCMap

    @saveAsFile("mesh.jpg")
    def plotMesh(self):
        plot(self.numericalModel.PM.mesh)

    @saveAsFile("initial.jpg")
    def drawInitialCondition(self):
        func = NumericalModel.num2fs(self.numericalModel.initialExpression,
                                     self.numericalModel.V)
        plt.title("initial condition")
        self.colorMappedPlot(func)
        plt.legend()
        plt.show()

    def drawSimulationStep(self, numStep):
        # TODO zmodyfikować nazwę trochę by dorzucić dodatkowy parametr
        @saveAsFile("step{}.jpg".format(numStep))
        def drawStep(self):
            func = NumericalModel.num2fs(self.numericalModel.rhoold,
                                         self.numericalModel.V)
            plt.title("simulation step")
            self.colorMappedPlot(func)
            # plt.legend()
            plt.show()
        drawStep(self)

    def drawSimulationVStep(self, numStep):
        @saveAsFile("stepV{}.jpg".format(numStep))
        def drawVStep(self):
            func = NumericalModel.num2fs(self.numericalModel.V_rhoold,
                                         self.numericalModel.V)
            plt.title("simulation velocity field step")
            self.colorMappedPlot(func)
            # plt.legend()
            plt.show()
        drawVStep(self)

    # TODO implement courantNumber - wzór
    @saveAsFile("courantNumber.png")
    def drawCourantStability(self):
        pass

    @saveAsFile("velocityFunc.png")
    def visualise(self):
        X = np.linspace(0.0, self.velocityFunction.RHOMAX*1.1, 100)
        Y = [self.velocityFunction.V(x) for x in X]
        plt.plot(X, Y)
        plt.grid(True)
        plt.axvspan(self.velocityFunction.RHOMAX,
                    self.velocityFunction.RHOMAX*1.1, alpha=0.25)

    @saveAsFile("fieldVfunc.png")
    def visualiseField(self):
        self.colorMappedPlot(self.numericalModel.V_rhoold)
        plt.legend()

    @saveAsFile("DX.jpg")
    def drawDX(self):
        plt.title("krok przestrzenny")
        spatialStep = project(CellDiameter(self.mesh),  self.V)
        print("!!!!!!!!1", type(spatialStep))
        p = plot(spatialStep, cmap=mpl.cm.hot)
        plt.colorbar(p, orientation='horizontal')
        plt.legend()
        plt.show()

    def __init__(self, Settings):
        # numericalModel, velocityFunction, velocityFuncSettins, simName
        vfs = Settings["velocityFuncSettings"]
        vf = VelocityFunction(vfs)
        geoS = Settings["geoSettings"]
        nms = Settings["numModelSetings"]

        pm = PhysicalModel(geoS, geoS["meshDensity"])
        pm.generateMesh()
        pm.generateUnitFieldPhi(nms["typeVectorField"])
        nm = NumericalModel(pm, vfs)

        simName = Settings["simulationsettings"]["simNameFolder"]

        self.settings = Settings
        self.numericalModel = nm
        self.velocityFuncSettins = vfs
        self.velocityFunction = vf
        self.V = nm.V
        self.mesh = nm.PM.mesh

        self.simulationSettings(Settings)

        if os.path.isdir(simName):
            shutil.rmtree(simName)

        os.mkdir(simName)
        os.chdir(simName)

    def simulationSettings(self, simSettings):
        firstStep = simSettings["simulationsettings"]["firstStep"]
        maxStep = simSettings["simulationsettings"]["maxStep"]
        frequency = simSettings["simulationsettings"]["frequency"]
        self.displayedSteps = range(firstStep, maxStep, frequency)
        self.maxStep = maxStep

    def simulate(self, makeSimulation=True):

        Settings = self.settings
        self.saveSimulationState(Settings)
        if os.path.isdir(Simulation.SIMULATION_FOLDER):
            shutil.rmtree(Simulation.SIMULATION_FOLDER)
        os.mkdir(Simulation.SIMULATION_FOLDER)
        os.chdir(Simulation.SIMULATION_FOLDER)

        self.simulateLoop(Settings)
        os.chdir("..")

    def simulateLoop(self, Settings):
        F = self.numericalModel.defineEquation()
        currentRho = Function(self.V)

        for i in range(self.maxStep):

            if i in self.displayedSteps:
                print("Solving equation with step {}".format(i))
                self.drawSimulationStep(i)
                self.drawSimulationVStep(i)

            a, L = lhs(F), rhs(F)
            solve(a == L, currentRho)
            self.numericalModel.rhoold.assign(currentRho)
            self.numericalModel.V_rhoold = VelocityFunction(
                self.velocityFuncSettins).Vfield2(self.numericalModel.rhoold)

    def saveSimulationState(self, settings):
        mainCode = open("../main.py", "rt").read()
        utilsCode = open("../utils.py", "rt").read()
        open("settings.json", "wt").write(json.dumps(settings))
        open("utils.py", "wt").write(mainCode)
        open("main.py", "wt").write(utilsCode)


class ManySimulationsManagement:
    def __init__(self, coreSetting):
        self.coreSetting = coreSetting

    def paramVariability(self, paramName, paramValues):
        import copy
        self.paramName = paramName
        self.paramValues = paramValues
        self.numberSettings = len(paramValues)
        self.manySettings = np.array([])
        for value in paramValues:
            setting = copy.deepcopy(self.coreSetting)
            a = paramName[0]
            b = paramName[1]
            setting[a][b] = value
            setting["simulationsettings"]["simNameFolder"] += "_p{}v{}".format(
                b, value)
            self.manySettings = np.append(self.manySettings, setting)

    def setting2file(self):
        import copy
        self.groupedSettings = copy.deepcopy(self.coreSetting)
        a = self.paramName[0]
        b = self.paramName[1]
        self.groupedSettings[a][b] = self.paramValues

    def myJob(self, setting):
        sim = Simulation(setting)

        sim.simulate(makeSimulation=False)
        print(setting)

    def simulateMany(self):
        from multiprocessing import Pool
        if os.path.isdir(self.coreSetting["simulationsettings"]["simNameFolder"])
        poolWorker = Pool(processes=self.numberSettings)
        poolWorker.map(self.myJob, self.manySettings)
