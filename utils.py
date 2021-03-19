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

from geometries import *

import numpy as np



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
        sigma = 1
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
        plt.show()


class VelocityFunction:

    def __init__(self, settings):
        self.accuracy = settings["TaylorChainAccuracy"]
        self.VMAX = settings["vmax"]
        self.RHOMAX = settings["rhomax"]
        self.computationVelovityMethod = settings["computationVelovityMethod"]

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
    
    

    def Vfield1(self, x, V):
          
        def num2fs(x, V):
            return interpolate(Expression(str(x), degree=2), V)
        
        sigma = self.RHOMAX/2.0
        invSigma = num2fs(1.0/sigma, V)
        zerro = num2fs(0.0, V)
        one = num2fs(1.0, V)

        def pow(x, n):
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
                result = result*num2fs(i, V)*one
            return one/result

        result = zerro
        for i in range(self.accuracy):
            sign = pow(-one, i)
            power = pow(x*invSigma, 2*i)
            invS = invSilnia(i, one)
            result += sign*invS*power

        return result*num2fs(self.VMAX, V)

    def Vfield2(self, xOrigin):
        x = xOrigin.copy(deepcopy=True)
        nodalValues = x.vector().vec().array
        newNodalValues = [self.V(x) for x in nodalValues]
        x.vector().vec().array = newNodalValues
        return x
    
    def Vfield(self, f, V):
        if self.computationVelovityMethod == "vector":
            return self.Vfield2(f)
        if self.computationVelovityMethod == "num2fs":
            return self.Vfield1(f, V)

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

    def __init__(self, PhysicalModel, velocityFuncSettings, numModelSetings):
        self.PM = PhysicalModel
        self.V = PhysicalModel.V
        self.initialExpression = PhysicalModel.initialExpression
        self.test = TestFunction(PhysicalModel.V)
        self.normal = FacetNormal(PhysicalModel.mesh)
        self.sigma = numModelSetings["sigma"]
        self.dt = numModelSetings["dt"]

        self.rho = TrialFunction(PhysicalModel.V)
        self.rhoold = NumericalModel.num2fs(
            PhysicalModel.initialExpression, PhysicalModel.V)
        print("initial definition", type(self.rho))
        # TODO - automatycznie tutaj parametry mają być takie same jak wszędzie !!!!!!!!
        self.V_rhoold = VelocityFunction(
            velocityFuncSettings).Vfield(self.rhoold, self.V)

    def defineEquation(self, exitBCtype):
        sigma = self.sigma
        dt = self.dt
        dtInv = 1.0/dt
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
        
        setting = "BasicAdvection"
        if setting == "KAMGAadvectionDiffusion":
            equation = {}
            equation["timeDer"] = rho*test*dtInv*dx - rhoold*test*dtInv*dx
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
            
        if setting == "BasicAdvection":
            equation = {}
            equation["timeDer"] = rho*test*dtInv*dx - rhoold*test*dtInv*dx
            equation["adv"] = -rho*dot(normed_phi, grad(test))*dx + rhoold*test*dot(normed_phi, n)*DS(WALL)
            F = equation["timeDer"] + equation["adv"]
        
        if setting == "BasicAdvectionDiffusion":
            equation = {}
            equation["timeDer"] = rho*test*dtInv*dx - rhoold*test*dtInv*dx
            equation["adv"] = -rho*dot(normed_phi, grad(test))*dx + rhoold*test*dot(normed_phi, n)*DS(WALL)
            equation["diff"] = sigma*dot(grad(rho), grad(test))*dx - sigma*rhoold * dot(grad(test), n)*DS(WALL)
            
                

        print("equation definition", type(rho))

        self.F = F
        return F


class Simulation:
# https://stackoverflow.com/questions/15861875/custom-background-sections-for-matplotlib-figure

    def colorMappedPlot(self, field):
        # M = 0.2
        # b = 200
        # a, c = int(b*M)+2, int(b*M)+2
        # new_min = M/(1+2*M)
        # new_max = (1+M)/(1+2*M)

        a, b, c = 100, 500, 100

        rho_max = self.velocityFunction.RHOMAX

        low = mpl.colors.LinearSegmentedColormap.from_list(
            "my", ["yellow",  "black"])
        # low = mpl.cm.get_cmap("brg_r")
        mid = mpl.cm.get_cmap("nipy_spectral")
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

        
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        
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
            
            f = File("step_paraView{}.pvd".format(numStep))
            f << func
            
            plt.title(r'simulation step')
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
        def num2fs(x, V):
            return interpolate(Expression(str(x), degree=2), V)

        V = self.numericalModel.V
        dxx = project(CellDiameter(self.mesh),  V)
        v_max = self.velocityFunction.VMAX
        kappa = self.numericalModel.sigma
        dt = num2fs(self.numericalModel.dt, V)

        c1 = 1/(4+4*np.sqrt(2))
        c2 = 1/(12+6*np.sqrt(2))
        c1 = num2fs(c1, V)
        c2 = num2fs(c2, V)
        V15 = num2fs(1.5, V)
        V2 = num2fs(2.0, V)
        V4 = num2fs(4.0, V)
        V05 = num2fs(0.5, V)
        Vkappa = num2fs(kappa, V)
        Vvmax = num2fs(v_max, V)
        
        alfa_opt = ((Vvmax*Vvmax)/V4 + Vkappa*c1/(c2*dxx*dxx))**V05
        czlon1 = alfa_opt/(c1)
        czlon2 = v_max**V2/(V4*alfa_opt*c1)
        czlon3 = Vkappa/(alfa_opt*c2*dxx*dxx)
        czlon123 = czlon1+czlon2+czlon3

        condition = V15*(dt/dxx)*(czlon123)

        p = plot(condition, cmap=mpl.cm.hot)
        plt.colorbar(p, orientation='horizontal')
        plt.legend()
        plt.show()
        return condition

    @saveAsFile("simpleCourantNumber.png")
    def drawSimpleCourantStability(self):
        def num2fs(x, V):
            return interpolate(Expression(str(x), degree=2), V)

        V = self.numericalModel.V
        spatialStep = project(CellDiameter(self.mesh),  V)
        dt = num2fs(self.numericalModel.dt, V)

        plt.title(r'$\frac{\Delta{t}}{\Delta{x}}$')
        p = plot(dt/spatialStep, cmap=mpl.cm.hot)
        plt.colorbar(p, orientation='horizontal')
        plt.legend()
        plt.show()
        return dt/spatialStep
    
    def computeSimpleCourantStability(self):
        dt = self.numericalModel.dt
        minSpatialStep = self.mesh.hmin()
        maxSpatialStep = self.mesh.hmax()
        minValue = dt/minSpatialStep
        maxValue = dt/maxSpatialStep

        return {"minValue": minValue, "maxValue": maxValue}

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
        plt.title("krok przestrzenny $h = \Delta{x}$")
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
        nm = NumericalModel(pm, vfs, nms)

        simName = Settings["simulationsettings"]["simNameFolder"]

        self.nms = nms
        self.PM = pm
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
        minTime = simSettings["simulationsettings"]["minTime"]
        maxTime = simSettings["simulationsettings"]["maxTime"]
        minStep = simSettings["simulationsettings"]["minStep"]
        maxStep = simSettings["simulationsettings"]["maxStep"]
        frequency = simSettings["simulationsettings"]["frequency"]

        deltat = self.settings["numModelSetings"]["dt"]
        
        if not Simulation.STEPPED:
            self.minStep = int(minTime/deltat)
            self.maxStep = int(maxTime/deltat)+1
            self.frequency = int(frequency/deltat)
        else:
            self.minStep = minStep
            self.maxStep = maxStep  
            
        self.displayedSteps = range(self.minStep, self.maxStep, frequency)


    def simulate(self, makeSimulation=True):
        Settings = self.settings
        self.saveSimulationState(Settings)
        if os.path.isdir(Simulation.SIMULATION_FOLDER):
            shutil.rmtree(Simulation.SIMULATION_FOLDER)
        os.mkdir(Simulation.SIMULATION_FOLDER)
        os.chdir(Simulation.SIMULATION_FOLDER)
        values = self.simulateLoop(Settings, makeSimulation)
        os.chdir("..")
        return values
        
    def removeNegatives(self, func):
        fvals = func.vector().get_local()     # temporary copy of function value arrays
        fvals[fvals < 0.0] = 0.0               # numpy syntax for overwriting negative values
        func.vector().set_local(fvals)

    def simulateLoop(self, Settings, makeSimulation):
        F = self.numericalModel.defineEquation(self.nms["exitBCtype"])
        currentRho = Function(self.V)

        indexes = []
        stability = []
        integral = []
        
        for i in range(self.maxStep):

            if i in self.displayedSteps and Simulation.VISUALISE_STEPS:
                print("Solving equation with step {}".format(i))
                self.drawSimulationStep(i)
                self.drawSimulationVStep(i)

            a, L = lhs(F), rhs(F)
            if self.nms["usedDirichletBCmethod"]:
                bc = DirichletBC(self.V, Constant(0.0), self.PM.exit)
                solve(a == L, currentRho, bc)
                
            if not self.nms["usedDirichletBCmethod"]:
                solve(a == L, currentRho)

            self.numericalModel.rhoold.assign(currentRho)
            self.numericalModel.V_rhoold = VelocityFunction(
                self.velocityFuncSettins).Vfield2(self.numericalModel.rhoold)
            
            indexes.append(i*Settings["numModelSetings"]["dt"])
            stability.append(assemble(project(currentRho*currentRho, self.V)*dx(self.mesh)))
            integral.append(assemble(project(currentRho, self.V)*dx(self.mesh)))
        ret = {}
        ret["indexes"] = indexes
        ret["stability"] = stability
        ret["integral"] = integral
        ret["Courant"] = self.computeSimpleCourantStability()
        return ret

    def saveSimulationState(self, settings):
        # TODO to miejsce jest trochę niewygodne i mało elastyczne określenie ścieże k
        mainCode = open("../../main.py", "rt").read()
        utilsCode = open("../../utils.py", "rt").read()
        geometries = open("../../geometries.py", "rt").read()
        open("settings.json", "wt").write(json.dumps(settings))
        open("utils.py", "wt").write(mainCode)
        open("main.py", "wt").write(utilsCode)
        open("geometries.py", "wt").write(geometries)


class ManySimulationsManagement:
    def __init__(self, coreSetting):
        self.coreSetting = coreSetting

    def paramVariability(self, paramName, paramValues):
        import copy
        self.paramName = np.array(paramName)
        self.paramValues = np.array(paramValues)
        self.numberSettings = len(paramValues)
        self.manySettings = np.array([])
        for value in paramValues:
            setting = copy.deepcopy(self.coreSetting)
            a = paramName[0]
            b = paramName[1]
            setting[a][b] = value
            setting["simulationsettings"]["simNameFolder"] = "{}_{}".format(
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
        if Simulation.VISUALISE_STEPS:
            print("DRAWED 1!!! wih {}".format(setting["numModelSetings"]["dt"]))
            sim.numericalModel.PM.plotUnitFieldPhi()
            print("DRAWED 2!!! wih {}".format(setting["numModelSetings"]["dt"]))
            sim.drawDX()
            print("DRAWED 3!!! wih {}".format(setting["numModelSetings"]["dt"]))
            sim.plotMesh()
            print("DRAWED 4!!! wih {}".format(setting["numModelSetings"]["dt"]))
            sim.drawCourantStability()
            print("DRAWED 5!!! wih {}".format(setting["numModelSetings"]["dt"]))
            sim.drawSimpleCourantStability()
            print("DRAWED 6!!! wih {}".format(setting["numModelSetings"]["dt"]))
        value = sim.simulate(makeSimulation=False)
        print("!! RETURNED VALUE IS", value)
        return value

    def simulateMany(self):
        ensSimName = self.coreSetting["simulationsettings"]["simNameFolder"]
        from multiprocessing import Pool
        if os.path.isdir(ensSimName):
            shutil.rmtree(ensSimName)
        os.mkdir(ensSimName)
        os.chdir(ensSimName)
        poolWorker = Pool(processes=self.numberSettings)
        self.raw = poolWorker.map(self.myJob, self.manySettings)
        print("SIM RESULTS ARE", self.raw)
        os.chdir("..")
        

    def drawSummary(self, typeName='stability'):
        extractedDatas = {}
        label = self.paramName[-1]
        indexes = range(len(self.paramValues))
        simName = self.coreSetting["simulationsettings"]["simNameFolder"]

        fig, ax = plt.subplots()
        
        for i, pv, content in reversed(list(zip(indexes, self.paramValues,  self.raw))):
            t = content['indexes']
            extractedDatas = content[typeName]
            etiquete = label+"="+str(pv)
            plt.title(typeName)
            plt.plot(t, extractedDatas, label=etiquete)
        plt.legend()

        numPeople = self.coreSetting["geoSettings"]["initialParams"]["peopleGroups"]["group1"]["numPeople"]
        #plt.ylim([-numPeople, 2*numPeople])
        plt.ylabel(typeName)
        if Simulation.STEPPED:
            plt.xlabel("steps of simulation")
        else:
            plt.xlabel("time of simulation")             
                        
        plt.ylabel("simulation steps")
        
        if typeName=='stability':
            ax.set_yscale('log')   
        
        plt.grid(True)
        plt.savefig(os.path.join(simName, typeName))
        plt.clf()   
        
    def drawSummaryAll(self):  
        self.drawSummary(typeName='stability')
        self.drawSummary(typeName='integral')
        
    def minmaxCflDraw(self):
        simName = self.coreSetting["simulationsettings"]["simNameFolder"]
        
        labels = self.paramValues
        width = 0.2
        
        indexes = np.arange(len(self.paramValues))
        minValues = list(range(len(self.paramValues)))
        maxValues = list(range(len(self.paramValues)))
        
        for i, pv, content in reversed(list(zip(indexes, self.paramValues,  self.raw))):
            minValues[i] = content["Courant"]["minValue"]
            maxValues[i] = content["Courant"]["maxValue"]
        
        plt.ylabel("Courant")                        
        plt.xlabel("simulation steps")
        fig, ax = plt.subplots()
        rects1 = ax.bar(indexes - width/2, minValues, width, label='minCourant')
        rects2 = ax.bar(indexes + width/2, maxValues, width, label='maxCourant')
        ax.set_xticklabels(labels)
        ax.set_xticks(indexes)
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        autolabel(rects1)
        autolabel(rects2)
        plt.title("")
        plt.savefig(os.path.join(simName, "courant"))
        plt.clf()   
        