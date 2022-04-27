from dolfin import *
import mshr as ms
import ufl as uf
import numpy as np
from sys import *
import os
import shutil
import multiprocessing
from geometries import *
import pandas as pd
from datetime import datetime, timedelta
#from scipy import interpolate

from multiprocessing import Pool

from settings import   Settings
from logs import *
from courant import *




    


class PhysicalModel(MeshedGeometry):
    def __init__(self, globalSettings):
        dens = globalSettings.mainSim.meshDensity
        super().__init__(globalSettings.geo, densityMesh=dens)

        #Wyznaczenie tego fragmentu brzzegu, który jest wyjściem
        # isExitCond - funkcja wyznaczająca wszystkie wartości w RxR
        # należące do wyjścia (czyli wszystkie wartości X=długość+korytarz)
        def isExit(x, on_boundary):
            return on_boundary and self.isExitCond(x)

        #Wyznaczenie tego fragmentu brszegu, który jest ścianą
        def isWall(x, on_boundary):
            return on_boundary and not self.isExitCond(x)

        self.globalSettings = globalSettings
        self.ID = globalSettings.ID
        # funkcje zachowane jako zmienne
        self.wall = isWall
        self.exit = isExit
        #unikalny identyfikator każdego z typów brzegu
        self.WALL_CODE = 1
        self.EXIT_CODE = 2

        isExitCondHere = self.isExitCond

        class Wall(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not isExitCondHere(x)

        class Exit(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and isExitCondHere(x)


        mf = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        mf.set_all(0)
        w = Wall()
        e = Exit()
        w.mark(mf, self.WALL_CODE)
        e.mark(mf, self.EXIT_CODE)
        #self.DS = Measure("ds")(subdomain_data=mf)
        #parametr podobny do "ds" z tą różnicą, że ma możliwość zorzóznienia brzegu na 2 typy
        self.DS = Measure('ds', domain=self.mesh, subdomain_data=mf)

    def isSetBoundaries(self):
        if hasattr(self, "wall") and hasattr(self, "exit"):
            return True

    def eikoEquationFirst(self):

        def myLn(xOrigin):
            x = xOrigin.copy(deepcopy=True)
            print("LOGARITHM", type(xOrigin))
            nodalValues = x.vector().vec().array
            newNodalValues = [np.log(x) for x in nodalValues]
            x.vector().vec().array = newNodalValues
            return project(x)

    

        # współczynnik delta ze wzoru (1.1) z publikacji Modelling Crowd Dynamics
        eikoDelta = self.globalSettings.eikonal.eikoDelta
        # funkcja skalarna 
        u = TrialFunction(self.V)

        #funkcja testowa w Modelling Crowd Dynamics wyrażana jako greckie NJU
        v = TestFunction(self.V)

        #równanie (4.4) z Modelling Crowd Dynamics
        F = u*v*dx + eikoDelta*eikoDelta*dot(grad(u), grad(v))*dx
        a, L = lhs(F), rhs(F)

        #Warunek brzegowy Dirichleta na wyjściu - trzeci warunek z (4.3)  Modelling Crowd Dynamics
        bc = DirichletBC(self.V, Constant(1.0), self.exit)
        u = Function(self.V)
        solve(a == L, u, bc)

        #odwrócone przekształcenie Hopf-Colea z wzoru (4.2) Modelling Crowd Dynamics
        #phi = -uf.ln(u)*eikoDelta
        phi = -myLn(u)*eikoDelta

        #zmienna wprowadzona w ramach przekształcenia Hopf-Colea
        self.u = u

        return phi

    def eikoEquationSecond(self):
        # Dla tego równania nie ma wpływu
        eikoDelta = self.globalSettings.eikonal.eikoDelta
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        f = Constant(1.0)
        F = -f*v*dx + eikoDelta*inner(grad(u), grad(v))*dx
        a, L = lhs(F), rhs(F)
        bc = DirichletBC(self.V, Constant(0.0), self.exit)
        uResult = Function(self.V)
        solve(a == L, uResult, bc)
        return uResult

    def eikoEquationThird(self):
        u, v = TrialFunction(self.V), TestFunction(self.V)
        phi = Function(self.V)
        f = Constant(1.0)
        bc = DirichletBC(self.V, Constant(1.0), self.exit)
        # Eikonal equation with stabilization
        eps = Constant(self.mesh.hmax()/10)
        F = sqrt(inner(grad(phi), grad(phi)))*v*dx - inner(f, v)*dx + eps*inner(grad(phi), grad(phi))*v*dx
        solve(F == 0, phi, bc)

        return phi

        # J = derivative(F, phi, u)
        # problem = NonlinearVariationalProblem(F, phi, bc, J)
        # solver  = NonlinearVariationalSolver(problem)
        # solver.solve()

    #funkcja zwracająca pole wektorowe "w" 
    def generateEikonalEqution(self):

        #współczynnik poprawiający normalizację dla bardzo małych wartości modułu.
        #Jest we wzorze (1.3) z publikacji Modelling Crowd Dynamics
        theta = self.globalSettings.eikonal.theta

        def num2fs(x, V):
            return interpolate(Expression(str(x), degree=2), V)

        def norm2(potentialField, theta):
            dx0 = potentialField.dx(0)
            dx1 = potentialField.dx(1)
            #unnormed_grad_phi = project(grad(potentialField))
            module = sqrt(dx0*dx0+dx1*dx1)#+num2fs(theta, self.V)**2)
            return module

        def normVectorField_norm(vectorField):
            #x = vectorField#.copy(deepcopy=True)
            #print("vectorField", type(vectorField))
            #xx = project(x, VectorFunctionSpace(mesh, 'P', 2))

            W = VectorFunctionSpace(self.mesh, 'P', 1)
            w = Function(W)
            xx = project(vectorField, W)
            # print("vectorField22", type(xx))
            # nodalValues = xx.vector().vec().array
            # print(type(xx), dir(xx))
            # print(len(nodalValues), nodalValues)
            # print("norm(xx)", norm(xx))
            dx0, dx1 = xx.split(deepcopy=True)

            dx0vec = dx0.vector().vec().array
            dx1vec = dx1.vector().vec().array
            #print("dx0vec!!!", len(dx0vec), dx0vec)
            answer = dx0.copy(deepcopy=True)
            answer.vector().vec().array = np.sqrt(dx0vec*dx0vec + dx1vec*dx1vec)
            #print("dx0!!!!!!!", type(dx0))
            return answer


        def normVectorField_vector(vectorField):
            #x = vectorField#.copy(deepcopy=True)
            #print("vectorField", type(vectorField))
            #xx = project(x, VectorFunctionSpace(mesh, 'P', 2))

            W = VectorFunctionSpace(self.mesh, 'P', 1)
            w = Function(W)
            xx = project(vectorField, W)
            # print("vectorField22", type(xx))
            # nodalValues = xx.vector().vec().array
            # print(len(nodalValues), nodalValues)
            # print("norm(xx)", norm(xx))
            return project(sqrt(inner(xx, xx)), self.V) 



        if self.globalSettings.eikonal.solverType == 1:
            phi = self.eikoEquationFirst()
        elif self.globalSettings.eikonal.solverType == 2:
            phi = self.eikoEquationSecond()
        elif self.globalSettings.eikonal.solverType == 3:
            phi = self.eikoEquationThird()
        else:
            pass

        gradPhi = -grad(phi)


        #zwielokrotnij operację:
        unitFieldPhi00 = gradPhi / normVectorField_norm(gradPhi)
        unitFieldPhi11 = unitFieldPhi00 / normVectorField_norm(unitFieldPhi00)
        unitFieldPhi22 = unitFieldPhi11 / normVectorField_norm(unitFieldPhi11)


        #pole skalarne będące rozwiązaniem równania (1.1)
        # z fizycznego punktu widzenia pole potencjału, 
        # w którym każda wartość odzwierciedla najkrótszą odległość do wyjścia
        self.phi0 = phi

        #pole skalarne będące normą każdego wektora z pola wektorowego
        self.normofphi = normVectorField_norm(gradPhi)#project(norm2(phi, theta=theta), self.V)
        
        if self.globalSettings.eikonal.isNormalizedVectorField:
            # pole wektorowe wektorów z modułem jeden
            # z równania (1.2).  Modelling Crowd Dynamics. Wyrażone jako "w" 
            self.unitFieldPhi = unitFieldPhi22#norm2(phi, theta=theta)
        else:
            #przekształcenia opisane w pracy w rozdziale 5.3 Druga wersja
            phiFirst = self.eikoEquationFirst()
            gradPhiFirst = -grad(phiFirst)
            self.normofphi = normVectorField_norm(gradPhiFirst)
            # pole wektorowe wektorów z modułem jeden
            # z równania (1.2), tylko bez normalizacji
            if self.globalSettings.eikonal.solverType == 2:
                self.unitFieldPhi = unitFieldPhi22*self.normofphi
            elif self.globalSettings.eikonal.solverType == 1:
                self.unitFieldPhi = gradPhi
            elif self.globalSettings.eikonal.solverType == 3:
                self.unitFieldPhi = gradPhi


# Implementacja wzorów z rozdziałiu 4.8 Definicja v(rho)
class VelocityFunction:

    def __init__(self, vSet, isCorrected=False):
        self.vSet=vSet
        self.accuracy = vSet.TaylorChainAccuracy
        self.VMAX = vSet.vmax
        self.RHOMAX = vSet.rhomax
        self.normRHOMAX = vSet.rhomax
        self.isCorrected = isCorrected
                

    def taylorRawExp(self, x, zerro, one):

        def cut_transform(x, result, zerro, one):
            #print(type(x))
            #print(type(result))
            if type(x) is np.ndarray and type(result) is int:
                result = np.array([result]*len(x))

            if type(x) is np.ndarray and type(result) is np.ndarray:
                newResult = np.copy(result)
                newResult[x<zerro] = one
                newResult[result<zerro] = zerro
                #print("X", x, "RES", newResult)
                return newResult
            else:
                if x < zerro:
                    return one
                elif result < zerro:
                    return zerro
                else:
                    return result

        if self.accuracy is None:
            return cut_transform(x, one-x, zerro, one)

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
        #pętla obliczjąca wzór (4.9) z rozdziałiu 4.8 Definivja v(rho)
        for i in range(self.accuracy):
            sign = pow(-one, i, 1)
            power = pow(x, 2*i, one)
            invS = invSilnia(i, one)

            if self.isCorrected:
                correction = 2*i + one
                result += sign*correction*invS*power
            else:
                result += sign*invS*power

        return cut_transform(x, result, zerro, one)
    


    def Vfield(self, xOrigin, Vs):
        def num2fs(x, V):
            return interpolate(Expression(str(x), degree=2), V)
        x = xOrigin.copy(deepcopy=True)
        nodalValues = x.vector().vec().array
        newNodalValues = [self.V(x) for x in nodalValues]
        x.vector().vec().array = newNodalValues
        x = conditional(lt(xOrigin, num2fs(0.0, Vs)), num2fs(self.VMAX, Vs), x)
        x = conditional(gt(xOrigin, num2fs(self.RHOMAX, Vs)), num2fs(0.0, Vs), x)  
        return project(x)
    


    def computeSigma(self):
        if self.accuracy is None:
            #return self.RHOMAX
            return 1.0

        stałe = {
            2: 1.0,
            4: 1.2634,
            6: 1.47669,
            8: 1.66102
        }
        if self.accuracy in stałe.keys():
            return stałe[self.accuracy]
        else:
            return 1.66102

    #wzór (4.11) z rozdziałiu 4.8 Definivja v(rho)
    def V(self, x, rescaleSigma=True):
        sigma = self.computeSigma()
        zerro = 0.0
        one = 1.0
        if rescaleSigma:
            newX = x*sigma/self.RHOMAX
        else:
            newX = x/self.RHOMAX

        unnormedResult = self.taylorRawExp(newX, zerro, one)
        return unnormedResult*self.VMAX  


class NumericalModel:

    def __init__(self, physicalModel):
        self.PM = physicalModel
        self.V = physicalModel.V
        self.test = TestFunction(physicalModel.V)
        self.normal = FacetNormal(physicalModel.mesh)
        self.rho = TrialFunction(physicalModel.V)
        self.rhoold = self.generateInitialCondition()
        self.V_rhoold = VelocityFunction(self.PM.globalSettings.velocity).\
            Vfield(self.rhoold, self.V)

    @staticmethod
    def num2fs(expression, V):
        return interpolate(expression, V)


    # realizacja równania (4.8) z rozdziału 4.7 Warunek oczątkowy 
    def generateInitialCondition(self):
        N = self.PM.globalSettings.initialCondition.numIterations
        deltaT = self.PM.globalSettings.initialCondition.deltaT
        initialParams = self.PM.globalSettings.geo.initialParams
        rhomax = self.PM.globalSettings.initialCondition.rhomax

        initCondObj = NotRegularizedInitialCondition(initialParams)
        self.initialCondition =\
        NumericalModel.num2fs(initCondObj\
        .ExpressionMultipleGroup(rhomax), self.V)
    
        rhoold = self.initialCondition.copy(deepcopy=True)
        #Zmienna pomocnicza zawierająca najbardziej aktualną ...
        #wartość funkcji skalarnej rho
        currentRho = Function(self.V)

        #zmmienna głównia równania. Wyrażone jako u
        rho = TrialFunction(self.V)
        #zmmienna głównia równania. Wyrażone jako v   
        test = TestFunction(self.V)

        # funkcja zwracająca cały brzeg
        def boundary(x, on_boundary):
            return on_boundary

        F = rho*test*dx + deltaT*dot(grad(rho), grad(test))*dx - rhoold*test*dx

        a, L = lhs(F), rhs(F)
        for i in range(N):
            solve(a == L, currentRho, DirichletBC(self.V, Constant(0), boundary))
            rhoold.assign(currentRho)

        targetAmount = initCondObj.SurfaceMultipleGroup()*rhomax
        actualAmount = assemble(project(rhoold, self.V)*dx(self.PM.mesh))
        print(targetAmount, actualAmount)
        self.regularisedInitialCondition = project(rhoold*(targetAmount/actualAmount))
        return project(rhoold*(targetAmount/actualAmount))



    # zdefiniowanie głłównego rówania (4.4) z rodziału "Równanie adwekcji-dyfuzji"
    def defineEquation(self):

        #współczynnik regularyzacji głównego równania adwekcji-dyfuzji
        kappa = self.PM.globalSettings.mainSim.kappa

        #krok czasowy pomiedzy sąsiadującymi klatkami symulacji
        dt = self.PM.globalSettings.mainSim.dt
        invDt = 1/dt
        #ZASADA OGÓLNA
        # rho    - bieżący krok symulacji
        # rhoold - poprzedni krok symulacji 
        #       (chyba że nie ma poprzedników to warunek początkowy)

        #zmienna przechowująca wartość funkcji z poprzedniego kroku symulacji.
        #chyba, że jest początek symulacji to przechowuje warunek początkowy
        rhoold = self.rhoold

        #funkcja próbna - zmienna przechowująca niewiadomą równania
        rho = self.rho

        #funkcja testowa
        test = self.test


        # pole wektorowe wektorów z modułem prawie jeden
        # z równania (1.2).  Modelling Crowd Dynamics. Wyrażone jako "w"
        #ale pod warunkiem zaznaczenia opcji eikonal.isNormalizedVectorField jako True
        normed_phi = self.PM.unitFieldPhi

        #wektor prostopadły do brzegu
        n = self.normal         

        WALL = self.PM.WALL_CODE
        EXIT = self.PM.EXIT_CODE
        DS = self.PM.DS
        V_rhoold = self.V_rhoold

        isBasicWeakForm = self.PM.globalSettings.mainSim.basicWeakForm
        BCtype = self.PM.globalSettings.mainSim.BCtype

        #wymieniam człony niebrzegowe
        # V_rhoold sprawia, że schemat staje się pół-zamknięty
        equation = {}

        equation["timeDer"] = rho*test*invDt*dx - rhoold*test*invDt*dx
        equation["adv"] = -rho*V_rhoold*dot(normed_phi, grad(test))*dx
        if isBasicWeakForm:
            try:
                equation["diff"] = kappa*dot(grad(rho), grad(test))*dx
            except Exception as e:
                pass
        else:
            equation["diff"] = 2*kappa*dot(grad(rho), grad(test))*dx
            equation["diff"] = equation["diff"] + kappa*rho*div(grad(test))*dx

        #zsumowanie wszystkich członów 
        #w celu sformuowania finalnego równania o postaci słabej
        F = equation["timeDer"]
        F = F + equation["adv"]
        F = F + equation["diff"]


        # potencjalnie możliwy człon brzegowy na przestrzeni DELTA OMEGA adwekcji 
        equation["boundaryAdv"] = rhoold*test*V_rhoold*dot(normed_phi, n)
        # potencjalnie możliwy człon brzegowy na przestrzeni DELTA OMEGA dyfuzji
        # wprawdze equation["boundaryDiff2"] jest nieużywane ale dla formalności umieściłem
        equation["boundaryDiff2"] = -kappa*dot(grad(rhoold), n)*test
        if not isBasicWeakForm:
            equation["boundaryDiff1"] = -kappa*dot(grad(test), n)*rhoold        
            

        if BCtype == "Neumann":
            F = F + equation["boundaryAdv"]*DS(EXIT)
            if not isBasicWeakForm:
                F = F + equation["boundaryDiff1"]*DS(WALL)
                F = F + equation["boundaryDiff1"]*DS(EXIT)

        if BCtype == "Dirichlet":
            # w obu wariantach równania nie uwzględniamy w ogóle członów brzegowych
            pass

        if BCtype == "D-N":
            F = F + equation["boundaryAdv"]*DS(EXIT)
            if not isBasicWeakForm:
                F = F + equation["boundaryDiff1"]*DS(EXIT)


        self.F = F
        return F

#TODO czy zrobić coś takiego jak parameter calculator - taką klasę ??

#utwórz katalog o nazwie "folderName" i wejdź (lub nie) do tego katalogu
def reMakeSimFolder(folderName, isChdir=True):
    if os.path.isdir(folderName):
        shutil.rmtree(folderName)              
    os.mkdir(folderName)
    if isChdir:
        os.chdir(folderName)

class Simulation:
    def __init__(self, settings, frequency=50):

        #ustalenie brzegu geometrii w dziedziczonej klasie MeshedGeometry
        #wprowadzenie zmiennych w celu rozróżnienia ścian i wyjścia ewakuacyjnego w dalszym użyciu
        pm = PhysicalModel(settings)

        #funkcja pochodząca z dziedziczonej klasy MeshedGeometrii generująca mesh - dyskretyzację
        pm.generateMesh(settings.mainSim.polynomialDegree)

        #funkcja dokładająca do instniejących już zmiennych pole wektorów jednostkowych
        pm.generateEikonalEqution()

        #PRZEKAZYWANIE ZMIENNYCH W CELU ZAPAMIĘTANIA STANU PM
        self.PM = pm

        #przekazanie zmiennej przechowującej wszystkie zmienne (współczynniki jednoznacznie wyznaczające końcowy wynik i zachowanie się symulacji)
        self.settings = settings

        #dołożenie do dotychczasowych informacji warunku początkowego (zmienna self.rhoold)
        #oraz skonstruowanie równania adwekcji-dyfuzji (1) - równanie różniczkowe
        nm = NumericalModel(pm)

        #PRZEKAZYWANIE ZMIENNYCH W CELU SKRÓCENIA ŚCIEŻKI DOSTEPU DO ZMIENNEJ
        self.numModel = nm
        self.V = nm.V
        self.numSteps = settings.vis.numSteps
        self.visFreqMatplotlib = settings.vis.visFreqMatplotlib
        self.visFreqParaViewPVD = settings.vis.visFreqParaViewPVD
        #dalszy przebieg tej ścieżki rozwidla się na 2 możliwości - 
        #1) albo zwizualizowanie dotyczchas wyliczonych zmiennych
        #2) albo dokonanie symulacji

        self.fullVis = settings.vis.fullVis
  

    def simulateLoop(self):
        # wyznacz postać funkcji słabej na podstawie parametrów z mainSim
        # decydujących o rodzaju równania, warunku brzegowym na wyjściu
        # oraz sposobie jego realizacji
        F = self.numModel.defineEquation()
        currentRho = Function(self.V)
        ID = self.settings.ID
                
        rho_pvd = File("paraView/rho_ID{}.pvd".format(ID))
        if self.fullVis:
            vrho_pvd = File("paraView/vrho_ID{}.pvd".format(ID))

        #przygotuj tabelę dla zbieranie 2 kluczowych wskaźników
        params_f = {
            "time":[], 
            "integral":[], 
            "stability":[],
            "min":[],
            "max":[], 
            "fluxWalls":[], 
            "fluxExit":[]
        }

        simTimeStep = SimTimeStep(ID)

        #zrealizuj główną pętlę symulacji
        TimeSimulation=timedelta(0)
        for i in range(2+int(self.numSteps*0.1)):
            printLog("Start step nr {}".format(i), filename="sim_ID{}".format(ID))
            
            # zaktualizuj plik z tablicami wskaźników 
            
            #ładowanie do pliku obliczone wskaźniki
            if True:#i%1==0:
                df = pd.DataFrame(params_f)
                df.to_csv("numParams_ID{}.csv".format(ID), encoding="utf-8")
                self.params_f = params_f
                TimeSimulation = simTimeStep.whenFinish(i, self.numSteps)

            #wizualizacje z matplotliba
            if self.fullVis and i%self.visFreqMatplotlib == 0:
                strLog = "Snapshot nr {} in Matpltlib rho, vrho simulation with ID{}".format(i, ID)
                printLog(strLog, filename="sim_ID{}".format(ID))
                try:
                    rhomax = self.settings.velocity.rhomax
                    vmax = self.settings.velocity.vmax
                    drawSimulationStep(self, i, ID, rhomax)
                    drawSimulationVStep(self, i, ID, vmax)
                except RuntimeError:
                    printLog("RuntimeError [matplotlib]", filename="sim_ID{}".format(ID))


            #wizualizacje z paraview
            #if i%self.visFreqParaViewPVD == 0:
            if self.fullVis and i%self.visFreqParaViewPVD == 0:
                strLog = "Snapshot nr {} in ParaViewPVD rho, vrho simulation with ID{}".format(i, ID)
                printLog(strLog, filename="sim_ID{}".format(ID))
                try:
                    rho_pvd << (self.numModel.rhoold, i*self.settings.mainSim.dt)
                    if self.fullVis:
                        vrho_pvd << (self.numModel.V_rhoold, i*self.settings.mainSim.dt)
                except RuntimeError:
                    printLog("RuntimeError [paraView PVD]", filename="sim_ID{}".format(ID))


            a, L = lhs(F), rhs(F)               

            #automatyczne wyznaczenie 
            #warunku brzegowego Dirichleta dla wyjścia przez 
            # warunek brzegowy w zależności od wersji równania
            if self.settings.mainSim.BCtype == "D-N":
                bc = DirichletBC(self.V, Constant(0.0), self.PM.wall)
                solve(a == L, currentRho, bc) 
            if self.settings.mainSim.BCtype == "Dirichlet":
                bc = DirichletBC(self.V, Constant(0.0), "on_boundary")
                solve(a == L, currentRho, bc) 
            if self.settings.mainSim.BCtype == "Neumann":
                solve(a == L, currentRho)      


            #przesunięcie się krok do przodu i przepisanie aktualnego 
            #rozwiązania równania do zmiennej pomocniczej 
            #jako poprzedni krok symulacji dla następnego przebiegu
            self.numModel.rhoold.assign(currentRho)
            self.numModel.V_rhoold = \
                VelocityFunction(self.settings.velocity).Vfield(self.numModel.rhoold, self.V)

            # wyliczenie wskaźników potrzebnych do oceny całej symulacji
            # tzn całka po OMEGA oraz kwadrat funkcji dla oceny stabliności 
            stability = assemble(currentRho*currentRho*dx)
            integral = assemble(currentRho*dx)
            minValue = min(currentRho.vector().vec().array)
            maxValue = max(currentRho.vector().vec().array)

            #dodatkowe wskaźniki, nie opisane w pracy
            vecField = self.PM.unitFieldPhi
            norma = self.numModel.normal
            myDS = self.numModel.PM.DS
            myMesh = self.numModel.PM.mesh
            mywall = self.PM.WALL_CODE
            myexit = self.PM.EXIT_CODE
            fluxWalls = assemble(currentRho*dot(vecField, norma)*myDS(mywall))
            fluxExit = assemble(currentRho*dot(vecField, norma)*myDS(myexit))


            # wstaw wyliczone wartości do struktury słownika
            params_f["time"].append(round(i*self.settings.mainSim.dt, 4))
            params_f["integral"].append(round(integral, 4))
            params_f["stability"].append(round(stability, 4))
            params_f["min"].append(round(minValue, 4))
            params_f["max"].append(round(maxValue, 4))
            params_f["fluxExit"].append(round(fluxExit, 4))
            params_f["fluxWalls"].append(round(fluxWalls, 4))
        simTimeStep.nowFinish()
        self.params_f = params_f
        return TimeSimulation
        


class EnsembleSimulation:
    def __init__(self, ensSettings):
        self.ensSettings = ensSettings

    def ensInitCondSingle(self, settings):
        printLog("init_ID{}_start".format(settings.ID), filename="PoolDebug")
        sim = Simulation(settings)
        drawInitials(sim)
        printLog("init_ID{}_finish".format(settings.ID), filename="PoolDebug")

    def ensSimulateSingle(self, settings):
        printLog("simu_ID{}_start".format(settings.ID), filename="PoolDebug")
        sim = Simulation(settings)
        TimeSimulation = sim.simulateLoop()
        printLog("simu_ID{}_finish".format(settings.ID), filename="PoolDebug")
        return TimeSimulation

    def ensSimulate(self, makeInitial=False, makeSimulation=True, fullVis=True):
        printLog("Simulation ID starts", filename="globalSettings")
        reMakeSimFolder("{}".format(self.ensSettings.simNameFolder))
        self.ensSettings.display().to_csv("ensembleSettings.csv")
        settings = self.ensSettings.array()
        if makeInitial:
            print("start initial")
            printLog("number of simulations are {}".format(self.ensSettings.len()), filename="globalSettings")
            reMakeSimFolder("paraView_init", isChdir=False)
            reMakeSimFolder("matplotlib_init", isChdir=False)
            pool1 = Pool(processes=self.ensSettings.len())
            pool1.map(self.ensInitCondSingle, settings)  
            pool1.close()
            print("finish initial")
        if makeSimulation:
            reMakeSimFolder("paraView", isChdir=False)
            reMakeSimFolder("matplotlib", isChdir=False)
            pool2 = Pool(processes=self.ensSettings.len())
            TimeSimEnsSim = pool2.map(self.ensSimulateSingle, settings)   
            pool2.close()    
        os.chdir("..")
        printLog("Simulation {} finish", filename="globalSettings")
        return TimeSimEnsSim 

    def MergeNumParams(self):
        targetTimeArr = np.array([0, 1, 2, 3])
        newDfDict = {}
        newDfDict["time"] = targetTimeArr
        for ID in self.ensSettings.len():
            simName = self.ensSettings.simName
            pathFile = "{}/ID{}_numParams.csv".format(simName, ID)
            df = pd.read_csv(pathFile)
            
            f = interpolate.interp1d(df["time"].to_numpy(), df["stability"].to_numpy())
            newDfDict["stability_ID{}"] = f(targetTimeArr)

            f = interpolate.interp1d(df["time"].to_numpy(), df["integral"].to_numpy())
            newDfDict["integral_ID{}"] = f(targetTimeArr)

        ensPathFile = "{}/ensNumParams.csv".format(simName)
        newDfDict.to_csv(ensPathFile)








## Wszystko jest poniżej to wizualizacje w paraview i matplotlib 
#wszystkich zmiennych początkowych
#########################################################################|
#########################################################################


from dolfin import *
import mshr as ms
import ufl as uf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from textwrap import wrap


from sys import *
import os
import shutil
import json
import pandas as pd

from matplotlib.cm import *
from matplotlib.colors import *
from matplotlib import rc

import multiprocessing

from geometries import *

import numpy as np

import tracemalloc
import time
import datetime


def colorMappedPlot(field, rho_max=10):
    # M = 0.2
    # b = 200
    # a, c = int(b*M)+2, int(b*M)+2
    # new_min = M/(1+2*M)
    # new_max = (1+M)/(1+2*M)

    a, b, c = 200, 500, 200

    low = mpl.colors.LinearSegmentedColormap.from_list(
        "my", ["yellow",  "black"])
    # low = mpl.cm.get_cmap("brg_r")
    mid = mpl.cm.get_cmap("cubehelix", 32)
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

    lev = np.linspace(vmin, vmax, int((a+b+c)/25)+1)

    print("!!!!!!!!!!!!!!!!!!!1", lev)

    im = plot(field, levels=lev,
                norm=newNorm, cmap=newCMap, extend="both")

    for c in im.collections:
        c.set_edgecolor("face")


    return newCMap, im




def plotNormPhi(simObj):
    ID = simObj.settings.ID
    PhysicalModelObj = simObj.numModel.PM

    fig, ax = plt.subplots(1,1)
    plt.title(r"Wartość modułu gradientu $||-\nabla\phi||$")
    newCmap = mpl.colors.LinearSegmentedColormap.from_list("new", ['red', 'salmon', 'green', 'lime', 'orange'], 16)
    #if simObj.settings.eikonal.solverType:
    #im = plot(PhysicalModelObj.normofphi, cmap=newCmap , min=0, max=2, levels=np.linspace(0, 2, 21), norm = Normalize(vmin=0, vmax=2))
    #else:
    #    im = plot(PhysicalModelObj.normofphi, cmap=newCmap)
        
    xml = File("paraView_init/krokPrzestrzenny_ID{}.xml".format(ID))
    xml << PhysicalModelObj.normofphi
    plot(PhysicalModelObj.mesh, lw=0.1, alpha=0.6)
    cbar = fig.colorbar(im, location="bottom")
    cbar.ax.set_xlabel("moduł z wektora")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/normaPhi_ID{}.pdf".format(ID), bbox_inches='tight')
    plt.clf()


def plotPhi(simObj):
    ID = simObj.settings.ID
    PhysicalModelObj = simObj.numModel.PM

    fig, ax = plt.subplots(1,1)
    plt.title(r"Pole potencjału $\phi$")
    newCmap = mpl.colors.LinearSegmentedColormap.from_list("new", ['salmon', 'green', 'lime'][::-1], 16)
    printLog("ID{} type phi0: {}".format(ID, type(PhysicalModelObj.phi0)), filename="eikonal") 
    print("#################", np.linspace(0, 120, 24+1))
    
    # if simObj.settings.eikonal.solverType:
    #     im = plot(PhysicalModelObj.phi0, cmap='viridis_r', levels=np.linspace(0, 120, 24+1), norm = Normalize(vmin=0, vmax=120))
    # else:
    #     im = plot(PhysicalModelObj.phi0, cmap='viridis_r', levels=np.linspace(0, 10000, 20+1))

    plot(PhysicalModelObj.mesh, lw=0.1, alpha=0.5)
    cbar = fig.colorbar(im, location="bottom")
    #cbar.ax.set_ylabel("moduł z wektora")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/phi_ID{}.pdf".format(ID), bbox_inches='tight')
    plt.clf()


def plotU(simObj):
    ID = simObj.settings.ID
    PhysicalModelObj = simObj.numModel.PM

    #10.0**np.arange(-20, 0)
    fig, ax = plt.subplots(1,1)
    plt.title(r"Wartość funkcji u")
    printLog("ID{} type u: {}".format(ID, type(PhysicalModelObj.u)), filename="eikonal") 
    #cmap1 = get_cmap('viridis')
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list("new2", ['yellow', 'darkblue'], 256)
    # if simObj.settings.eikonal.solverType:
    #     im = plot(PhysicalModelObj.u, levels=10.0**np.arange(-20, 1), norm=colors.LogNorm(vmin=10**-20, vmax=1.0), cmap='viridis')
    # else:
    #     im = plot(PhysicalModelObj.u, cmap='viridis_r')
    plot(PhysicalModelObj.mesh, lw=0.2, alpha=0.6)
    cbar = fig.colorbar(im, location="bottom", extend='min')
    cbar.ax.minorticks_on()
    #plt.clabel(im, levels=10.0**np.arange(-20, 0))
    #cbar.set_ticklabels(10**np.arange(0, 10))
    cbar.ax.set_xlabel("wartość funkcji u")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/u_ID{}.pdf".format(ID), bbox_inches='tight')
    plt.clf()


def plotVectorFieldPhi(simObj):
    ID = simObj.settings.ID
    PhysicalModelObj = simObj.numModel.PM
    
    fig, ax = plt.subplots(1,1)
    #plt.title(r"Znormalizowana wartość gradientu $\phi: \frac{\phi}{\sqrt{||-\nabla\phi||^2+\theta^2}}$")
    newCmap = mpl.colors.LinearSegmentedColormap.from_list("new", ['red', 'salmon', 'green', 'lime', 'orange'], 16)
    printLog("ID{} type unitFieldPhi: {}".format(ID, type(PhysicalModelObj.unitFieldPhi)), filename="eikonal")
    im = plot(PhysicalModelObj.unitFieldPhi, cmap=newCmap, lw=0.06, width=0.0012, scale=40, norm = Normalize(vmin=0, vmax=2))
    plot(PhysicalModelObj.mesh, lw=0.2, alpha=0.4)
    cbar = fig.colorbar(im, location="bottom")
    cbar.ax.set_xlabel("moduł z wektora")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/phiZnormalizowane_ID{}.pdf".format(ID), bbox_inches='tight')
    plt.clf()

def plotVectorFieldPhi_streamPlot(simObj):
    ID = simObj.settings.ID
    PhysicalModelObj = simObj.numModel.PM
    
    fig, ax = plt.subplots(1,1)
    #plt.title(r"Znormalizowana wartość gradientu $\phi: \frac{\phi}{\sqrt{||-\nabla\phi||^2+\theta^2}}$")
    newCmap = mpl.colors.LinearSegmentedColormap.from_list("new", ['red', 'salmon', 'green', 'lime', 'orange'], 16)
    printLog("ID{} type unitFieldPhi: {}".format(ID, type(PhysicalModelObj.unitFieldPhi)), filename="eikonal")
    im = plot(PhysicalModelObj.unitFieldPhi, cmap=newCmap, lw=0.06, width=0.0012, scale=40, norm = Normalize(vmin=0, vmax=2))
    plot(PhysicalModelObj.mesh, lw=0.2, alpha=0.4)
    cbar = fig.colorbar(im, location="bottom")
    cbar.ax.set_xlabel("moduł z wektora")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/phiZnormalizowane_ID{}.pdf".format(ID), bbox_inches='tight')
    plt.clf()


def drawInitialCondition(simObj):
    ID = simObj.settings.ID
    rho_max = simObj.settings.velocity.rhomax
    NumModelObj = simObj.numModel

    fig, ax = plt.subplots()
    plt.title(r"Warunek początkowy przed regularyzacją")
    func = NumericalModel.num2fs(NumModelObj.initialCondition,
                                    NumModelObj.V)
    newCMap, im = colorMappedPlot(func, rho_max=rho_max)
    f = File("paraView_init/initialCondition_ID{}.pvd".format(ID))
    fxml = File("paraView_init/initialCondition_ID{}.xml".format(ID))
    f << func
    fxml << func
    cbar = fig.colorbar(im, fraction=0.07, pad=0.1, shrink=5.0, orientation='horizontal')
    cbar.ax.set_xlabel(r"wartość funkcji $\rho$")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            

    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/initialCondition_ID{}.pdf".format(ID), bbox_inches='tight')
    plt.clf()


def drawRegularisedInitialCondition(simObj):
    ID = simObj.settings.ID
    rho_max = simObj.settings.velocity.rhomax
    NumModelObj = simObj.numModel

    fig, ax = plt.subplots()
    plt.title(r"Warunek początkowy ze zregularyzowaną dyfuzją")
    func = NumericalModel.num2fs(NumModelObj.regularisedInitialCondition,
                                    NumModelObj.V)
    newCMap, im = colorMappedPlot(func, rho_max=rho_max)
    f = File("paraView_init/initialRegularisedCondition.pvd".format(ID))
    fxml = File("paraView_init/initialRegularisedCondition.xml".format(ID))
    f << func
    fxml << func
    cbar = fig.colorbar(im, fraction=0.07, pad=0.1, shrink=5.0, orientation='horizontal')
    cbar.ax.set_xlabel(r"wartość funkcji $\rho$")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/regularisedInitialCondition_ID{}.pdf".format(ID), bbox_inches='tight')
    plt.clf()


def plotMesh(simObj):
    ID = simObj.settings.ID

    plt.title("mesh z maksymalną wartością średnicy ELEMENTU wartości {}".format(round(simObj.PM.hmax), 2))
    plot(simObj.numModel.PM.mesh, lw=0.1)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=1.0, alpha=1.0)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/mesh_ID{}.svg".format(ID), bbox_inches='tight')
    printLog("mesh_ID{}.pdf".format(ID), filename="PoolDebug")
    plt.clf()


def drawSimulationStep(simObj, numStep, ID, rho_max):
    fig, ax = plt.subplots()
    plt.title(r"Zwizualizowanie pola $\rho$ dla kroku {}".format(numStep))
    func = simObj.numModel.rhoold
    newCMap, im = colorMappedPlot(func, rho_max=rho_max)
    cbar = fig.colorbar(im, fraction=0.07, pad=0.1,
                shrink=5.0, orientation='horizontal')
    cbar.ax.set_xlabel(r"wartość funkcji $\rho$")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib/step{}_ID{}.pdf".format(numStep, ID), bbox_inches='tight')
    plt.clf()


def drawSimulationVStep(simObj, numStep, ID, vmax):
    #@saveAsFile("vrho_ParaView/ID{}_stepV{}.svg".format(numStep))
    fig, ax = plt.subplots()
    plt.title(r"Zwizualizowanie pola $V(\rho)$ dla kroku {}".format(numStep))
    func = simObj.numModel.V_rhoold
    newCMap, im = colorMappedPlot(func, rho_max=vmax)

    cbar = fig.colorbar(im, fraction=0.07, pad=0.1, shrink=5.0, orientation='horizontal')
    cbar.ax.set_xlabel(r"wartość funkcji $V(\rho)$")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib/stepV{}_ID{}.pdf".format(numStep, ID), bbox_inches='tight')
    plt.clf()


    
def drawCourantCondition(simObj):
    ID = simObj.settings.ID

    def num2fs(x, V):
        return interpolate(Expression(str(x), degree=2), V)

    h = project(CellDiameter(simObj.numModel.PM.mesh),  simObj.V).copy(deepcopy=True)
    condition = h.copy(deepcopy=True)
    nodalValuesH = h.vector().vec().array
    courantObj = CFL(simObj.settings.mainSim, simObj.settings.velocity.vmax)
    condition.vector().vec().array = [courantObj.cfl_h(h) for h in nodalValuesH]

    courant = r"\frac{3}{2} \Delta{t} \left( \frac{\alpha}{C^{1} h}+\frac{|\vec{V}|^2}{4 \alpha C^{1} h}+\frac{\kappa}{\alpha C^{2} h^{3}} \right)"
    alphaLatex = r"\alpha = \sqrt{\frac{|\vec{V}|^{2}}{4}+\frac{\kappa C^{1}}{C^{2} h^{2}}}"
    fig, ax = plt.subplots()
    plt.title("\nliczba Couranta C = ${}$\n${}$\n".format(courant, alphaLatex))
    plot(simObj.numModel.PM.mesh, lw=0.01, color='darkgrey', alpha=0.5)
    #ax.tricontour(condition, levels = [0.0, 0.1, 0.5, 1.0], lw=0.02)
    #plt.clabel(condition)
    im = plot(condition, cmap=mpl.cm.hot)
    for c in im.collections:
        c.set_edgecolor("face")
    cbar = fig.colorbar(im, orientation='horizontal')
    cbar.ax.set_xlabel("wartość liczby Couranta")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/liczbaCouranta_ID{}.pdf".format(ID))
    plt.clf()


def drawDX(simObj):
    ID = simObj.settings.ID

    plt.title("krok przestrzenny dla $h = \Delta{x}$")
    spatialStep = project(CellDiameter(simObj.numModel.PM.mesh),  simObj.numModel.PM.V)
    plot(simObj.numModel.PM.mesh, lw=0.03, color='darkgrey', alpha=0.5)
    p = plot(spatialStep, cmap=mpl.cm.hot)
    for c in p.collections:
        c.set_edgecolor("face")
    f = File("paraView_init/krokPrzestrzenny.pvd")
    fxml = File("paraView_init/krokPrzestrzenny.xml")
    f << spatialStep
    fxml << spatialStep
    plt.colorbar(p, orientation='horizontal')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    plt.grid(True, which="major", color="blue", linewidth=0.8, alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", color="royalblue", linewidth=0.4, alpha=0.5)
    plt.savefig("matplotlib_init/krokPrzestrzenny_ID{}.pdf".format(ID), bbox_inches='tight')
    plt.clf()


def drawInitials(simObj):
    ID = simObj.settings.ID    
    nameFile = "paraView_init/mesh_raw_ID{}.xml".format(ID)
    File(nameFile) << simObj.numModel.PM.mesh
    rhomax = simObj.settings.velocity.rhomax
    plotNormPhi(simObj)

    # if simObj.settings.eikonal.solverType:
    #     plotU(simObj)

    plotPhi(simObj)
    plotVectorFieldPhi(simObj)
    drawInitialCondition(simObj)
    drawRegularisedInitialCondition(simObj)
    plotMesh(simObj)
    drawDX(simObj)
    drawCourantCondition(simObj)






