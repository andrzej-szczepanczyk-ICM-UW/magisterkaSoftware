from geometries import MeshedGeometry
from courant import *
import copy
import numpy as np
import pandas as pd
import resource
import os, sys
# import psutil

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from dolfin import Mesh


#predefiniowane parametry związane z GĘSTOŚCIĄ LUDZI
PEOPLE_WALKING=2
PEOPLE_STAGNATION=5
PEOPLE_EXTREME_CROWD=10

#predefiniowane parametry związane z PRĘDKOŚCIĄ PORUSZANIA SIĘ LUDZI
MOVE_WALK=1
MOVE_RUNNING=4
MOVE_EXTREME_RUNNING=8

def computeNumSteps(geoSettings, vmax=10, dtt=0.01, factor=1.4):
    distance = geoSettings.calibration["length"] + geoSettings.calibration["corridorLength"]

    simTime = distance/vmax
    numSteps = simTime/dtt
    #print("vmax:", vmax, "dtt:", dtt, "distance", distance, "simTime", simTime, "factor", factor, "numSteps", numSteps, numSteps*factor)
    return int(numSteps*factor)


def varName(v):
    return [ i for i, a in locals().items() if a == v][0]


####################################################
# predefiniowane ustawienia dla warunku początkowego
# dla klasy NotRegularizedInitialCondition
####################################################
TEST_EXIT = {
    "xmin": 90.0,
    "xmax": 95.0,
    "ymin": 20.0, 
    "ymax": 30.0}

BROAD_BACK = {
    "xmin": 10.0,
    "xmax": 40.0,
    "ymin": 20.0, 
    "ymax": 30.0}

NARROW_BACK = {
    "xmin": 30.0,
    "xmax": 40.0,
    "ymin": 10.0, 
    "ymax": 40.0}


RIGHT = {
    "xmin": 30.0,
    "xmax": 60.0,
    "ymin": 0.0, 
    "ymax": 10.0}

LEFT = {
    "xmin": 30.0,
    "xmax": 60.0,
    "ymin": 40.0, 
    "ymax": 50.0}

RIGHT_CORNER = {
    "xmin": 80.0,
    "xmax": 90.0,
    "ymin": 0.0, 
    "ymax": 10.0}

LEFT_CORNER = {
    "xmin": 80.0,
    "xmax": 90.0,
    "ymin": 40.0, 
    "ymax": 50.0}


#######################################


class geoSettings:
    def __init__(self, obstackleType="radiusObstackle"):
        #parametry 
        self.calibration = {
            "length": 90, "width": 50, "exitWidth": 15, "roundedCorridorLength": 5.0, "corridorLength": 10.0
        }

        #słownik parametrów dla każdego z 4 typów przeszkody osobno
        self.bowlingObstackle = {
            "frontDistance": 10.0, "middlePosition": 0.0, "r": 3.0, "d": 0.5
        }
        self.anvilObstackle = {
            "width": 20.0, "length": 20.0, "frontDistance": 10.0, "middlePosition": 0.0, "thickness": 4.0
        }
        self.rectangleObstackle = {
            "width": 20.0, "frontDistance": 5.0, "backDistance": 3.0
        }
        self.radiusObstackle = {
            "frontDistance": 5.0, "radius": 12.0
        }

        #w przypadku braku jakiejkolwiek przeszkody typ jest "lack"
        if obstackleType is None:
            self.obstackleType = "lack"
        else:
            self.obstackleType = obstackleType

        #parametr mówiący o tym, że jest symeryczne
        self.geometryType = "simpleSymmetric"

        # peopleGroups - słownik zawierający wymiary prostokątów 
        # Rozdział 4.7 "Warunek początkowy" , wzór (4.6)
        self.initialParams = {
            "peopleGroups": {}
        }

    # dodaj wymiary kolejnego prostokąta
    def addInitGroup(self, boundaries, nameGroup):
        self.initialParams["peopleGroups"][nameGroup] = boundaries

    # dodaj wymiary kilku kolejnych prostokątów
    def addDefaultGroups(self, groups):
        for i, gr in enumerate(groups): 
            self.initialParams["peopleGroups"][i] = gr

    #zmień pole (self.obstackleType) zawierające typ przeszkody na brak przeszkody 
    def deleteObstackle(self):
        self.obstackleType = "lack"

    def setAnvilFront(self, frontDistance):
        self.anvilObstackle["frontDistance"]=frontDistance

    # zmień parametry przeszkody typu tak, by przewężenie między 
    # przeszkodą a ścianą wynosiła narrowness
    def bowling4frontDistance(self, narrowness=5):
        r1 = self.calibration["roundedCorridorLength"]
        r2 = 0.5*self.anvilObstackle["thickness"]
        przypr1 = 0.5*(self.calibration["exitWidth"] - 
            self.anvilObstackle["width"]) + r1
        przeciwpr = r1 + narrowness + r2
        przypr2 = (przeciwpr**2 - przypr1**2)**0.5
        self.anvilObstackle["frontDistance"] = przypr2 - r2
        self.narrowness = narrowness

    # zmień parametry tak, by przewężenie między 
    # przeszkodą a ścianą wynosiła narrowness
    def bowling4Thickness(self, narrowness=5):
        r1 = self.calibration["roundedCorridorLength"]
        r2 = 0.5*self.anvilObstackle["thickness"]
        przypr1 = 0.5*(self.calibration["exitWidth"] - 
            self.anvilObstackle["width"]) + r1
        przypr2 = self.anvilObstackle["frontDistance"] + r2
        przeciwpr = (przypr1**2 + przypr2**2)**0.5
        self.anvilObstackle["thickness"] = 2.0*(przeciwpr - r1 - narrowness)
        print(przeciwpr, r1, narrowness)
        self.narrowness = narrowness

    # zmień parametry tak, by przewężenie między 
    # przeszkodą a ścianą wynosiła narrowness
    def circle4Thickness(self, narrowness=5):
        r1 = self.calibration["roundedCorridorLength"]
        r2 = 0.5*self.radiusObstackle["radius"]
        przypr1 = 0.5*(self.calibration["exitWidth"]) + r1
        przeciwpr = r1 + narrowness + r2
        przypr2 = (przeciwpr**2 - przypr1**2)**0.5
        self.radiusObstackle["frontDistance"] = przypr2 - r2 - r1
        self.narrowness = narrowness

    # zmień parametry tak, by przewężenie między 
    # przeszkodą a ścianą wynosiła narrowness
    def circle4Thickness_constFrontDistance(self, narrowness=5):
        r1 = self.calibration["roundedCorridorLength"]
        r2Old = 0.5*self.radiusObstackle["radius"]
        oldFrontDistance = self.radiusObstackle["frontDistance"]
        przypr1 = 0.5*(self.calibration["exitWidth"]) + r1
        #przeciwpr = r1 + narrowness + r2
        przypr2 = oldFrontDistance + r1 + r2Old
        przeciwpr = (przypr1**2 + przypr2**2)**0.5
        narrownessOld = przeciwpr - r1 - r2Old
        r2New = r2Old + narrownessOld - narrowness
        newFrontDistance = oldFrontDistance + r2Old - r2New
        print(oldFrontDistance, narrownessOld, narrowness, r2New, r2Old, r1)
        self.radiusObstackle["frontDistance"] = newFrontDistance
        self.narrowness = narrowness
        self.frontDistance = self.radiusObstackle["frontDistance"]
        self.radiusObstackle["radius"] = r2New
        self.radius = self.radiusObstackle["radius"]


    def genDict(self):
        className = type(self).__name__
        prefix = className[:3]+"_"
        newDict = {}
        newDict[prefix+"_obstackleType"] = self.obstackleType[:8]
        if hasattr(self, 'FunctionSpaceMainSim'):  
            newDict[prefix+"_FS_mainSim"] = self.FunctionSpaceMainSim
        if hasattr(self, 'FunctionSpaceEikonal'):  
            newDict[prefix+"_FS_eikonal"] = self.FunctionSpaceEikonal
        if hasattr(self, 'narrowness'):  
            newDict[prefix+"_narrowness"] = self.narrowness
        if hasattr(self, 'frontDistance'):  
            newDict[prefix+"_frontDistance"] = self.frontDistance
        if hasattr(self, 'radius'):  
            newDict[prefix+"_radius"] = self.radius
        if self.obstackleType == "anvilObstackle":
            newDict[prefix+"_round_obs"] = 0.5*self.anvilObstackle["thickness"]
        newDict[prefix+"_round_wall"] = self.calibration["roundedCorridorLength"]
        return newDict



class display:
    def genDict(self):
        className = type(self).__name__
        prefix = className[:4]+"_"
        oldDict = copy.deepcopy(vars(self))
        newDict = { k.replace(k, prefix+k):v for k,v in oldDict.items() }
        return newDict

# Parametry równania eikonalnego
class eikonal(display):
    def __init__(self, theta=0.001, eikoDelta=1.0, isNormalizedVectorField=True):
        self.theta = theta
        self.eikoDelta = eikoDelta
        self.isNormalizedVectorField = isNormalizedVectorField
        self.solverType=3

# Parametry do docelowego warunku początkowego
class initialCondition(display):
    def __init__(self, rhomax=5):
        self.rhomax = rhomax

    def computeParams(self, deltaXmin):
        tolerance = 0.8
        Time = 4.0
        self.deltaT = 0.5*deltaXmin*deltaXmin*tolerance
        self.numIterations = int(Time/self.deltaT)
        return self

#Parametry dla funkcji v(rho)
class velocity(display):
    def __init__(self, vmax=10, rhomax=20, TaylorChainAccuracy=1):
        self.TaylorChainAccuracy = TaylorChainAccuracy
        self.vmax = vmax
        self.rhomax = rhomax

#Parametry dla głównej pętli symulacji
class mainSim(display):
    def __init__(self, geoSettings, meshDensity=25, dt=0.01, kappa=0.1, polynomialDegree=2):
        self.meshDensity = meshDensity
        self.polynomialDegree = polynomialDegree
        self.dt = dt
        self.kappa = kappa
        self.compute_h(geoSettings)


    #zmień parametr dt nie zmieniając pozostałych 
    # parametrów tak, aby liczba couranta wyniosła "cfl"
    def taylor_dt2CFL(self, cfl=0.1, vmax=10):
        self.dt = CFL(self, vmax=vmax).correct_dt(CFL=cfl)
        return self


    def taylor_dh2CFL(self, cfl=0.1, vmax=10, h_type="min"):
        # wyznaczone z analizy wykresu zależności między meshDensity a hmin/hmax
        DEFAULT_GEOMETRY = 10**(1.6599095+1.77082903)/2
        if h_type=="min":
            self.hmin = CFL(self, vmax=vmax).correct_dh(CFL=cfl, h_type=h_type)
            self.meshDensity = DEFAULT_GEOMETRY/self.hmin
        if h_type=="max":
            self.hmax = CFL(self, vmax=vmax).correct_dh(CFL=cfl, h_type=h_type)
            self.meshDensity = DEFAULT_GEOMETRY/self.hmax
        return self


        
    #utwórz nową instancję klasy mainSim zmieniając parametr "kappa"
    def newMainSim_kappa(self, kappa):
        newObj = copy.deepcopy(self)
        newObj.kappa = kappa
        return newObj

    #utwórz nową instancję klasy mainSim zmieniając zadany parametr "meshDensity" oraz h
    def newMainSim_MeshDensity(self, geoSettings, meshDensity):
        newObj = copy.deepcopy(self)
        newObj.meshDensity = meshDensity
        newObj.compute_h(geoSettings)
        return newObj

    #utwórz nową instancję klasy mainSim zmieniając parametr "dt"
    def newMainSim_dt(self, dt):
        newObj = copy.deepcopy(self)
        newObj.dt = dt
        return newObj

    # dla danej instancji geoSettings jednoznacznie określającej geometrię
    # oblicz kmin oraz hmax
    def compute_h(self, geoSettings):
        # print(geoSettings, type(geoSettings))
        dens = self.meshDensity

        geo=MeshedGeometry(geoSettings, densityMesh=dens)
        geo.generateMesh(self.polynomialDegree)
        self.hmin, self.hmax = geo.hmin, geo.hmax
        return geo.hmin, geo.hmax



# parametry odpowiedzialne za to, co wizualizujemy i w jaki sposób
class visualisation(display):
    def __init__(self, geoSettings, vmax=10, dtt=0.01, frequency=20, factor=1.1, fullVis=False):
        self.factor = factor
        self.firstStep = 0
        self.numSteps = computeNumSteps(geoSettings, vmax=vmax, dtt=dtt, factor=self.factor)
        self.visFreqMatplotlib = int(self.numSteps/frequency)
        self.visFreqParaViewXDMF = 1
        self.visFreqParaViewPVD = 1
        self.fullVis = fullVis

#to samo co "visualisation" tylko na potrzeby debugowania
class visualisation_DEBUG(display):
    def __init__(self, numSteps=10, factor=0.1, fullVis=False):
        self.factor = factor
        self.firstStep = 0
        self.numSteps = numSteps
        self.visFreqMatplotlib = 1
        self.visFreqParaViewXDMF = 2
        self.visFreqParaViewPVD = 3
        self.fullVis = fullVis

# klasa zawierająca wszystkie ustawienia spięte i zebrane razem
class Settings:
    def __init__(self, geoObj, mainSimObj):
        self.geo=geoObj
        self.mainSim=mainSimObj

    def setVelocity(self, velocity):
        self.velocity=velocity

    def setEikonal(self, eikonal):
        self.eikonal=eikonal

    def setRegularisation(self, initialConditionObj):
        self.initialCondition = initialConditionObj

    def setVisualisation(self, DEBUG=None, frequency=10):
        first=0
        if DEBUG is None:
            self.vis = visualisation(self.geo, vmax=self.velocity.vmax, dtt=self.mainSim.dt, frequency=frequency)
        else:
            self.vis = DEBUG

    # wygeneruj słownik reprezentujący instancję klasy
    def generateDictSummary(self, fullList=True):
        geo = self.geo.genDict()
        main = self.mainSim.genDict()
        vel = self.velocity.genDict()
        eik = self.eikonal.genDict()
        ini = self.initialCondition.genDict()
        vis = self.vis.genDict()
        if fullList:
            globalSet = {**geo, **main, **vel, **eik, **ini, **vis}
        else:
            globalSet = {**geo, **main, **vel, **vis}            
        globalSet["ID"] = self.ID

        globalSet["CFLmax"] = self.CFLmax
        globalSet["CFLmax_adv"] = self.CFLmax_adv
        globalSet["CFLmax_diff"] = self.CFLmax_diff
        globalSet["CFLmin"] = self.CFLmin
        globalSet["CFLmin_adv"] = self.CFLmin_adv
        globalSet["CFLmin_diff"] = self.CFLmin_diff
        globalSet["dxdt"] = self.dxdt
        return globalSet

    def __str__(self):
        d = self.generatDictSummary()
        return df.DataFrame([d]).T


# generowanie statystyk odnośnie geometrii
class GeometryResearch:
    def __init__(self, geoObj, vmax=10, minDensFactor=10, maxDensFactor=100):
        self.densArr = np.arange(minDensFactor, maxDensFactor)
        self.vmax = vmax
        self.geoObj = geoObj

    def researchSingleGeometry(self, geoObj, mainSimObj):
        hmins = np.full_like(self.densArr, 0.0, dtype='float')
        hmaxs = np.full_like(self.densArr, 0.0, dtype='float')
        DTs = np.full_like(self.densArr, 0.0, dtype='float')
        NSteps = np.full_like(self.densArr, 0.0)
        for i, d in enumerate(self.densArr):
            mainSimObj.meshDensity=d
            hmin, hmax = mainSimObj.compute_h(self.geoObj)
            hmins[i], hmaxs[i] = hmin, hmax
            cfl = CFL(mainSimObj, vmax=self.vmax)
            optimal_dt = cfl.correct_dt()
            N = computeNumSteps(self.geoObj, vmax=self.vmax, dtt=optimal_dt)
            DTs[i]=optimal_dt
            NSteps[i]=N
        ret = {}
        ret["meshDensity"]=self.densArr
        ret["hmin"]=hmins
        ret["hmax"]=hmaxs
        ret["opt_dt"]=DTs
        ret["num_steps"]=NSteps
        return ret

    def densFactorMemoryCorrelation(self, isPsUtil=False):
        memArr = np.full_like(self.densArr, 0.0, dtype='float')
        entities = np.full_like(self.densArr, 0.0, dtype='float')
        vertices = np.full_like(self.densArr, 0.0, dtype='float')
        getSizeInfo = np.full_like(self.densArr, 0.0, dtype='float')
        if isPsUtil:
            memPsutilArray = np.full_like(self.densArr, 0.0, dtype='float')
            proc = psutil.Process(os.getpid())
        for i, d in enumerate(self.densArr):
            memgeo = MeshedGeometry(self.geoObj, densityMesh=d)
            entities[i] = memgeo.mesh.num_entities(2)
            vertices[i] = memgeo.mesh.num_vertices()
            getSizeInfo[i] = sys.getsizeof(Mesh(memgeo.mesh))
            if isPsUtil:
                memPsutilArr[i] = proc.memory_info()[0]
        ret = {}
        ret["meshDensity"] = self.densArr
        ret["entities"] = entities
        ret["vertices"] = vertices
        ret["getsizeinfo"] = getSizeInfo[i]
        if isPsUtil:
            ret["psutil_mem"] = memPsutilArr
        return ret
            

def geometricSeries(middleValue=3.0, r=1.2, N=5):
    params = np.arange(N) - N-1/2.0
    return np.power(r, params)*middleValue

def arithmeticSeries(middleValue=3.0, diff=0.9, N=5):
    params = np.arange(N) - N-1/2.0
    return params*diff + middleValue

#tworzenie wiązek (tablic) różnych instancji Settings
class EnsembleCalibrateParams_gmvN:
    def __init__(self, geoSettingObj, mainSimObj, velocityObj, N=5, fullVis=False):
        self.geo = geoSettingObj
        self.mainSim = mainSimObj
        self.vel=velocityObj
        self.vmax=velocityObj.vmax
        self.N = N
        self.fullVis = fullVis


    def defineDiscretization(self, paramType=None, value=100):
        if paramType==None:
            middleValue = self.mainSim.meshDensity
        elif paramType=="meshDensity":
            middleValue = value
        elif paramType=="optimalNumSteps":
            middleValue = None #TODO!!!!
        elif paramType=="spatialStep":
            middleValue = None #TODO!!!!
        elif paramType=="timeStep":
            middleValue = None #TODO!!!!
        else:
            raise AttributeError("Bad paramType !!!")
        self.middleValue = middleValue
        self.mainSim.meshDensity = middleValue
        return middleValue #optimal meshDensity value

    #utwórz tablicę liczbami ciągu arytmetycznego lub geometrycznego z zadaną środkową wartością
    def computeSeriesValuesMid(self, typeSeries="geo", middleValue=None, factor=1.1):
        if middleValue==None:
            middleValue = self.middleValue
        else:
            middleValue = middleValue
        if typeSeries == "geo":
            params = np.arange(self.N) - (self.N-1)/2.0
            series = np.power(factor, params)*middleValue
        elif typeSeries == "ari":
            params = np.arange(self.N) - (self.N-1)/2.0
            series = params*factor + middleValue
        else:
            raise AttributeError("Bad typrSeries !!!")
        #print("generated series is ... {}".format(series))
        self.series = series

    #utwórz tablicę liczbami ciągu arytmetycznego lub geometrycznego z zadaną początkową wartością
    def computeSeriesValuesStart(self, typeSeries="geo", startValue=None, factor=1.1, leadingNum=None):
        if startValue==None:
            startValue = self.middleValue
        else:
            startValue = startValue
        if typeSeries == "geo":
            params = np.arange(self.N)
            series = np.power(factor, params)*startValue
        elif typeSeries == "ari":
            params = np.arange(self.N)
            series = params*factor + startValue
        else:
            raise AttributeError("Bad typrSeries !!!")


        if leadingNum is not None:
            series = np.concatenate(([leadingNum], series))
            self.N = self.N + 1
        #print("generated series is ... {}".format(series))
        self.series = series


    def computeSeriesValuesBetween(self, typeSeries="geo", minValue=0, maxValue=10):
        self.series = np.linspace(minValue, maxValue, self.N)

    def regularizationResearch(self):
        self.N = 9
        self.computeSeriesValuesMid(typeSeries="geo", middleValue=0.01, factor=np.sqrt(10))




    #wiązka symulacji ze zemieniającym się tylko jednym parametrem "kappa"
    def KAPPA_VAR(self):
        mainSimArray = []
        for kappaI in self.series:
            mainSimArray.append(self.mainSim.newMainSim_kappa(kappaI))
        self.mainSim_series = mainSimArray


    #wiązka symulacji ze zemieniającym się tylko jednym parametrem "dt"
    def DT_VAR(self):
        mainSimArray = []
        for dtI in self.series:
            mainSimArray.append(self.mainSim.newMainSim_dt(dtI))
        self.mainSim_series = mainSimArray

    # wiązka symulacji ze zemieniającymi się dwoma parametrami
    # jeden to zadana tablica parametru dt 
    # a drugie to dopasowanie parametru "h" by liczba couranta wynosiła "cfl" 
    # parametr cfl można wstawić na 2 sposoby - jako lista lub stała
    def DT_VAR_TAYLOR_DX(self, cfl=1.0):
        mainSimArray = []
        if isinstance(cfl, np.ndarray):
            for i, dtI in enumerate(self.series):
                mainSimNew = self.mainSim.newMainSim_dt(dtI)
                mainSimArray.append(mainSimNew.taylor_dh2CFL(cfl=cfl[i], vmax=self.vmax))
            #print("mainSimNew", mainSimNew)
            #print("series!!!!", self.series)
            #print("mainSimNew.taylor_dh2CFL(cfl=cfl[i], vmax=self.vmax!!!!", mainSimNew.taylor_dt2CFL(cfl=cfl[i], vmax=self.vmax))
        else:
            for dtI in self.series:
                mainSimNew = self.mainSim.newMainSim_dt(dtI)
                mainSimArray.append(mainSimNew.taylor_dh2CFL(cfl=cfl, vmax=self.vmax))
        #print("inside DT_VAR_TAYLOR_DX", mainSimArray)
        self.mainSim_series = mainSimArray

    # wiązka symulacji ze zemieniającym się jednym parametrem
    # dopasowanie parametru "dt" by liczba couranta wynosiła "cfl" 
    def TAYLOR_DT(self, cfl=1.0):
        mainSimArray = []
        if isinstance(cfl, np.ndarray):
            for i, dtI in enumerate(self.series):
                mainSimNew = self.mainSim.newMainSim_dt(dtI)
                mainSimArray.append(mainSimNew.taylor_dt2CFL(cfl=cfl[i], vmax=self.vmax))
            #print("mainSimNew", mainSimNew)
            #print("series!!!!", self.series)
            #print("mainSimNew.taylor_dh2CFL(cfl=cfl[i], vmax=self.vmax!!!!", mainSimNew.taylor_dt2CFL(cfl=cfl[i], vmax=self.vmax))
        else:
            for dtI in self.series:
                mainSimNew = self.mainSim.newMainSim_dt(dtI)
                mainSimArray.append(mainSimNew.taylor_dt2CFL(cfl=cfl, vmax=self.vmax))
        #print("inside DT_VAR_TAYLOR_DX", mainSimArray)
        self.mainSim_series = mainSimArray

    #wiązka symulacji ze zemieniającym się tylko jednym parametrem "h"
    def DX_VAR(self):
        mainSimArray = []
        for meshDensityI in self.series:
            mainSimArray.append(self.mainSim.newMainSim_MeshDensity(self.geo, meshDensityI))
        self.mainSim_series = mainSimArray

    # wiązka symulacji ze zemieniającymi się dwoma parametrami
    # jeden to zadana tablica parametru "h"
    # a drugie to dopasowanie parametru "dt" by liczba couranta wynosiła "cfl" 
    # parametr cfl można wstawić na 2 sposoby - jako lista lub stała
    def DX_VAR_TAYLOR_DT(self, cfl=1):
        mainSimArray = []
        for meshDensityI in self.series:
            mainSimNew = newMainSim_MeshDensity(self.geo, meshDensityI)
            mainSimArray.append(mainSimNew.taylor_dt2CFL(cfl=cfl, vmax=self.vmax))
        self.mainSim_series = mainSimArray

    # wiązka symulacji ze zemieniającymi się dwoma parametrami
    # jeden to zadana tablica parametru "vmax"
    # a drugie to dopasowanie parametru "dt" by liczba couranta wynosiła "cfl" 
    # parametr cfl można wstawić na 2 sposoby - jako lista lub stała 
    def VEL_VAR_TAYLOR_DT(self, cfl=1):
        velArray = []
        mainSimArray = []
        for vmaxI in self.series:
            velObj = copy.deepcopy(self.vel)
            velObj.vmax = vmaxI
            velArray.append(velObj)
            mainSimNew = newMainSim_MeshDensity(meshDensityI)
            mainSimArray.append(mainSimNew.taylor_dt2CFL(cfl=cfl, vmax=vmaxI))
        self.vel_series = velArray
        self.mainSim_series = mainSimArray

    # Dostosuj kształt przeszkód w geometrii tak, by przewężenia miały zadaną wartość.
    # zmieniając parametry kształtu przeszkody
    def BraessParadoxNarrowness(self):
        mainSimArray = []
        geoArray = []
        for narrownessI in self.series:
            geoNewI = copy.deepcopy(self.geo)
            geoNewI.bowling4Thickness(narrowness=narrownessI)
            mainSimNewI = copy.deepcopy(self.mainSim)
            mainSimNewI.compute_h(geoNewI)
            mainSimArray.append(mainSimNewI)
            geoArray.append(geoNewI)
        self.geo_series = geoArray
        self.mainSim_series = mainSimArray

    # Dostosuj kształt przeszkód w geometrii tak, by przewężenia miały zadaną wartość.
    # zmieniając odległość przeszkody od wyjścia (distance)
    def BraessParadoxDistance(self, includeLack=True):
        mainSimArray = []
        geoArray = []
        for narrownessI in self.series:
            geoNewI = copy.deepcopy(self.geo)
            if self.geo.obstackleType == "anvilObstackle":
                geoNewI.setAnvilFront(narrownessI)
            if self.geo.obstackleType == "radiusObstackle":
                geoNewI.circle4Thickness_constFrontDistance(narrowness=narrownessI)
            mainSimNewI = copy.deepcopy(self.mainSim)
            mainSimNewI.compute_h(geoNewI)
            mainSimArray.append(mainSimNewI)
            geoArray.append(geoNewI)
        #jeżeli dodatkowo jeszcze chcemy przeszkody zestawić z geometrię bez przeszkody
        if includeLack:
            geoNewI = copy.deepcopy(self.geo)
            geoNewI.deleteObstackle()
            mainSimNewI = copy.deepcopy(self.mainSim)
            mainSimNewI.compute_h(geoNewI)
            mainSimArray.append(mainSimNewI)
            geoArray.append(geoNewI)
            self.N = self.N + 1

        self.geo_series = geoArray
        self.mainSim_series = mainSimArray

    #wygeneruj tablicę reprezentującą wiązkę symulacji
    #kolumny kolejne symulacje z wiązki (tablicy)
    #wiersze kolejne pola z pojedynczego Settings (ustawienia parametrów symulacji)
    def genGlobalSettings_ei(self, eik, ini, numSteps=10, visFreq=20, DEBUG=False, factor=1.1):
        globalSettingsArr = []
        for i in range(self.N):
            # XXXobj - tablica instancji symulacji, ta, która była uzmienniana
            if hasattr(self, 'geo_series'):
                geoObj = self.geo_series[i]
            else:
                geoObj = copy.deepcopy(self.geo)   
            if hasattr(self, 'mainSim_series'):
                mainSimObj = self.mainSim_series[i]
                #print("inside genGlobalSettings_ei: ",self.mainSim_series)
            else:
                mainSimObj = copy.deepcopy(self.mainSim)
            if hasattr(self, 'vel_series'):
                velObj = self.vel_series[i]
            else:
                velObj = copy.deepcopy(self.vel)

            #print(dir(self))

            globalSetting = Settings(geoObj, mainSimObj)
            globalSetting.ID = i

            globalSetting.CFLmax = CFL(mainSimObj, velObj.vmax).cfl(h_type="max")
            globalSetting.CFLmax_adv = CFL(mainSimObj, velObj.vmax).cfl_rawAdv(h_type="max")
            globalSetting.CFLmax_diff = CFL(mainSimObj, velObj.vmax).cfl_rawDiff(h_type="max")
            globalSetting.CFLmin = CFL(mainSimObj, velObj.vmax).cfl(h_type="min")
            globalSetting.CFLmin_adv = CFL(mainSimObj, velObj.vmax).cfl_rawAdv(h_type="min")
            globalSetting.CFLmin_diff = CFL(mainSimObj, velObj.vmax).cfl_rawDiff(h_type="min")
            globalSetting.dxdt = self.mainSim.hmin/self.mainSim.dt

            globalSetting.setEikonal(eik)
            globalSetting.setVelocity(velObj)
            globalSetting.setRegularisation(ini)
            if not DEBUG:
                vis = visualisation(self.geo, vmax=self.vmax, 
                    dtt=mainSimObj.dt, frequency=visFreq, factor=factor, fullVis=self.fullVis)
            else:
                vis = visualisation_DEBUG(numSteps=numSteps, factor=0.2, fullVis=self.fullVis)
            globalSetting.setVisualisation(vis)
            globalSettingsArr.append(globalSetting)
        return globalSettingsArr


    

# narzędzia programistyczne do oglądania wiązki (tablicy) symulacji
class EnsembleSettings:
    def __init__(self, globalSettingsArray, name="simulation1"):
        if isinstance(globalSettingsArray, list):
            self.globalSettingsArray = globalSettingsArray
        elif isinstance(globalSettingsArray, pd.DataFrame):
            self.df2globalSettingsArray()
        self.simNameFolder = name

    def array(self):
        return self.globalSettingsArray

    def len(self):
        return len(self.globalSettingsArray)


    def display(self, fullList=True):
        dictArr = [s.generateDictSummary(fullList=fullList) for s in self.globalSettingsArray]
        return pd.DataFrame(dictArr).T

    def df2globalSettingsArray(self):
        pass #TODO - użyć pandas


    def getParamsArray(self, nameProperty):
        if nameProperty == "dt":
            return [setting.mainSim.dt for setting in self.settingsArray]
        if nameProperty == "kappa":
            return [setting.mainSim.kappa for setting in self.settingsArray]
        if nameProperty == "vmax":
            return [setting.mainSim.vmax for setting in self.settingsArray]
        if nameProperty == "meshDensity":
            return [setting.mainSim.meshDensity for setting in self.settingsArray]
        
        print("Parametr {} is not supported".format(nameProperty))


class DrawCourant:
    def __init__(self, mainSim, velocity):
       #  self.h = mainSim.hmin
        self.dt = mainSim.dt
        self.kappa = mainSim.kappa
        self.pd = mainSim.polynomialDegree
        self.vmax = velocity.vmax


    def draw2D(self, X=[], Y=[]):
        d = len(X)
        Values = np.zeros(d*d).reshape(d, d)
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                Values[i][j] = cfl(dt=self.dt, h=x, vmax=self.vmax, kappa=y, polynomialDegree=self.pd)

        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.grid(True, which="minor")
        print(Values)
        pc = ax.pcolor(X, Y, Values, cmap='seismic', norm=colors.LogNorm(vmin=10**(-8), vmax=10**8), alpha=0.9)
        plt.title("CFL condition")
        plt.xlabel('X - krok przestrzenny (h)')
        plt.ylabel('Y - współczynnik dyfuzji (kappa)')

        clb = plt.colorbar(pc, cmap='seismic', extend='max')
        clb.set_label('CFL condition values', labelpad=-40, y=1.05, rotation=0)        
        plt.show()


    

    
