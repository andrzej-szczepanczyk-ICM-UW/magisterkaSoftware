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
import resource

from matplotlib.cm import *
from matplotlib.colors import *
from matplotlib import rc


#TODO rozbić przeszkody z "obstackleCalibration" na kilka kategorii


class NotRegularizedInitialCondition:
    def __init__(self, config):
        self.groups = config["peopleGroups"]
        self.groupsnames = config["peopleGroups"].keys()
        self.groupsList = config["peopleGroups"].values()


    def ExpressionSingleGroup(self, singleGroupConfig):
        xmin = singleGroupConfig["xmin"]
        ymin = singleGroupConfig["ymin"]
        xmax = singleGroupConfig["xmax"]
        ymax = singleGroupConfig["ymax"]  
        return '({} < x[0] & x[0] < {} & {} < x[1] & x[1] < {})'.format(xmin, xmax, ymin, ymax)

    def CornersSingleGroup(self, singleGroupConfig):
        xmin = singleGroupConfig["xmin"]
        ymin = singleGroupConfig["ymin"]
        xmax = singleGroupConfig["xmax"]
        ymax = singleGroupConfig["ymax"]  
        return [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]


    def SurfaceSingleGroup(self, singleGroupConfig):
        xmin = singleGroupConfig["xmin"]
        ymin = singleGroupConfig["ymin"]
        xmax = singleGroupConfig["xmax"]
        ymax = singleGroupConfig["ymax"]  
        return (xmax-xmin)*(ymax-ymin)

    #zwróć wyrażenie zrozumiałe dla Fenics realizujące wzór (4,6)
    #stadium pośrednie do wygenerowania warunkua początkowego
    def ExpressionMultipleGroup(self, rhoValue):
        expressions = map(lambda x: self.ExpressionSingleGroup(x) ,self.groupsList)
        surface = self.SurfaceMultipleGroup()  
        finalex = "|".join(expressions)
        finalex = "({})*{}".format(finalex, rhoValue)
        return Expression(finalex, degree=2)



    def CornersMultipleGroup(self):
        CornerSet = []
        for g in self.groupsList:
            CornerSet.extend(self.CornersSingleGroup(g))
        return CornerSet


    def SurfaceMultipleGroup(self):
        return sum(map(lambda x: self.SurfaceSingleGroup(x) ,self.groupsList))


class Formatter:
    def __init__(self):
        self.title = "mesh"
        self.namefile = "namefile"

#parametry podane dla meshDensity=45 oraz przy standardowej geometrii
#po to, żeby rozmiar siatko był ten sam dla różnych promieni
def computeSegments(radius):
    return int(64*radius/13.5)+3


class MeshedGeometry:

    def __init__(self, geoSettings, densityMesh=20):   
        self.geoSettings = geoSettings

        #parametr używany tylko dla niekktórych rodzai przeszków
        self.numSegments = 8
        densityMesh = densityMesh

        #zmienne pomocnicze dla jedynego możliwego geometryType 
        if self.geoSettings.geometryType == "simpleSymmetric":
            _length = self.geoSettings.calibration["length"]
            _width = self.geoSettings.calibration["width"]
            _exitWidth = self.geoSettings.calibration["exitWidth"]
            _corridorLength = self.geoSettings.calibration["corridorLength"]
            _roundedCorridorLength = self.geoSettings.calibration["roundedCorridorLength"]
            geometry = MeshedGeometry.simpleSymmetric(
                length=_length, width=_width, exitWidth=_exitWidth, corridorLength=_corridorLength, roundedCorridorLength=_roundedCorridorLength)
            domain = ms.Polygon(geometry["geometry"])
            self.geometry = geometry

        #zmienne pomocnicze dla przeszkody typu "radius", czyli pojednycza kolumna
        if self.geoSettings.obstackleType == "radiusObstackle":
            rad = self.geoSettings.radiusObstackle["radius"]
            frontDistance = self.geoSettings.radiusObstackle["frontDistance"]
            setting = vars(self.geoSettings)
            print(setting["calibration"])
            X = setting["calibration"]["length"] - frontDistance - rad
            Y = setting["calibration"]["width"]/2.0#+1.0
            obstackleGeo = ms.Circle(Point(X, Y), rad)#, segments=computeSegments(rad))

        #zmienne pomocnicze dla przeszkody typu "rectangle"
        if self.geoSettings.obstackleType == "rectangleObstackle":
            _obstackleWidth = self.geoSettings.rectangleObstackle["width"]
            _frontDistance = self.geoSettings.rectangleObstackle["frontDistance"]
            _backDistance = self.geoSettings.rectangleObstackle["backDistance"]

            # konwersja w ukłądzie współrzędnych dla SAMEJ przeszkody na układ właściwy dla pomieszczenia
            _frontDistance = self.geoSettings["calibration"]["length"] - \
                _frontDistance
            _backDistance = self.geoSettings["calibration"]["length"] - \
                _backDistance
            _middlePosition = self.geoSettings["calibration"]["width"]/2.0
            obstackle = MeshedGeometry.rectangleObstackle(
                width=_obstackleWidth, frontDistance=_frontDistance, backDistance=_backDistance, middlePosition=_middlePosition)
            # TODO generalize this piece of code to set of parts of obstackle
            obstackleGeo = ms.Polygon(obstackle["rect"])

        #zmienne pomocnicze dla przeszkody typu "anvilOnstackle" w kilku różnych odmianach
        if self.geoSettings.obstackleType == "anvilObstackle":
            length = self.geoSettings.calibration["length"]
            width = self.geoSettings.calibration["width"]

            #przepisane parametry przeszkody w celu skrócenia nazwy zmiennej
            _obstackleWidth = self.geoSettings.anvilObstackle["width"]
            _obstackleLength = self.geoSettings.anvilObstackle["length"]
            _frontDistance = self.geoSettings.anvilObstackle["frontDistance"]
            _thickness = self.geoSettings.anvilObstackle["thickness"]
            _middlePosition = self.geoSettings.anvilObstackle["middlePosition"]

            # w zależności ustawień tych flag mamy różne ODMIANY przeszkody anvil
            isFrontWall = False
            isBackWall = False
            isLeftWall = True
            isRightWall = True

            obstackle = MeshedGeometry.anvilObstackle(
                length=_obstackleLength,
                width=_obstackleWidth,
                thickness=_thickness,
                middlePosition=_middlePosition,
                frontDistance=_frontDistance)

            # konwersja w ukłądzie współrzędnych dla SAMEJ przeszkody na układ właściwy dla pomieszczenia
            # obliczenie współczędnych punktów w układzie współczędnych całego pomieszczenia
            cornerFL = ms.Circle(
                Point(
                    length - obstackle["LFCircleMP"].x(),
                    width/2.0+obstackle["LFCircleMP"].y()),
                _thickness/2.0, segments=self.numSegments)
            cornerFR = ms.Circle(
                Point(
                    length - obstackle["RFCircleMP"].x(),
                    width/2.0+obstackle["RFCircleMP"].y()),
                _thickness/2.0, segments=self.numSegments)
            cornerBL = ms.Circle(
                Point(
                    length - obstackle["LBCircleMP"].x(),
                    width/2.0+obstackle["LBCircleMP"].y()),
                _thickness/2.0, segments=self.numSegments)
            cornerBR = ms.Circle(
                Point(
                    length - obstackle["RBCircleMP"].x(),
                    width/2.0+obstackle["RBCircleMP"].y()),
                _thickness/2.0, segments=self.numSegments)
            obstackleGeo = cornerFL + cornerFR + cornerBL + cornerBR

            if isFrontWall:
                FL = Point(
                    length - obstackle["LFCircleMP"].x()+_thickness/2.0,
                    width/2.0+obstackle["LFCircleMP"].y())
                FR = Point(
                    length - obstackle["RFCircleMP"].x()+_thickness/2.0,
                    width/2.0+obstackle["RFCircleMP"].y())
                BL = Point(
                    length - obstackle["LFCircleMP"].x()-_thickness/2.0,
                    width/2.0+obstackle["LFCircleMP"].y())
                BR = Point(
                    length - obstackle["RFCircleMP"].x()-_thickness/2.0,
                    width/2.0+obstackle["RFCircleMP"].y())
                rect = ms.Polygon([FL, FR, BR, BL])
                obstackleGeo += rect

            if isBackWall:
                FL = Point(
                    length - obstackle["LBCircleMP"].x()+_thickness/2.0,
                    width/2.0+obstackle["LBCircleMP"].y())
                FR = Point(
                    length - obstackle["RBCircleMP"].x()+_thickness/2.0,
                    width/2.0+obstackle["RBCircleMP"].y())
                BL = Point(
                    length - obstackle["LBCircleMP"].x()-_thickness/2.0,
                    width/2.0+obstackle["LBCircleMP"].y())
                BR = Point(
                    length - obstackle["RBCircleMP"].x()-_thickness/2.0,
                    width/2.0+obstackle["RBCircleMP"].y())
                rect = ms.Polygon([FL, FR, BR, BL])
                obstackleGeo += rect

            if isLeftWall:
                FL = Point(
                    length - obstackle["LFCircleMP"].x(),
                    width/2.0+obstackle["LFCircleMP"].y()-_thickness/2.0)
                FR = Point(
                    length - obstackle["LFCircleMP"].x(),
                    width/2.0+obstackle["LFCircleMP"].y()+_thickness/2.0)
                BL = Point(
                    length - obstackle["LBCircleMP"].x(),
                    width/2.0+obstackle["LBCircleMP"].y()-_thickness/2.0)
                BR = Point(
                    length - obstackle["LBCircleMP"].x(),
                    width/2.0+obstackle["LBCircleMP"].y()+_thickness/2.0)
                rect = ms.Polygon([FL, FR, BR, BL])
                obstackleGeo += rect

            if isRightWall:
                FL = Point(
                    length - obstackle["RFCircleMP"].x(),
                    width/2.0+obstackle["RFCircleMP"].y()-_thickness/2.0)
                FR = Point(
                    length - obstackle["RFCircleMP"].x(),
                    width/2.0+obstackle["RFCircleMP"].y()+_thickness/2.0)
                BL = Point(
                    length - obstackle["RBCircleMP"].x(),
                    width/2.0+obstackle["RBCircleMP"].y()-_thickness/2.0)
                BR = Point(
                    length - obstackle["RBCircleMP"].x(),
                    width/2.0+obstackle["RBCircleMP"].y()+_thickness/2.0)
                rect = ms.Polygon([FL, FR, BR, BL])
                obstackleGeo += rect

        #zmienne pomocnicze dla przeszkody typu "bowling"
        if self.geoSettings.obstackleType == "bowlingObstackle":

            obstackle = MeshedGeometry.bowling(
                length=self.geoSettings.calibration["length"],
                width=self.geoSettings.calibration["width"],
                middlePosition=self.geoSettings.bowlingObstackle["middlePosition"],
                r = self.geoSettings.bowlingObstackle["r"],
                d = self.geoSettings.bowlingObstackle["d"], 
                frontDistance=self.geoSettings.bowlingObstackle["frontDistance"])

            rad = self.geoSettings.bowlingObstackle["r"]
            Am = ms.Circle(obstackle["Am"], rad, segments=computeSegments(rad))
            Bl = ms.Circle(obstackle["Bl"], rad, segments=computeSegments(rad))
            Br = ms.Circle(obstackle["Br"], rad, segments=computeSegments(rad))
            Cl = ms.Circle(obstackle["Cl"], rad, segments=computeSegments(rad))
            Cm = ms.Circle(obstackle["Cm"], rad, segments=computeSegments(rad))
            Cr = ms.Circle(obstackle["Cr"], rad, segments=computeSegments(rad))
            obstackleGeo = Am+Bl+Br+Cl+Cr


        if self.geoSettings.obstackleType == "lack":
            hasObstackle = False
        else:
            hasObstackle = True

        #zmienne pomocnicze do złożenia docelowego kształtu geometrii OMEGA:
        antiVestibule = MeshedGeometry.antiCorridor(
            length=_length, width=_width, exitWidth=_exitWidth, shift=0, corridorLength=_corridorLength, roundedCorridorLength=_roundedCorridorLength)


        entireCorridor = MeshedGeometry.entireCorridor(
            length=_length, width=_width, exitWidth=_exitWidth, shift=0, corridorLength=_corridorLength, roundedCorridorLength=_roundedCorridorLength)

        memoryBefore = resource.getrusage(resource.RUSAGE_SELF)[2]

        if hasObstackle:
            generatedMesh = ms.generate_mesh(
                #docelowa geometria
                domain-antiVestibule+entireCorridor-obstackleGeo, densityMesh)
            self.mesh = generatedMesh
        else:
            self.mesh = ms.generate_mesh(domain-antiVestibule+entireCorridor, densityMesh)
            
        memoryAfter = resource.getrusage(resource.RUSAGE_SELF)[2]  
        self.meshMemory = memoryAfter - memoryBefore 


        def isExitCond(p):
            xmin = self.geometry["exit"]["left"]
            xmax = self.geometry["exit"]["right"]
            ymin = self.geometry["exit"]["bottom"]
            ymax = self.geometry["exit"]["top"]
            X = self.geometry["exit"]["X"]
            return near(p[0], X)
            #return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax 
            
        self.isExitCond = isExitCond

    # generowanie siatki wraz z określoną już z przestrzenią elementów bazowych
    # opsane w rodziale (3.1) funkcje bazowe
    def generateMesh(self, polynomialDegree):
        V = FunctionSpace(self.mesh, "P", polynomialDegree)
        self.V = V
        self.hmin = self.mesh.hmin()
        self.hmax = self.mesh.hmax()
        # print(self.hmin, self.hmax)


    ##############################
    # geometry patterns
    ##############################
    #obliczanie współrzędnych punktów dla geometrii typu "simpleSymmetric"
    @staticmethod
    def simpleSymmetric(length=10, width=12, exitWidth=16, corridorLength=0.5, roundedCorridorLength=0.5):
        margin = 0.05
        return {
            "geometry": [

                Point(length+roundedCorridorLength, 0.0),
                Point(length+roundedCorridorLength, width),
                Point(0.0, width),
                Point(0.0, 0.0)
            ],
            "exit": {
                "bottom": width/2.0-exitWidth/2.0-margin,
                "left": length+corridorLength-margin,
                "right": length+corridorLength+margin,
                "top": width/2.0+exitWidth/2.0+margin,
                "X": length+corridorLength
            }
        }

    @staticmethod
    def simpleAsymmetric(length=10, width=12, exitWidth=16, shift=2, corridorLength=0.5, roundedCorridorLength=0.5):
        margin = 0.05
        return {
            "geometry": [
                Point(0.0, 0.0),
                Point(0.0, length+roundedCorridorLength),
                Point(width, length+roundedCorridorLength),
                Point(width, 0.0)
            ],
            "exit": {
                "bottom": width/2.0-exitWidth/2.0-margin+shift,
                "left": length+corridorLength-margin,
                "right": length+corridorLength+margin,
                "top": width/2.0+exitWidth/2.0+margin+shift,
                "X": length+corridorLength
            }
        }

    ##############################
    # vestibule patterns
    ##############################
    #implementacja geometrii pomocnieczej dla utworzenia zaokrągleń przy wyjściu
    @staticmethod
    def antiCorridor(length=10, width=12, exitWidth=16, shift=2, roundedCorridorLength=5.0, corridorLength=2.0):
        middlePosition = width/2 + shift

        FL = Point(length+roundedCorridorLength, middlePosition-exitWidth/2.0-roundedCorridorLength)
        FR = Point(length+roundedCorridorLength, middlePosition+exitWidth/2.0+roundedCorridorLength)
        BL = Point(length, middlePosition-exitWidth/2.0-roundedCorridorLength)
        BR = Point(length, middlePosition+exitWidth/2.0+roundedCorridorLength)

        CFL = Point(length+roundedCorridorLength, 0)
        CFR = Point(length+roundedCorridorLength, width)
        CBL = Point(length, 0)
        CBR = Point(length, width)


        mainSpaceL = ms.Polygon([FR, CFR, CBR, BR])
        mainSpaceR = ms.Polygon([CFL, FL, BL, CBL])
        rcl = roundedCorridorLength
        cornerL = ms.Circle(FL, rcl, segments=16)#computeSegments(rcl)) #-rcl%8  # 16
        cornerR = ms.Circle(FR, rcl, segments=16)#computeSegments(rcl))
        return mainSpaceL + mainSpaceR + cornerL + cornerR
    
    #implementacja geometrii pomocnieczej dla utworzenia wyjścia (exit)
    @staticmethod
    def entireCorridor(length=10, width=12, exitWidth=16, shift=2, roundedCorridorLength=0.5, corridorLength=2.0):
        middlePosition = width/2 + shift
        
        corrFL = Point(length+corridorLength, middlePosition-exitWidth/2.0)
        corrFR = Point(length+corridorLength, middlePosition+exitWidth/2.0)
        corrBL = Point(length, middlePosition-exitWidth/2.0)
        corrBR = Point(length, middlePosition+exitWidth/2.0)
        corridor = ms.Polygon([corrFL, corrFR, corrBR, corrBL])

        return corridor

    ##############################
    # obstackle patterns
    ##############################

    # współrzędne punktów w ukłądzie współczędnym dla różnych typów przeszkody. 
    # Początek tego pośredniego ukłądu współrzędnych znajduje się pośrodku wyjścia i 
    # wzdłuż osi X jest skierowany przeciwnie niż globalny ukłąd współczędnych dla całej 
    # geometrii OMEGA 

    # współrzędne punktów w ukłądzie współczędnym dla przeszkody typu "rectangle"
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

    # współrzędne punktów w ukłądzie współczędnym dla przeszkody typu "anvil"
    @staticmethod
    def anvilObstackle(length=10, width=12, thickness=0.5, middlePosition=0, frontDistance=2.0):
        return {
            "LFCircleMP": Point(frontDistance, middlePosition-width/2.0),
            "RFCircleMP": Point(frontDistance, middlePosition+width/2.0),
            "LBCircleMP": Point(frontDistance+length, middlePosition-width/2.0),
            "RBCircleMP": Point(frontDistance+length, middlePosition+width/2.0)
        }

    # współrzędne punktów w ukłądzie współczędnym dla przeszkody typu "bowling"
    @staticmethod 
    def bowling(length=10, width=12, middlePosition=0, r=3.0, d=2.0, frontDistance=1.0):
        a=2*r+d
        H = a*sqrt(3)*0.5
        distanceC = length - frontDistance
        distanceB = length - frontDistance - H
        distanceA = length - frontDistance - H*2
        
        middlePosition = width*0.5+middlePosition
        
        return {
            "radius": r,
            "Am": Point(distanceA, middlePosition),
            "Bl": Point(distanceB, middlePosition-a*0.5),
            "Br": Point(distanceB, middlePosition+a*0.5),
            "Cl": Point(distanceC, middlePosition-a),
            "Cm": Point(distanceC, middlePosition),
            "Cr": Point(distanceC, middlePosition+a)
        }