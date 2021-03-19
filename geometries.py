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

class InitialCondition:
    class Person:
        def __init__(self, coordsAndSigma, curveType="exp"):
            px = coordsAndSigma[0]
            py = coordsAndSigma[1]
            sigma = coordsAndSigma[2]
            if curveType == "exp":
                #TODO do sprawdzenia i poprawienia
                valueMax = 1.0/(2.0*np.pi*sigma*sigma)
                self.funcStr = "{}*exp((-(x[0]-{})*(x[0]-{})-(x[1]-{})*(x[1]-{}))/(2*{}*{}))".format(valueMax, px, px, py, py, sigma, sigma)
            elif curveType == "invSquare":
                #TODO do sprawdzenia i poprawienia
                valueMax = 1.0/(2.0*np.pi*sigma*sigma)
                self.funcStr =  "{}/((x[0]-{})*(x[0]-{})+(x[1]-{})*(x[1]-{})+{})".format(valueMax, px, px, py, py, sigma)
                
                
    def singleGroup(self, singleGroupConfig):
        num = singleGroupConfig["numPeople"]
        xmin = singleGroupConfig["xmin"]
        ymin = singleGroupConfig["ymin"]
        xmax = singleGroupConfig["xmax"]
        ymax = singleGroupConfig["ymax"]  
        X = np.random.uniform(low=xmin, high=xmax, size=num)
        Y = np.random.uniform(low=ymin, high=ymax, size=num) 
        sigma = np.full(num, singleGroupConfig["sigma"])
        points = np.array([X, Y, sigma])
        return points.T      
        
    def multipleGroups(self, configs):
        names = configs.keys()
        points = [self.singleGroup(configs[name]) for name in names]
        return np.concatenate(points)

    def __init__(self, initialParams, curveType="exp"):
        self.peopleGroups = initialParams["peopleGroups"]
        self.curveType = curveType
        self.pointsAndSigma = self.multipleGroups(initialParams["peopleGroups"])
    
    def generateExpression(self):
        #print(self.pointsAndSigma)
        peopleFuncStr = [InitialCondition.Person(p).funcStr for p in self.pointsAndSigma]       
        expression = "+".join(peopleFuncStr)
        #print(expression)
        return Expression(expression, degree=2)
        

def saveAsFile(namefile):
    def decorator(func):
        def draw(self, *args):
            func(self, *args)
            
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')            

            plt.grid(True, which="major", color="green", linewidth=0.8, alpha=0.8)
            plt.minorticks_on()
            plt.grid(True, which="minor", color="green", linewidth=0.4, alpha=0.8)
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
    @staticmethod
    def antiCorridor(length=10, width=12, exitWidth=16, shift=2, roundedCorridorLength=0.5, corridorLength=2.0):
        middlePosition = width/2 + shift

        #print("vest length in func ", vestibuleLength)
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
        cornerL = ms.Circle(FL, roundedCorridorLength, segments=16)
        cornerR = ms.Circle(FR, roundedCorridorLength, segments=16)
        return mainSpaceL + mainSpaceR + cornerL + cornerR
    
    
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
    def strangeCircles(length=10, width=12, middlePosition=0, r=3.0, d=2.0, frontDistance=1.0):
        a=2*r+d
        H = a*sqrt(3)*0.5
        distanceA = length - frontDistance
        distanceB = length - frontDistance - H
        distanceC = length - frontDistance - H*2
        
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

    def __init__(self, compSettings, denisityMesh):
        
        self.polynomialDegree = compSettings["polynomialDegree"]

        ######################################
        # TODO program each type of symmetric
        ######################################
        if compSettings["geometryType"] == "simpleSymmetric":
            _length = compSettings["calibration"]["length"]
            _width = compSettings["calibration"]["width"]
            _exitWidth = compSettings["calibration"]["exitWidth"]
            _corridorLength = compSettings["calibration"]["corridorLength"]
            _roundedCorridorLength = compSettings["calibration"]["roundedCorridorLength"]
            geometry = MeshedGeometry.simpleSymmetric(
                length=_length, width=_width, exitWidth=_exitWidth, corridorLength=_corridorLength, roundedCorridorLength=_roundedCorridorLength)
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

            # konwersja w ukłądzie współrzędnych dla SAMEJ przeszkody na układ właściwy dla pomieszczenia
            _frontDistance = compSettings["calibration"]["length"] - \
                _frontDistance
            _backDistance = compSettings["calibration"]["length"] - \
                _backDistance
            _middlePosition = compSettings["calibration"]["width"]/2.0
            obstackle = MeshedGeometry.rectangleObstackle(
                width=_obstackleWidth, frontDistance=_frontDistance, backDistance=_backDistance, middlePosition=_middlePosition)
            # TODO generalize this piece of code to set of parts of obstackle
            obstackleGeo = ms.Polygon(obstackle["rect"])
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

            isFrontWall = True
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
            cornerFL = ms.Circle(
                Point(
                    length - obstackle["LFCircleMP"].x(),
                    width/2.0+obstackle["LFCircleMP"].y()),
                _thickness/2.0, segments=16)
            cornerFR = ms.Circle(
                Point(
                    length - obstackle["RFCircleMP"].x(),
                    width/2.0+obstackle["RFCircleMP"].y()),
                _thickness/2.0, segments=16)
            cornerBL = ms.Circle(
                Point(
                    length - obstackle["LBCircleMP"].x(),
                    width/2.0+obstackle["LBCircleMP"].y()),
                _thickness/2.0, segments=16)
            cornerBR = ms.Circle(
                Point(
                    length - obstackle["RBCircleMP"].x(),
                    width/2.0+obstackle["RBCircleMP"].y()),
                _thickness/2.0, segments=16)

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

            hasObstackle = True

        if compSettings["obstackleType"] == "strangeCircles":

            obstackle = MeshedGeometry.strangeCircles(
                length=compSettings["calibration"]["length"],
                width=compSettings["calibration"]["width"],
                middlePosition=compSettings["obstackleCalibration"]["middlePosition"],
                r = compSettings["obstackleCalibration"]["r"],
                d = compSettings["obstackleCalibration"]["d"], 
                frontDistance=compSettings["obstackleCalibration"]["frontDistance"])

            radius = compSettings["obstackleCalibration"]["r"]
            Am = ms.Circle(obstackle["Am"],radius, segments=16)
            Bl = ms.Circle(obstackle["Bl"],radius, segments=16)
            Br = ms.Circle(obstackle["Br"],radius, segments=16)
            Cl = ms.Circle(obstackle["Cl"],radius, segments=16)
            Cm = ms.Circle(obstackle["Cm"],radius, segments=16)
            Cr = ms.Circle(obstackle["Cr"],radius, segments=16)
            obstackleGeo = Am+Bl+Br+Cl+Cm+Cr
            
            hasObstackle = True
            
            

        #print("vest length eqals ", _vestibuleLength)
        antiVestibule = MeshedGeometry.antiCorridor(
            length=_length, width=_width, exitWidth=_exitWidth, shift=0, corridorLength=_corridorLength, roundedCorridorLength=_roundedCorridorLength)


        entireCorridor = MeshedGeometry.entireCorridor(
            length=_length, width=_width, exitWidth=_exitWidth, shift=0, corridorLength=_corridorLength, roundedCorridorLength=_roundedCorridorLength)


        if hasObstackle:
            self.mesh = ms.generate_mesh(
                domain-antiVestibule+entireCorridor-obstackleGeo, denisityMesh)
        else:
            self.mesh = ms.generate_mesh(domain-antiVestibule+entireCorridor, denisityMesh)

            
        initial = InitialCondition(compSettings["initialParams"])
        self.initialExpression = initial.generateExpression()

        compSettings["calibration"]["width"]

        def isExitCond(p):
            xmin = self.geometry["exit"]["left"]
            xmax = self.geometry["exit"]["right"]
            ymin = self.geometry["exit"]["bottom"]
            ymax = self.geometry["exit"]["top"]
            X = self.geometry["exit"]["X"]
            return near(p[0], X)
            #return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax 

        self.isExitCond = isExitCond

    def generateMesh(self):
        V = FunctionSpace(self.mesh, "P", self.polynomialDegree)
        self.V = V
        self.hmin = self.mesh.hmin()
        self.hmax = self.mesh.hmax()
        print("hmin::", self.hmin, "hmaxx::", self.hmax)