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
        margin = 0.8
        return {
            "geometry": [

                Point(length+vestibuleLength, 0.0),
                Point(length+vestibuleLength, width),
                Point(0.0, width),
                Point(0.0, 0.0)
            ],
            "exit": {
                "bottom": width/2.0-exitWidth/2.0-margin,
                "left": length+vestibuleLength-margin,
                "right": length+vestibuleLength+margin,
                "top": width/2.0+exitWidth/2.0+margin,
                "X": length+vestibuleLength
            }
        }

    @staticmethod
    def simpleAsymmetric(length=10, width=12, exitWidth=16, shift=2, vestibuleLength=0.5):
        margin = 0.8
        return {
            "geometry": [
                Point(0.0, 0.0),
                Point(0.0, length+vestibuleLength),
                Point(width, length+vestibuleLength),
                Point(width, 0.0)
            ],
            "exit": {
                "bottom": width/2.0-exitWidth/2.0-margin+shift,
                "left": length+vestibuleLength-margin,
                "right": length+vestibuleLength+margin,
                "top": width/2.0+exitWidth/2.0+margin+shift,
                "X": length+vestibuleLength
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

        print("vest length eqals ", _vestibuleLength)

        vestibule = MeshedGeometry.vestibule(
            length=_length, width=_width, exitWidth=_exitWidth, shift=0, vestibuleLength=_vestibuleLength)

        if hasObstackle:
            self.mesh = ms.generate_mesh(
                domain-vestibule-obstackleGeo, denisityMesh)
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
            X = self.geometry["exit"]["X"]
            #return near(p[0], X)
            return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax 

        self.isExitCond = isExitCond

    def generateMesh(self):
        V = FunctionSpace(self.mesh, "P", 2)
        self.V = V
        self.hmin = self.mesh.hmin()
        self.hmax = self.mesh.hmax()
        print("hmin::", self.hmin, "hmaxx::", self.hmax)