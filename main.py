#import mshr as ms
import ufl as uf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import *
import os
import sys

from utils import *


geoSettings = {
    "calibration": {
        "length": 10, "width": 20, "exitWidth": 9, "roundedCorridorLength": 3, "corridorLength": 10
    },
    "obstackleCalibration": {
        "width": 5.0, "length": 5.0, "frontDistance": 0.0, "backDistance": 3.0, "middlePosition": 0.0, "thickness": 1.0, "r": 1.0, "d":0.5
    },
    "geometryType": "simpleSymmetric",
    "obstackleType": "strangeCircles",
    "initialParams": {
        "peopleGroups": {
            "group1": {
                "numPeople": 25,
                "sigma": 0.8,
                "xmin": 14.0,
                "xmax": 15.0, 
                "ymin": 9.0, 
                "ymax": 11.0}
            # },
            # "group2": {
            #     "numPeople": 2,
            #     "sigma": 1.0,
            #     "xmin": 1.0,
            #     "xmax": 2.0, 
            #     "ymin": 1.0, 
            #     "ymax": 2.0
            # }
        }
        
    },
    "meshDensity": 9.0,
    "polynomialDegree": 2
}

numModelSetings = {
    "dt": 0.001,
    "sigma": 0.00005,
    "vmax": 0.7,
    "rhomax": 20,
    "typeVectorField": "EIKONAL",
    "exitBCtype": "Dirichlet", ### Dirichlet / Neumann
    "usedDirichletBCmethod": True
}

velocityFuncSettings = {
    "vmax": 0.7,
    "rhomax": 20,
    "TaylorChainAccuracy": 16,
    "computationVelovityMethod": "num2fs"
}

simulationsettings = {
    "simNameFolder": "DUPLICATE_PERFECT_SIM_graphtrial",
    "minTime": 0.01,
    "maxTime": 0.03,
    "minStep": 1,
    "maxStep": 100,
    "frequency": 20
}

allSettings = {
    "geoSettings": geoSettings,
    "numModelSetings": numModelSetings,
    "velocityFuncSettings": velocityFuncSettings,
    "simulationsettings": simulationsettings
}

Simulation.SIMULATION_FOLDER = "steps"
Simulation.PARAVIEW_FORMAT_FOLDER = "paraViewFormat"
Simulation.STEPPED = False
Simulation.VISUALISE_STEPS = True

msm = ManySimulationsManagement(allSettings)
#msm.paramVariability(["geoSetings", "meshDensity"], [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
msm.paramVariability(["geoSettings", "meshDensity"], [25.0, 20.0, 15.0, 9.0])


msm.simulateMany()
msm.drawSummaryAll()
msm.minmaxCflDraw()

# Test printing
#fen.info("\nCompact output of 2D geometry:")
#fen.info(domain)
#fen.info("")
#fen.info("\nVerbose output of 2D geometry:")
#fen.info(domain, True)

# Convert subdomains to mesh function for plotting
#mf = fen.MeshFunction("size_t", mesh2d, 2, mesh2d.domains())
#fen.plot(mf, "Subdomains")
