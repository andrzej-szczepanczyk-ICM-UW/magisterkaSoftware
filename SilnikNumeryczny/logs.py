from datetime import datetime

def printLog(comunicat, filename="log"):
    d = datetime.now().strftime("%m/%d/%Y, %H:%M:%S:%f")
    with open(filename, "a") as f:
            f.write("[{}] {}\n".format(d, comunicat))

class SimTimeStep:
    def __init__(self, ID):
        self.START = datetime.now()
        d = self.START.strftime("%D %H:%M:%S:%f")
        comunicat =  "Symulacja rozpoczela siee.".format(d)
        self.f="evaluateTime_ID{}".format(ID)
        printLog(comunicat, filename=self.f)

    def whenFinish(self, curr, last):
        if curr==0:
            return 
        differ = datetime.now() - self.START
        approxSimTime = differ*last/curr
        d = self.START + approxSimTime
        d = d.strftime("%m/%d/%Y, %H:%M:%S:%f")
        comunicat = "Szacowane zakonczenie symulacji [{}] Krok {} / {} ".format(d, curr, last)
        printLog(comunicat, filename=self.f)
        printLog("czas trwania symulacji: {}".format(approxSimTime.total_seconds()), filename="T{}".format(self.f))
        return approxSimTime

    def nowFinish(self):
        comunicat = "Zakonczenie symulacji"
        printLog(comunicat, filename=self.f)


# simTimeStep = SimTimeStep()
# N = 50
# freq=7
# for j in range(N):
#     for i in range (10^15):
#         t=12*13*14^2-43-12-12-(43+45+75)**34
#     simTimeStep.whenFinish(j*freq, N*freq)
# simTimeStep.nowFinish()