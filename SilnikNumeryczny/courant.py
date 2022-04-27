
import numpy as np

#Oblicz anie stałych geometrycznych według wzoróa (8.5), (8.6) 
# z rodziału (8.3) "Równania"
def compute_c(polynomialDegree=1):
    if polynomialDegree==1:
        c1 = 1/(4+4*np.sqrt(2))
        c2 = 1/(12+6*np.sqrt(2))
    elif polynomialDegree>1:
        c1 = 1/(6+6*np.sqrt(2))
        c2 = 1/(120+66*np.sqrt(2))
    #c3 = c2*c1/(c2+c1)
    alpha0 = np.sqrt(c1/c2)
    c3 = c2*c1*alpha0/(c2*alpha0*alpha0+c1)
    return c1, c2, c3

#obliczanie liczby Couranta według wzoru (8.1) nu_AdvDiff
def cond(h=1, vmax=1, kappa=1, polynomialDegree=2):
    c1, c2, c3 = compute_c(polynomialDegree=polynomialDegree)
    alfa_opt = ((vmax**2)/4 + (kappa**2)*c1/(c2*(h**2)))**0.5
    czlon1 = alfa_opt/(c1*h)
    czlon2 = vmax**2/(4*alfa_opt*c1*h)
    czlon3 = kappa**2/(alfa_opt*c2*(h**3))
    czlon123 = czlon1+czlon2+czlon3
    condition = 1.5*(czlon123)
    return condition

def cfl(dt=1, h=1, vmax=1, kappa=1, polynomialDegree=2):
    return dt*cond(h=h, vmax=vmax, kappa=kappa, polynomialDegree=polynomialDegree)


class CFL:
    def __init__(self, mainSim, vmax=10):
        self.hmax = mainSim.hmax
        self.hmin = mainSim.hmin
        self.vmax = vmax
        self.kappa = mainSim.kappa
        self.pd = mainSim.polynomialDegree
        self.dt = mainSim.dt
        self.c1, self.c2, self.c3 = compute_c(polynomialDegree=self.pd)


    def cfl_h(self, h):
        return cfl(dt=self.dt, h=h, vmax=self.vmax, kappa=self.kappa, polynomialDegree=self.pd)



    def h_type(self, h_type):
        if h_type == "max":
            return self.hmax
        elif h_type == "min":
            return self.hmin
        else:
            raise AttributeError("Wrong h_type - only 2 posiible values")

    def cond(self, h_type="min"):
        return cond(h=self.h_type(h_type), vmax=self.vmax,
                    kappa=self.kappa, 
                    polynomialDegree=self.pd)

    # oblicznie liczby couranta
    def cfl(self, h_type="min"):
        return self.dt*self.cond(h_type=h_type)

    # oblicznie liczby couranta "pure advection"
    def cfl_rawAdv(self, h_type="min"):
        return 1.5*self.dt*self.vmax/(self.c1*self.h_type(h_type))

    # oblicznie liczby couranta "pure diffusion"
    def cfl_rawDiff(self, h_type="min"):
        return 1.5*self.dt*self.kappa/(self.c3*self.h_type(h_type)**2)

    # zmień wartość "dt" tak, by liczba couranta była równa CFL
    def correct_dt(self, CFL=1, h_type="min"):
        return CFL/self.cond()

    # #TODO dokończyć !!!!
    # def correct_vmax(self, CFL=1, h_type="min"):
    #     return CFL*self.vmax/self.cfl_rawAdv()

    # #TODO dokończyć !!!!
    # def correct_sigma(self, CFL=1, h_type="min"):
    #     return CFL*self.vmax/self.cfl_rawAdv()

    # zmień wartość "h" tak, by liczba couranta była równa CFL
    def correct_dh(self, CFL=1, h_type="min"):
        if h_type == "min":
            hh = self.hmin
        if h_type == "max":
            hh = self.hmax

        case = "Adv"
        if case=="Adv":
            return hh*self.cfl_rawAdv()/CFL
        if case=="Diff":
            return hh*np.sqrt(self.cfl_rawAdv()/CFL)


    
