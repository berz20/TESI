import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import math

def deg_to_rad(x):
    return x * 2 * np.pi / 360

class SpinHamiltonian():
    def __init__(self):
        self.sx = qt.jmat(1,'x')
        self.sy = qt.jmat(1,'y')
        self.sz = qt.jmat(1,'z')

        self.ge     = 28
        self.D      = 2.87
        self.ms0    = 0

    def eigenstates(self, B, theta, phi):
        Bx = B*math.sin(deg_to_rad(theta))*math.cos(deg_to_rad(phi))
        By = B*math.sin(deg_to_rad(theta))*math.sin(deg_to_rad(phi))
        Bz = B*math.cos(deg_to_rad(theta))

        Hs = self.D*((self.sz*self.sz)-(2/3)*qt.qeye(3)) + self.ge*( Bx*self.sx + By*self.sy + Bz*self.sz )      # Electric term ignored right??

        return Hs.eigenstates()

    def transitionFreqs(self, B, theta, phi):
        egvals = self.eigenstates(B, theta, phi)[0]
        # print(self.eigenstates(B, theta, phi))
        if(B == 0): self.ms0 = egvals[0]
        f1 = abs(egvals[2] - egvals[0])
        f0 = abs(egvals[1] - egvals[0])
        f_center = (f1+f0)/2
        
        return np.array([f1,f0,f_center])
