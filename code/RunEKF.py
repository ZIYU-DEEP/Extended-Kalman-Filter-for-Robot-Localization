from __future__ import division, print_function
from numpy.linalg import multi_dot, inv
from EKF import EKF
from numpy import *
import numpy as np
import sys
import math
import matplotlib.pyplot as plt


class RunEKF(object):

    def __init__(self):
        self.R = array([[2.0, 0.0, 0.0],[0.0, 2.0, 0.0],[0.0, 0.0, radians(2)]])*1E-4
        self.Q = array([[1.0, 0.0],[0.0, radians(1)]])*1E-6
        self.U = [] # Array that stores the control data where rows increase with time
        self.Z = [] # Array that stores the measurement data where rows increase with time
        self.XYT = [] # Array that stores the ground truth pose where rows increase with time
        self.MU = [] # Array in which to append mu_t as a row vector after each iteration
        self.VAR = [] # Array in which to append var(x), var(y), var(theta)

    def readData(self, filenameU, filenameZ, filenameXYT):
        print("Reading control data from %s and measurement data from %s" % (filenameU, filenameZ))
        self.U = loadtxt(filenameU, comments='#', delimiter=',')
        self.Z = loadtxt(filenameZ, comments='#', delimiter=',')
        self.XYT = loadtxt(filenameXYT, comments='#', delimiter=',')

    def run(self):

        mu0 = array([[-4.0, -4.0, math.pi/2]])# FILL ME IN: initial mean
        Sigma0 = eye(3) #[]# FILL ME IN: initial covariance
        self.VAR = array([[Sigma0[0,0], Sigma0[1,1], Sigma0[2,2]]])
        self.MU = mu0 # Array in which to append mu_t as a row vector after each iteration
        self.ekf = EKF(mu0, Sigma0, self.R, self.Q)

        for t in range(size(self.U,0)):
            self.ekf.prediction(self.U[t,:])
            self.ekf.update(self.Z[t,:])
            self.MU = concatenate((self.MU, self.ekf.getMean()))
            self.VAR = concatenate((self.VAR, self.ekf.getVariances()))

    def plot(self):

        # Plot the estimated and ground truth trajectories
        ground_truth = plt.plot(self.XYT[:,0], self.XYT[:,1], 'g.-', label='Ground Truth')
        mean_trajectory = plt.plot(self.MU[:,0], self.MU[:,1], 'r.-', label='Estimate')
        plt.legend()

        # Try changing this to different standard deviations
        sigma = 1 # 2 or 3

        # Plot the errors with error bars
        Error = self.XYT - self.MU
        T = range(size(self.XYT,0))
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(T,Error[:,0],'r-')
        axarr[0].plot(T,sigma*sqrt(self.VAR[:,0]),'b--')
        axarr[0].plot(T,-sigma*sqrt(self.VAR[:,0]),'b--')
        axarr[0].set_title('X error')
        axarr[0].set_ylabel('Error (m)')

        axarr[1].plot(T,Error[:,1],'r-')
        axarr[1].plot(T,sigma*sqrt(self.VAR[:,1]),'b--')
        axarr[1].plot(T,-sigma*sqrt(self.VAR[:,1]),'b--')
        axarr[1].set_title('Y error')
        axarr[1].set_ylabel('Error (m)')

        axarr[2].plot(T,degrees(unwrap(Error[:,2])),'r-')
        axarr[2].plot(T,sigma*degrees(unwrap(sqrt(self.VAR[:,2]))),'b--')
        axarr[2].plot(T,-sigma*degrees(unwrap(sqrt(self.VAR[:,2]))),'b--')
        axarr[2].set_title('Theta error (degrees)')
        axarr[2].set_ylabel('Error (degrees)')
        axarr[2].set_xlabel('Time')

        plt.show()

        return


if __name__ == '__main__':

    ekf = RunEKF()
    ekf.readData('../data/U.txt', '../data/Z.txt', '../data/XYT.txt')
    ekf.run()

    ekf.plot()
