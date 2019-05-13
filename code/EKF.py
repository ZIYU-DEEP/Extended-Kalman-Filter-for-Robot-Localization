from __future__ import division, print_function
from numpy.linalg import multi_dot, inv
from numpy import *
import numpy as np
import math


class EKF(object):
    def __init__(self, mu, Sigma, R, Q):
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

    def getMean(self):
        return self.mu

    def getCovariance(self):
        return self.Sigma

    def getVariances(self):
        return np.array(
            [[self.Sigma[0, 0], self.Sigma[1, 1], self.Sigma[2, 2]]])

    def prediction(self, u):
        x_pre = self.mu[0][0]
        y_pre = self.mu[0][1]
        theta_pre = self.mu[0][-1]
        d_now = u[0]
        delta_now = u[1]

        f = np.array([[x_pre + d_now * math.cos(theta_pre)],
                      [y_pre + d_now * math.sin(theta_pre)],
                      [theta_pre + delta_now]])
        F = np.array([[1, 0, - math.sin(theta_pre) * d_now],
                      [0.0, 1.0, cos(theta_pre) * d_now],
                      [0.0, 0.0, 1.0]])

        self.mu = np.array([np.ravel(f)])
        self.Sigma = multi_dot([F, self.Sigma, F.transpose()]) + self.R

    def update(self, z):
        mu_, Sigma_ = self.mu, self.Sigma

        x_ = mu_[0][0]
        y_ = mu_[0][1]
        theta_ = mu_[0][2]
        h = np.array([[np.square(x_) + np.square(y_)],
                      [theta_]])
        H = np.array([[2 * x_, 2 * y_, 0],
                      [0, 0, 1]])
        K = multi_dot([Sigma_,
                       H.transpose(),
                       inv(multi_dot([H, Sigma_, H.transpose()]) + self.Q)])

        self.mu = mu_ + np.ravel(dot(K, (z.reshape(-1, 1) - h)))
        self.Sigma = dot((np.identity(3) - dot(K, H)), Sigma_)
