#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:01:18 2017

@author: anthonybonner
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Plot the contours of the decision function (of classifier clf)
# and highlight the decision boundary in solid black.

def dfContour(clf):

    ax = plt.gca()
    # The extent of xy space
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
     
    # form a mesh/grid over xy space
    h = 0.02    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    mesh = np.c_[xx.ravel(), yy.ravel()]
    
    # evaluate the decision functrion at the grid points
    Z = clf.predict_proba(mesh)[:, 1]
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    mylevels=np.linspace(0.0,1.0,11)
    ax.contourf(xx, yy, Z, levels=mylevels, cmap=cm.RdBu, alpha=0.5)
    
    # draw the decision boundary in solid black
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='solid')


def sigmoid(z):

    return 1.0/(1.0 + np.exp(-z))


def MYdfContour(V, v0, W, w0):

    ax = plt.gca()
    # The extent of xy space
    x_min, x_max = ax.get_xlim()

    y_min, y_max = ax.get_ylim()

    # form a mesh/grid over xy space
    h = 0.02    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    # evaluate the decision function at the grid points

    # using forward pass

    U = np.dot(mesh, V) + v0  # MODIFIED

    H = np.tanh(U)  # MODIFIED

    Z = np.dot(H, W) + w0  # MODIFIED

    O = sigmoid(Z) # MODIFIED

    # plot the contours of the decision function
    O = O.reshape(xx.shape) # MODIFIED
    mylevels=np.linspace(0.0, 1.0, 11)
    ax.contourf(xx, yy, O, levels=mylevels, cmap=cm.RdBu, alpha=0.5) # MODIFIED

    # draw the decision boundary in solid black
    ax.contour(xx, yy, O, levels=[0.5], colors='k', linestyles='solid') # MODIFIED
    

# Plot the decision function of classifier clf in 3D.
# if Cflag=1 (the default), draw a contour plot of the decision function
# beneath the 3D plot.
# ax is the axes on which the function is to be drawn.

def df3D(clf,ax,cFlag=1):    
    
    # the extent of xy space
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
    
    # form a mesh/grid over xy space
    h = 0.01    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision functrion at the grid points
    Z = clf.predict_proba(mesh)[:,1]
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    ax.plot_surface(xx, yy, Z, cmap=cm.RdBu, linewidth=0, rcount=75, ccount=75)
    mylevels=np.linspace(0.0,1.0,11)
    ax.contour(xx,yy,Z,levels=mylevels,linestyles='solid',linewidths=3,cmap=cm.RdBu)
    
    # limits of the vertical axis
    z_min = 0.0
    z_max = 1.0
    
    if cFlag == 1:
        # display a contour plot of the decision function
        z_min = -0.5
        ax.contourf(xx, yy, Z, levels=mylevels, cmap=cm.RdBu, offset=z_min)
    
    ax.set_zlim(z_min,z_max)
    