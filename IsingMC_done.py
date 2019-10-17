
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numbers
import math
import matplotlib as mpl
from matplotlib import animation, rc


def Conf2D(LSpin, q=2, type0='up'):
    """
    Returns a 2d integer array of s=+/-1 vaules representing a configuration of
    an (LSpin x LSpin) Ising system.
    Allowed types:
        "up":           all-spins-up cofiguration
        "down":         all-spins-down cofiguration
        "interface":    l.h.s.-half-up and r.h.s.-half down
        "alternating":  checkerboard pattern
        "random":       random
        default:        "up"
    """
    if type0 == 'up':
        conf = np.zeros( (LSpin,LSpin) , dtype=int)
    elif type0 == 'alternating' or type0 == 'checkerboard':
        if q%2 == 0:
            conf = np.zeros( (LSpin,LSpin) , dtype=int)
            conf[1::2,0::2] = q//2 # in even lines set odd elemets
            conf[0::2,1::2] = q//2 # in odd lines set even elements
        else:
            print('No checkerboard with odd q yet. Returning up')
            conf = np.zeros( (LSpin,LSpin) , dtype=int)
    elif type0 == 'down':
            if q%2 == 0:
                conf = q//2*np.ones( (LSpin,LSpin) , dtype=int)
            else:
                print('No checkerboard with odd q yet. Returning up')
                conf = np.zeros( (LSpin,LSpin) , dtype=int)
    elif type0 == 'random':
        conf = np.random.randint(0,q,(LSpin,LSpin))
    else:
        conf = np.zeros( (LSpin,LSpin) , dtype = int)


    return conf


def BlackOrWhiteSweep(h,k,conf, q=2, even = True):
    """
    returns conf after executing one odd or even checkerboard spin flip sweep
    """
    
    LSpin = len(conf[1])

    #construction de direction random
    trial = Conf2D(LSpin,q,'random')
    conftrial = np.copy(conf)
    newconf = np.copy(conf)

    #sweep dans la conftrial
    
    
    if even:
        conftrial[1::2,0::2]=trial[1::2,0::2]
        conftrial[0::2,1::2]=trial[0::2,1::2]
    else:
        conftrial[0::2,0::2]=trial[0::2,0::2]
        conftrial[1::2,1::2]=trial[1::2,1::2]

    #calcul des énergies à t et t+1
    Et = -h*np.cos((2*np.pi/q)*conf) - k*LocalNNCoupling((2*np.pi/q)*conf)
    Ett = -h*np.cos((2*np.pi/q)*conftrial) - k*LocalNNCoupling((2*np.pi/q)*conftrial)
    DeltaE = Ett - Et

    #tirage des probas
    xi = np.random.rand(LSpin,LSpin)

    #matrice des probas en fonction de l'énergie
    acc = np.exp(-DeltaE)
    acc[acc>1] = 1

    change = acc - xi
    newconf[change > 0] = conftrial[change > 0]
    return np.array(newconf)
  

def CheckerboardSweep(h,k,conf,q = 2):
    """
    returns conf after executing one odd and one even checkerboard spin flip sweep in random order
    """
    LSpin=len(conf[1])
    if (LSpin % 2 == 1):
        print ('no checkerboard moves for odd system sizes')
        return conf

    #tirage d'une valeur aléatoire
    if np.random.rand(1)[0]<0.5:
        newconf = BlackOrWhiteSweep(h,k,conf,q = q,even=False)
        newconf = BlackOrWhiteSweep(h,k,newconf,q = q,even=True)
    else:
        newconf = BlackOrWhiteSweep(h,k,conf,q = q,even=True)
        newconf = BlackOrWhiteSweep(h,k,newconf,q = q,even=False)
    return newconf


def testrun(h,k,startconf,q=2,NSweeps=10):
    """
    returns all the conf for NSweeps
    """
    run = [startconf]
    conf = np.copy(startconf)
    for i in range(0,NSweeps):
        conf = CheckerboardSweep(h,k,conf,q=q)
        run.append(conf)
    return run



def LocalNNCoupling(confrad):
    """ Returns a matrix of the dimension of the configuration matrix with total nearest-neighbor
    coupling for a spin configuration. Conf must be in radians for the scalar product"""
    Theta_diff_right = np.cos(confrad - np.roll(confrad, 1, axis = 1))
    Theta_diff_left =  np.cos(confrad - np.roll(confrad, -1, axis = 1))
    Theta_diff_up =  np.cos(confrad - np.roll(confrad, 1, axis = 0))
    Theta_diff_down =  np.cos(confrad - np.roll(confrad, -1, axis = 0))
    LocalNNC = Theta_diff_right + Theta_diff_left + Theta_diff_up + Theta_diff_down
    return LocalNNC


def Magnetization(confrad):
    """
    Returns the total magnetization for a spin configuration. conf in radians!!
    """
    return np.sum(np.cos(confrad))

def NNCoupling(confrad):
    """
    Returns the total nearest-neighbor coupling for a spin configuration. conf in radians!!
    """
    return np.sum(LocalNNCoupling(confrad))


#%%
def AnimateMCRun(run,q=2):
    """
    Animation of an array of 2D spin conformations. 
    Down spins are shown blue. Up spins red.
    """
    fig = plt.figure()
    ax = plt.axes()
    cmap='plasma'
    norm = mpl.colors.Normalize(vmin=0,vmax=q-1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    im = ax.matshow(run[0], cmap=cmap,norm=norm)
    ax.set_axis_off()
 
    def init():
        im.set_data(run[-1])
 
    def animate(i):
        im.set_array(run[i])
        return [im]


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(run), interval=100)
    rc('animation', html='html5')
    return anim



