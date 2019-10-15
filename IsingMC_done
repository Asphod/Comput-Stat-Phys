# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numbers
import math
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
#    elif type0 == 'down':
#        conf = -np.ones( (LSpin,LSpin) , dtype=int)
#    elif type0 == 'alternating' or type == 'checkerboard':
#        conf = np.ones( (LSpin,LSpin) , dtype=int)
#        conf[1::2,0::2]=-1 # in even lines set odd elemets
#        conf[0::2,1::2]=-1 # in odd lines set even elements
    elif type0 == 'random':
        conf=np.random.randint(0,q,(LSpin,LSpin))
    else:
        conf = np.zeros( (LSpin,LSpin) , dtype=int)
    return conf



def BlackOrWhiteSweep(h,k,conf,q,even):
    """
    returns conf after executing one odd or even checkerboard spin flip sweep
    """

    LSpin = len(conf[1])

    #construction de direction random
    trial = Conf2D(LSpin,q,'random')
    conftrial = np.copy(conf)

    #sweep dans la conftrial
    if even:
        conftrial[0::2,0::2]=trial[0::2,0::2]
    else:
        conftrial[1::2,1::2]=trial[1::2,1::2]

    #calcul des énergies à t et t+1
    Et = h*conf + k*LocalNNCoupling(conf)
    Ett = h*conftrial + k*LocalNNCoupling(conftrial)
    DeltaE = Ett - Et

    #tirage des probas
    xi = np.random.rand(LSpin,LSpin)

    #matrice des probas en fonction de l'énergie
    acc = np.exp(DeltaE)

    #flip ou non
    flip = np.sign(xi-acc).astype(int)
    flip[x == -1] = 0

    #construction de la conf après trial
    trial = trial*flip
    if even:
        conf[0::2,0::2]=trial[0::2,0::2]
    else:
        conf[1::2,1::2]=trial[1::2,1::2]
    return conf



def CheckerboardSweep(h,k,conf,q):
    """
    returns conf after executing one odd and one even checkerboard spin flip sweep in random order
    """
    LSpin=len(conf[1])
    if (LSpin % 2 == 1):
        print ('no checkerboard moves for odd system sizes')
        return conf

    #tirage d'une valeur aléatoire
    if np.random.rand(1)[0]<0.5:
        newconf = BlackOrWhiteSweep(h,k,conf,q,even=False)
        newconf = BlackOrWhiteSweep(h,k,newconf,q,even=True)
    else:
        newconf = BlackOrWhiteSweep(h,k,conf,q,even=True)
        newconf = BlackOrWhiteSweep(h,k,newconf,q,even=False)
    return newconf



def LocalNNCoupling(conf):
    """ Returns a matrix of the dimension of the configuration matrix with total nearest-neighbor
    coupling for a spin configuration. Conf must be in radians for the scalar product"""
    Theta_diff_right = np.cos(conf - np.roll(conf, 1, axis = 1))
    Theta_diff_left =  np.cos(conf - np.roll(conf, -1, axis = 1))
    Theta_diff_up =  np.cos(conf - np.roll(conf, 1, axis = 0))
    Theta_diff_down =  np.cos(conf - np.roll(conf, -1, axis = 0))
    LocalNNC = Theta_diff_right + Theta_diff_left + Theta_diff_up + Theta_diff_down
    return LocalNNC



def ImportanceSampling(h,k,LSpin,NRuns,NSweeps,NInit=0,run_firstconf_list = [],rule='Glauber'):
    """
    Returns four NRunsxNSweeps arrays containing M, Q, m and q for
    LSpin x LSpin Ising conformations generated using Metropolis MC
    with checkerboard sweeps
    plus the list of the NRuns last conformations
    """
    NSpin = LSpin**2

    #check assez de matrices initiales pour faire Nruns (si on donne une liste non vide)
    if len(run_firstconf_list)!=0 and len(run_firstconf_list)<NRuns:
        print(len(run_firstconf_list), ' conformations in firstconf_list insufficient for ',NRuns, ' runs')
        return

    #check taille des matrices donnés
    if len(run_firstconf_list)!=0 and len(run_firstconf_list[0])!=LSpin:
        print('Size ', len(run_firstconf_list[0]), ' of conformations in firstconf_list incompatible with LSpin = ',LSpin)
        return

    #si liste de taille au moins Nruns, go
    if len(run_firstconf_list)!=0:
        firstconf_list = run_firstconf_list

################################################################################
    ##############ATTENTION, PB SUR LES CONFS DE BASE
    else:
        firstconf_list = []
#        RandomConf = Conf2D(LSpin,'random')
        UpConf = Conf2D(LSpin,'up')
        DownConf = Conf2D(LSpin,'down')
        AltConf = Conf2D(LSpin,'alternating')
        for j in np.arange(0,NRuns):
            if j<NRuns/4:
                firstconf_list.append(UpConf)
            elif j<NRuns/2:
                firstconf_list.append(DownConf)
            elif j<3*NRuns/4:
                firstconf_list.append(AltConf)
            else:
                firstconf_list.append(-AltConf)
#                firstconf_list.append(RandomConf)
#                Random conformations take a very long time to equilibrate
#                at low temperatures
################################################################################

    conf = np.empty((LSpin,LSpin))

    run_M_list = []
    run_Q_list = []
    run_lastconf_list = []
    for j in np.arange(0,NRuns):
        conf = firstconf_list[j]
        # first equilibrate
        for i in np.arange(0,NInit):
            conf = CheckerboardSweep(h,k,conf,rule)

        # and then start the data acquisition
        M_list = []
        Q_list = []
        for i in np.arange(0,NSweeps):
            conf = CheckerboardSweep(h,k,conf,q)
            M = Magnetization(conf)
            Q = NNCoupling(conf)
            M_list.append(M)
            Q_list.append(Q)

        run_M_list.append(M_list)
        run_Q_list.append(Q_list)
        run_lastconf_list.append(conf)

    run_M_array = np.asarray(run_M_list)
    run_Q_array = np.asarray(run_Q_list)
    run_m_array = 1.*run_M_array.astype(float)/NSpin
    run_q_array = 1.*run_Q_array.astype(float)/NSpin/2.
    run_lastconf_array = np.asarray(run_lastconf_list)

    return (run_M_array, run_Q_array, run_m_array, run_q_array, run_lastconf_array )
