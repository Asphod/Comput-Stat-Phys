
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



def ImportanceSampling(h,k,LSpin,NRuns,NSweeps,q=2,NInit=0,run_firstconf_list = []):
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

    else:
        firstconf_list = []
#        RandomConf = Conf2D(LSpin,'random')
        UpConf = Conf2D(LSpin,q,'up')
        DownConf = Conf2D(LSpin,q,'down')
        RandConf = Conf2D(LSpin,q,'random')
        CheckConf = Conf2D(LSpin,q,'checkerboard')
        for j in np.arange(0,NRuns):
            if j<NRuns/4:
                firstconf_list.append(UpConf)
            elif j<2*NRuns/4:
                firstconf_list.append(DownConf)
            elif j<3*NRuns/4:
                firstconf_list.append(RandConf)
            else:
                firstconf_list.append(CheckConf)
#                firstconf_list.append(RandomConf)
#                Random conformations take a very long time to equilibrate
#                at low temperatures
################################################################################

#    conf = np.empty((LSpin,LSpin))

    run_M_list = []
    run_Q_list = []
    run_MVec_list = []
    run_lastconf_list = []
    for j in np.arange(0,NRuns):
        conf = firstconf_list[j]
        # first equilibrate
        for i in np.arange(0,NInit):
            conf = CheckerboardSweep(h,k,conf,q)

        # and then start the data acquisition
        M_list = []
        Q_list = []
        MVec_list = []
        for i in np.arange(0,NSweeps):
            conf = CheckerboardSweep(h,k,conf,q)
            M = Magnetization((2*np.pi/q)*conf)
            Q = NNCoupling((2*np.pi/q)*conf)
            MVec = VecMagnetization((2*np.pi/q)*conf)
            
            M_list.append(M)
            Q_list.append(Q)
            MVec_list.append(MVec)

        run_M_list.append(M_list)
        run_Q_list.append(Q_list)
        run_lastconf_list.append(conf)
        run_MVec_list.append(MVec_list)

    run_M_array = np.asarray(run_M_list)
    run_Q_array = np.asarray(run_Q_list)
    run_m_array = 1.*run_M_array.astype(float)/NSpin
    run_q_array = 1.*run_Q_array.astype(float)/(NSpin/2.)
    run_lastconf_array = np.asarray(run_lastconf_list)
    run_MVec_array = np.asarray(run_MVec_list)
    run_mVec_array = 1.*run_MVec_array.astype(float)/NSpin

    return (run_m_array, run_q_array, run_mVec_array, run_lastconf_array )



def CheckEquilibrationPlot(identifier,m,q):
    """ identifier of the form 'h='+ str(h)+', k='+str(k) """
    
    time_array = np.arange(0,len(m[1]))

    fig = plt.figure(figsize=(20,12))

#    ax1 = fig.add_subplot(211)
    ax1 = plt.subplot2grid((2,4), (0,0), colspan=2)
    ax1.plot(time_array,np.transpose(m),"-")
    ax1.set_title('Magnetization ( ' + identifier +' )')
    ax1.set_xlabel('MC sweeps')
    ax1.set_ylabel('m')
    ax1.set_ylim([-1.1, 1.1])

#    ax2 = fig.add_subplot(212)
    ax2 = plt.subplot2grid((2,4), (1,0), colspan=2)
    ax2.plot(time_array,np.transpose(q),"-")
    ax2.set_title('NN Coupling ( ' + identifier +' )')
    ax2.set_xlabel('MC sweeps')
    ax2.set_ylabel('q')
    ax2.set_ylim([-1.1, 1.1])
    
    ax3 = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=2)
    ax3.plot(np.transpose(m),np.transpose(q),"o")
    ax3.set_title('Observed combinations')
    ax3.set_xlabel('m')
    ax3.set_ylabel('q')
    ax3.set_xlim([-1.1, 1.1])
    ax3.set_ylim([-1.1, 1.1])

    plt.tight_layout()
    return plt.show()


#%%
def run(q,NSweeps=200,listk=np.arange(1/20,1+1/20,1/20)):
    """ Simulation for a given q. Create a file with all the data."""
    q = q
    h = 0
    LSpin = 64
    NRuns = 10
#    NSweeps = 10000
    
#    listk = listk
    
#    M_ISL64Q = {}
#    Q_ISL64Q = {}
    m_ISL64Q = {}
    q_ISL64Q = {}
    mVec_ISL64Q = {}
    lastconf_ISL64Q = {}
    
    for k in tqdm(listk):
        m_ISL64Q[(h,k)], q_ISL64Q[(h,k)], mVec_ISL64Q[(h,k)], lastconf_ISL64Q[(h,k)] = ImportanceSampling(h,k,LSpin,NRuns,NSweeps,q=q)
    
    
    with open('/Users/Aspho/github/Comput-Stat-Phys/ISL64Q{}MC{}Overnight.p'.format(q,NSweeps),'wb') as pfile:
        pickle.dump((m_ISL64Q, q_ISL64Q, mVec_ISL64Q, lastconf_ISL64Q),pfile)
    
    return


#%% Cluster Algo
def Cluster(h,k,conf,q=2):
    """ Build a trial cluster to sweep. Proba to build the cluster such as acceptance probability is 1.
    Return the new configuration """
    LSpin,LSpin = conf.shape
    i,j = np.random.randint(0,LSpin,2)
    
    cluster_value = conf[i,j]%q
    explore_points = [[i,j]]
    cluster = [[i,j]]
    
    accept_proba = 1-np.exp(-2*k)

    compteur = 0
    while explore_points != []:
        print(compteur)
        compteur+=1
        for [n,m] in explore_points:
            if conf[(n+1)%LSpin , m ]%q == cluster_value and [(n+1)%LSpin,m] not in cluster:
                xi = np.random.random()
                if xi <= accept_proba:
                    explore_points.append([(n+1)%LSpin,m])
                    cluster.append([(n+1)%LSpin , m])
            elif conf[(n-1)%LSpin , m ]%q == cluster_value and [(n-1)%LSpin,m] not in cluster:
                xi = np.random.random()
                if xi <= accept_proba:
                    explore_points.append([(n-1)%LSpin , m ])
                    cluster.append([(n-1)%LSpin , m])
            elif conf[n, (m+1)%LSpin ]%q == cluster_value and [n,(m+1)%LSpin] not in cluster:
                xi = np.random.random()
                if xi <= accept_proba:
                    explore_points.append([n, (m+1)%LSpin ])
                    cluster.append([n, (m+1)%LSpin ])
            elif conf[n, (m-1)%LSpin ]%q == cluster_value and [n,(m-1)%LSpin] not in cluster:
                xi = np.random.random()
                if xi <= accept_proba:
                    explore_points.append([n, (m-1)%LSpin ])
                    cluster.append([n, (m-1)%LSpin ])
        explore_points.remove([n,m])
    
    conftrial = np.copy(conf)
    cluster_newvalue = int((cluster_value + q//2) %q)
    for [i,j] in cluster:
        conftrial[i,j] = cluster_newvalue
    return conftrial



#%% Run overnight 31/10
    
listkq2 = [0.1,0.2,0.3,0.35,0.4,0.42,0.43,0.435,0.44,0.45,0.46,0.48,0.5,0.55,0.6,0.7,0.8,0.9,1]    
listkq4 = list(np.arange(0.1, 0.6,0.1)) + list(np.arange(0.55,1.15,0.05)) + list(np.arange(1.2,2.5,0.1))
listkq6 = list(np.arange(0.1,0.8,0.1)) + list(np.arange(0.75,1.35,0.05)) + list(np.arange(1.4,3,0.1))
listkq8 = list(np.arange(0.1,0.8,0.1)) + list(np.arange(0.75,1.35,0.05)) + list(np.arange(1.4,3,0.1))
listkq10 = list(np.arange(0.1,0.9,0.1)) + list(np.arange(0.85,1.45,0.05)) + list(np.arange(1.5,3,0.1))

NSweeps = 50

for i in [2,4,6,8,10]:
    run(i,NSweeps=NSweeps,listk=eval('listkq{}'.format(i)))
