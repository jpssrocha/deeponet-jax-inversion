#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:53:03 2024

@author: mrborges
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import sys
import time

###############################################################################
###############################################################################
def field_to_intern_simul(simulator, subdom, inputsimul, Y, param):
    '''Description: This function returns the field to the internal
    simulator'''
    field = 0.0
    if simulator == 5:
        if subdom:
            field = Y
        else:
            field = param[inputsimul.numb_Gauss_par]
    return field
###############################################################################
        
###############################################################################
###############################################################################
def internal_simulator_dic():
    '''Description: This function returns the internal simulator dictionary'''
    options = {'slab2D': slab_simulation, 
               'fivespot2D': five_spot_simulation}
    return options
###############################################################################

###############################################################################
###############################################################################
def search_str_by_line(file_path, word, functype):
    '''Description: This function searches for a string in a file and returns'''
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        # read all content of a file
        content = line.strip()
        # check if string present in a file
        if word in content:
            strnum = content[content.find(':')+1:len(content)]
            strnum = strnum.strip()
            if strnum.isdigit() == True:
                strnum = functype(strnum)
    return strnum
###############################################################################

###############################################################################
###############################################################################
def input_simulation_parameters(filename):
    '''Description: This function reads the input simulation parameters'''
    simul_setup = ''
    Dom  = [0., 0., 0.]
    mesh = [1, 1, 1]
    BHP, PR, PL, rw, q, mu, beta, rho = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pos, numb_Gauss_par = 0, 0
    # =========================================================================
    if filename != None:
        simul_setup = search_str_by_line(filename,
                                         'Simulation domain configuration:',
                                         np.int64)
        numb_Gauss_par = search_str_by_line(filename,
                                         'Number of Gaussian parameter:',
                                         np.int64)
        numb_Gauss_par = numb_Gauss_par - 1
        beta = np.fromstring(
            search_str_by_line(filename,'Beta parameter of the permeability field:',
                               np.int64), dtype = float, sep = ' ')
        rho = np.fromstring(
            search_str_by_line(filename,
                               'Rho (strength) parameter of the permeability field:',
                               np.int64), dtype = float, sep = ' ')
        mu = np.fromstring(search_str_by_line(filename, 'Water viscosity (mu):',
                                              np.int64), dtype = float, sep = ' ')
        L  = search_str_by_line(filename, 'Domain size:', np.int64)
        Dom= np.zeros((3,), dtype = 'float')
        Dom= np.fromstring(L, dtype = float, sep = ' ')
        M  = search_str_by_line(filename, 'Computational mesh:', np.int64)
        mesh= np.zeros((3,), dtype = 'int')
        mesh= np.fromstring(M, dtype = int, sep = ' ')
        pos = np.fromstring(search_str_by_line(filename, 'Monitor cell positions:',
                                               np.int64), dtype = int, sep = ',')
        # =====================================================================
        if simul_setup == 'slab2D':
            PR, PL = slab2Dconf(filename)
        # =====================================================================
        if simul_setup == 'fivespot2D':
            BHP, rw, q = fivespot2Dconf(filename)
        # =====================================================================
    return simulationpar(simul_setup,beta,rho,mu,Dom,mesh,BHP,PR,PL,
                         rw,q,pos,numb_Gauss_par)
###############################################################################

###############################################################################
###############################################################################
def slab2Dconf(filename):
    '''Description: This function reads the slab configuration'''
    PR = np.fromstring(search_str_by_line(filename,
                                          'Right side Dirichlet pressure:',
                                           np.int64), dtype = float, sep = ' ')
    PL = np.fromstring(search_str_by_line(filename,
                                          'Left side Dirichlet pressure:',
                                           np.int64), dtype = float, sep = ' ')
    return PR, PL
###############################################################################

###############################################################################
###############################################################################
def fivespot2Dconf(filename):
    '''Description: This function reads the five-spot configuration'''
    BHP = np.fromstring(search_str_by_line(filename,
                                           'Bottom hole pressure (BHP):',
                                           np.int64), dtype = float, sep = ' ')
    rw = np.fromstring(search_str_by_line(filename,
                                           'Well radius (rw):',
                                           np.int64), dtype = float, sep = ' ')
    q  = np.fromstring(search_str_by_line(filename,
                                           'Production rate (q):',
                                           np.int64), dtype = float, sep = ' ')
    return BHP, rw, q
###############################################################################

###############################################################################
###############################################################################
class simulationpar:
    '''Description: This class contains the simulation parameters'''
    def __init__(self, simul_setup, beta, rho, mu, Dom, mesh, BHP, PR, PL,
                 rw, q, pos, numb_par):
        self.simul_setup = simul_setup
        self.beta = beta
        self.rho  = rho
        self.mu   = mu
        self.Dom  = Dom
        self.mesh = mesh
        self.BHP  = BHP
        self.PR   = PR
        self.PL   = PL
        self.rw   = rw
        self.q    = q
        self.positions = pos
        self.numb_Gauss_par = numb_par
###############################################################################

###############################################################################
###############################################################################
def format_seconds_to_hhmmss(seconds, name):
    '''Description: This function formats the seconds to hh:mm:ss'''
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('\n=====================================================')
    print('=====================================================')
    print(name)
    print("Total time elapsed (h:m:s)..............: %02d:%02d:%2.2f" %
          (hours, minutes, seconds))
    print('=====================================================')
    print('=====================================================')
    return hours, minutes, seconds
###############################################################################

###############################################################################
###############################################################################
def slab3D(inputpar, fieldY = 0.0):
    start_timing = time.time()    
    ###########################################################################
    # Define the grid =========================================================
    nx, ny, nz = inputpar.mesh[0], inputpar.mesh[1], inputpar.mesh[2]
    Lx, Ly, Lz = inputpar.Dom[0], inputpar.Dom[1], inputpar.Dom[2]
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    idx, coord = coordinates3D(nx, ny, nz, Lx, Ly, Lz)
    ###########################################################################
    beta= inputpar.beta
    rho = inputpar.rho
    mu  = inputpar.mu
    ###########################################################################
    # Permeability field (heterogeneous) ======================================
    fieldY  = np.ones((nx*ny*nz,))
    conduct = hydraulic_condutivity3D(fieldY, nx, ny, nz, mu, rho, beta)
    ###########################################################################
    # Variables ===============================================================
    P = np.zeros((nx, ny), dtype=float)
    Q = np.zeros((nx, ny), dtype=float)
    ###########################################################################
    # Transmissibilities ======================================================
    Tx, Ty, Tz = transm3D(conduct, nx, ny, nz, dx, dy, dz)
###############################################################################

###############################################################################
###############################################################################
def transm3D(conduct, nx, ny, nz, dx, dy, dz):
    nTx, nTy, nTz = nx * nz * (ny + 1), ny * nz * (nx + 1), ny * nx * (nz + 1)
    Tx, Ty, Tz = np.zeros((nTx,)), np.zeros((nTy,)), np.zeros((nTy,))
    # Transmissibilities x ====================================================
    fatx = 2.0 * dy * dz
    ## Internal element
    for k in range(1,nz-1):
        for j in range(1,ny-1):
            for i in range(1,nx-1):
                index         = k * (nx*ny) + j * nx + i
                index_plus_x  = j * nx + i + 1
                index_minus_x = j * nx + i - 1
                index_plus_y  = (j + 1) * nx + i
                index_minus_y = (j - 1) * nx + i
                print(index)
    return Tx, Ty, Tz
###############################################################################

###############################################################################
###############################################################################
def slab_simulation(inputpar, fieldY = 0.0):
    '''Description: This function solves the Darcy equation for the 
    slab configuration'''
    start_timing = time.time()    
    ##########################################################
    # Define the grid ========================================
    nx, ny = inputpar.mesh[0], inputpar.mesh[1]
    Lx, Ly = inputpar.Dom[0], inputpar.Dom[1]
    dx, dy = Lx / nx, Ly / ny
    idx, coord = coordinates(nx, ny, Lx, Ly)
    ##########################################################
    beta= inputpar.beta
    rho = inputpar.rho
    mu  = inputpar.mu
    ##########################################################
    # Permeability field (heterogeneous) =====================
    conduct = hydraulic_condutivity(fieldY, nx, ny, mu, rho, beta)
    ##########################################################
    # Variables ==============================================
    P = np.zeros((nx, ny), dtype=float)
    Q = np.zeros((nx, ny), dtype=float)
    ##########################################################
    # boundary conditions ====================================
    prod_well_index = well_position(coord, Lx/2., Ly/2.,
                                         dx, dy, nx, ny)
    wells_position  = np.array([prod_well_index, nx*ny-1,
                                nx*ny-nx, 0, nx-1], dtype=int)
    prod_well_idx   = (idx[prod_well_index,0],
                       idx[prod_well_index,1])
    BHP = inputpar.BHP + np.zeros(len(wells_position), dtype=float)
    Pr  = inputpar.PR
    Pe  = inputpar.PL
    WI  = []
    ##########################################################
    # darcy flow solver ======================================
    pres_flat, pres = darcy_solver(conduct, Q, P, nx, ny, dx,
                                   dy, Pr, Pe, 1, BHP, 
                                   prod_well_idx, mu, WI)
    ##########################################################
    # save the pressure field ================================
    positions = inputpar.positions
    ##########################################################
    # Save the pressure field ================================
    data = pres_flat[positions]
#    file_name = './' + expname + '/pressure.dat'
#    save_pressure(data, file_name)
    end_timing = time.time()
    seconds    = end_timing - start_timing
    hours, minutes, \
    # seconds = format_seconds_to_hhmmss(seconds, 'Simulation')
    ##########################################################
    # Plot the pressure field ================================
#    plot_pres(pres, Lx, Ly)
    return data
###############################################################################

###############################################################################
###############################################################################
def five_spot_simulation(inputpar, fieldY):
    '''Description: This function solves the Darcy equation for the
    five-spot configuration'''
    start_timing = time.time()
    ##########################################################
    # Define the grid ========================================
    nx, ny = inputpar.mesh[0], inputpar.mesh[1]
    Lx, Ly = inputpar.Dom[0], inputpar.Dom[1]
    dx, dy = Lx / nx, Ly / ny
    idx, coord = coordinates(nx, ny, Lx, Ly)
    ##########################################################
    beta= inputpar.beta
    rho = inputpar.rho
    mu  = inputpar.mu
    ##########################################################
    # Permeability field (heterogeneous) =====================
    conduct = hydraulic_condutivity(fieldY, nx, ny, mu, rho, beta)
    ##########################################################
    # Variables ==============================================
    P = np.zeros((nx, ny), dtype=float)
    Q = np.zeros((nx, ny), dtype=float)
    ##########################################################
    # boundary conditions ====================================
    prod_well_index = well_position(coord, Lx/2., Ly/2.,
                                         dx, dy, nx, ny)
    wells_position  = np.array([prod_well_index, nx*ny-1,
                                nx*ny-nx, 0, nx-1], dtype=int)
    prod_well_idx   = (idx[prod_well_index,0],
                       idx[prod_well_index,1])
    day, mega = 86400.0, 1.0e06
    q   = inputpar.q / day  # Production rate
    BHP = inputpar.BHP + np.zeros(len(wells_position), dtype=float)
    Pr  = 0.0
    Pe  = 0.0
    rw  = inputpar.rw
    WI  = WIfunction(dx, dy, rw, conduct*mu, idx, wells_position)
    Q[prod_well_idx[0], prod_well_idx[1]] = -q
    ##########################################################
    # darcy flow solver ======================================
    pres_flat, pres = darcy_solver(conduct, Q, P, nx, ny, dx,
                                   dy, Pr, Pe, 2, BHP, 
                                   prod_well_idx, mu, WI)

    ##########################################################
    # save the pressure field ================================
    positions = inputpar.positions
    
    ##########################################################
    # Save the pressure field ================================
    data = pres_flat / mega
#    file_name = './' + expname + '/pressure.dat'
#    save_pressure(data, file_name)
    end_timing = time.time()
    seconds    = end_timing - start_timing

    # hours, minutes, \
    # seconds = format_seconds_to_hhmmss(seconds, 'Simulation')
    ##########################################################
    # Plot the pressure field ================================
#    plot_pres(pres / mega, Lx, Ly)
    return data
###############################################################################

###############################################################################
###############################################################################
def harmonic_mean(k1, k2):
    '''Compute the harmonic mean f k1, k2'''
    return (k1 * k2) / (k1 + k2)
    
###############################################################################
###############################################################################
def save_pressure(tht, filen):
    '''Description: Save the pressure field'''
    with open(filen, 'wb') as f:
#        np.savetxt(f,tht, fmt = '%1.10e', delimiter=' ', encoding= 'utf8')
        np.save(f, tht)
    f.close()
    return None
###############################################################################
    
###############################################################################
###############################################################################
def save_pressure_bin(tht, filen):
    '''Description: Save the pressure field'''
    with open(filen, 'wb') as f:
        np.save(f,tht)
    f.close()
###############################################################################

###############################################################################
###############################################################################
def WIfunction(dx, dy, rw, conduct, idx, wells_position):
    '''Compute the well index (Peaceman)'''
    n = np.size(wells_position, axis=0)
    wi= np.zeros((n,))
    for i in range(0,n):
        wellp = (idx[wells_position[i],0],
                 idx[wells_position[i],1])
        permx = conduct[wellp[0], wellp[1]]
        permy = conduct[wellp[0], wellp[1]]
        re1   = 0.28 * np.sqrt(np.sqrt(permy/permx) * dx**2 +
                               np.sqrt(permx/permy) * dy**2)
        re2   = np.power(permy/permx, 1/4) + np.power(permx/permy, 1/4)
        re    = re1 / re2
        wi[i] = (2.0 * np.pi * np.sqrt(permx * permy)) / \
            (np.log(re/rw))
    return wi
###############################################################################

###############################################################################
###############################################################################
def hcondutivity_field(filename, nx, ny, mu, rho, beta):
    '''Generate a heterogeneous permeability field'''
    k = get_permeability_field(filename)
#    k = beta * np.exp(rho * k) / mu
    return k.reshape((nx,ny), order = 'F')
###############################################################################

###############################################################################
###############################################################################
def hydraulic_condutivity(k, nx, ny, mu, rho, beta):
    '''Generate a heterogeneous permeability field'''
    k = beta * np.exp(rho * k) / mu
    return  k.reshape((nx,ny), order = 'F')
###############################################################################

###############################################################################
###############################################################################
def hydraulic_condutivity3D(k, nx, ny, nz, mu, rho, beta):
    '''Generate a heterogeneous permeability field'''
    k = beta * np.exp(rho * k) / mu
    return  k.reshape((nx,ny,nz), order = 'F')
###############################################################################

###############################################################################
###############################################################################
def get_permeability_field(filen):
    '''Load the permeability field'''
    with open(filen, 'rb') as f:
#        perm = np.load(f)
        perm = np.loadtxt(f, dtype = 'float', encoding= 'utf8')
    f.close()
    return perm
###############################################################################

###############################################################################
###############################################################################
def get_rho(filen):
    '''Load the permeability field'''
    perm = np.loadtxt(filen, dtype = 'float', encoding= 'utf8')
    return perm
###############################################################################

###############################################################################
###############################################################################
def coord2index(lx,ly,dx,dy,nx,ny):
    ''' converts coordinates to indices '''
    i = int(lx/dx)
    j = int(ly/dy)  
    return i,j
###############################################################################

###############################################################################
###############################################################################
def coordinates3D(nx, ny, nz, Lx, Ly, Lz):
    '''Generate the grid coordinates'''
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    x = np.linspace(dx/2, Lx-dx/2, nx)
    y = np.linspace(dy/2, Ly-dy/2, ny)
    z = np.linspace(dz/2, Lz-dz/2, nz)
    coord = np.zeros((nx*ny*nz, 3))
    idx   = np.zeros((nx*ny*nz, 3), dtype = 'int')
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = (k*nx*ny) + (j*nx)+i
                coord[p,:] = x[i], y[j], z[k]
                idx[p,:]   = i, j, k
    return idx, coord
###############################################################################

###############################################################################
###############################################################################
def coordinates(nx, ny, Lx, Ly):
    '''Generate the grid coordinates'''
    dx = Lx / nx
    dy = Ly / ny
    x = np.linspace(dx/2, Lx-dx/2, nx)
    y = np.linspace(dy/2, Ly-dy/2, ny)
    coord = np.zeros((nx*ny, 2))
    idx   = np.zeros((nx*ny, 2), dtype = 'int')
    for j in range(ny):
        for i in range(nx):
            coord[j*nx+i, 0] = x[i]
            coord[j*nx+i, 1] = y[j]
            idx[j*nx+i, 0] = i
            idx[j*nx+i, 1] = j
    return idx, coord
###############################################################################

###############################################################################
###############################################################################
def well_position(coord, px, py, dx, dy, nx, ny):
    '''Find the production well position'''
    for i in range(nx*ny):
        if np.abs(coord[i, 0] - px) <= dx/2 and \
            np.abs(coord[i, 1] - py) <= dy/2:
            return i
###############################################################################

###############################################################################
###############################################################################
def darcy_solver(k, Q, P, nx, ny, dx, dy, Pr, Pe, bd,
                 BHP, prod_well_idx, mu, WI):
    '''Solve the Darcy equation via finite differences'''
    # Coefficient matrix and right-hand side vector
    A = lil_matrix((nx * ny, nx * ny))
    b = np.zeros(nx * ny)
    # Fill the coefficient matrix and right-hand side vector
    auxx = 2.0 * dy / (dx)
    auxy = 2.0 * dx / (dy)
    ## Internal elements                                                                             
    for j in range(1,ny-1):
        for i in range(1,nx-1):
            index = j * nx + i
            index_plus_x  = j * nx + i + 1
            index_minus_x = j * nx + i - 1
            index_plus_y  = (j + 1) * nx + i
            index_minus_y = (j - 1) * nx + i
        
            Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
            Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
            Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
            Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
            A[index, index]         = -(Ta + Tb + Tc + Td)
            A[index, index_plus_x]  = Ta
            A[index, index_minus_x] = Tb
            A[index, index_plus_y]  = Tc
            A[index, index_minus_y] = Td
            b[index] = Q[i, j]
    if bd == 1:
        A,b = slab_boundary_condition(A, b, k, Q, nx, 
                                      ny, dx, dy, Pr, Pe, 
                                      BHP, prod_well_idx,
                                      mu, WI)
    if bd == 2:
        A,b = fivespot_boundary_condition(A, b, k, Q, nx, ny,
                                          dx, dy, Pr, Pe, BHP,
                                          prod_well_idx, mu, WI)
    # Solve the linear system =================================================
    P_flat = spsolve(A.tocsr(), b)
    P = P_flat.reshape((nx,ny), order='F')
    return P_flat, P
###############################################################################

###############################################################################
###############################################################################
def fivespot_boundary_condition(A, b, k, Q, nx, ny, dx, dy, Pr,
                                Pe, BHP, prod_well_idx, mu, WI):
    '''Boundary conditions for the five-spot well'''
    # Coefficient matrix and right-hand side vector
    auxx = 2.0 * dy / (dx)
    auxy = 2.0 * dx / (dy)
    ## Boundary elements======================================
    ##########################################################
    # Right boundary
    i = nx - 1
    for j in range(1,ny-1):
        index = j * nx + i
        index_plus_x  = j * nx + i + 1
        index_minus_x = j * nx + i - 1
        index_plus_y  = (j + 1) * nx + i
        index_minus_y = (j - 1) * nx + i
        Ta = 0.0
        Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
        Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
        Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
        A[index, index]         = -(Ta + Tb + Tc + Td)
        A[index, index_minus_x] = Tb
        A[index, index_plus_y]  = Tc
        A[index, index_minus_y] = Td
        b[index] = Q[i, j]
    ##########################################################
    ## Left boundary
    i = 0
    for j in range(1,ny-1):
        index = j * nx + i
        index_plus_x  = j * nx + i + 1
        index_minus_x = j * nx + i - 1
        index_plus_y  = (j + 1) * nx + i
        index_minus_y = (j - 1) * nx + i
        Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
        Tb = 0.0
        Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
        Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
        A[index, index]         = -(Ta + Tb + Tc + Td)
        A[index, index_plus_x]  = Ta
        A[index, index_plus_y]  = Tc
        A[index, index_minus_y] = Td
        b[index] = Q[i, j]
    ##########################################################
    ## Top boundary
    j = ny - 1
    for i in range(1,nx-1):
        index = j * nx + i
        index_plus_x  = j * nx + i + 1
        index_minus_x = j * nx + i - 1
        index_plus_y  = (j + 1) * nx + i
        index_minus_y = (j - 1) * nx + i

        Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
        Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
        Tc = 0.0
        Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
        A[index, index]         = -(Ta + Tb + Tc + Td)
        A[index, index_plus_x]  = Ta
#        A[index, index_plus_y]  = Tc
        A[index, index_minus_x] = Tb
        A[index, index_minus_y] = Td
        b[index] = Q[i, j]
    ##########################################################
    ## Botton boundary
    j = 0
    for i in range(1,nx-1):
        index = j * nx + i
        index_plus_x = j * nx + i + 1
        index_minus_x = j * nx + i - 1
        index_plus_y = (j + 1) * nx + i

        Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
        Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
        Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
        Td = 0.0
        A[index, index]         = -(Ta + Tb + Tc + Td)
        A[index, index_plus_x]  = Ta
        A[index, index_minus_x] = Tb
        A[index, index_plus_y]  = Tc
#        A[index, index_minus_y] = Td
        b[index] = Q[i, j]
    ##########################################################
    # Right-top corner
    i = nx - 1
    j = ny - 1
    index = j * nx + i
    index_minus_x = j * nx + i - 1
    index_minus_y = (j - 1) * nx + i
    Ta = 0.0
    Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
    Tc = 0.0
    Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
    A[index, index]         = -(Ta + Tb + Tc + Td + WI[1]/mu)
#    A[index, index_plus_x]  = Ta
    A[index, index_minus_x] = Tb
#    A[index, index_plus_y]  = Tc
    A[index, index_minus_y] = Td
#    b[index] = Q[i, j] - BHP[1] * (Ta + Tc)
    b[index] = Q[i, j] - BHP[1] * WI[1]/mu
    ##########################################################
    # Left-top corner
    i = 0
    j = ny - 1
    index = j * nx + i
    index_plus_x = j * nx + i + 1
    index_minus_y = (j - 1) * nx + i
    Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
    Tb = 0.0
    Tc = 0.0
    Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
    A[index, index]         = -(Ta + Tb + Tc + Td + WI[2]/mu)
    A[index, index_plus_x]  = Ta
#    A[index, index_minus_x] = Tb
#    A[index, index_plus_y]  = Tc
    A[index, index_minus_y] = Td
#    b[index] = Q[i, j] - BHP[2] * (Tb + Tc)
    b[index] = Q[i, j] - BHP[2] * (WI[2]/mu)
    ##########################################################
    # Left-botton corner
    i = 0
    j = 0
    index = j * nx + i
    index_plus_x = j * nx + i + 1
    index_plus_y = (j + 1) * nx + i
    Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
    Tb = 0.0
    Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
    Td = 0.0
    A[index, index]         = -(Ta + Tb + Tc + Td + WI[3]/mu)
    A[index, index_plus_x]  = Ta
#    A[index, index_minus_x] = Tb
    A[index, index_plus_y]  = Tc
#    A[index, index_minus_y] = Td
#    b[index] = Q[i, j] - BHP[3] * (Tb + Td)
    b[index] = Q[i, j] - BHP[3] * (WI[3]/mu)
    ##########################################################
    # Right-botton corner
    i = nx - 1
    j = 0
    index = j * nx + i
    index_minus_x = j * nx + i - 1
    index_plus_y = (j + 1) * nx + i
    Ta = 0.0
    Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
    Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
    Td = 0.0
    A[index, index]         = -(Ta + Tb + Tc + Td + WI[4]/mu)
#    A[index, index_plus_x]  = Ta
    A[index, index_minus_x] = Tb
    A[index, index_plus_y]  = Tc
#    A[index, index_minus_y] = Td
#    b[index] = Q[i, j] - BHP[4] * (Ta + Td)
    b[index] = Q[i, j] - BHP[4] * (WI[4]/mu)\
    ##########################################################
    ##########################################################
    return A, b
###############################################################################

###############################################################################
###############################################################################
def slab_boundary_condition(A, b, k, Q, nx, ny, dx, dy, Pr,
                            Pe, BHP, prod_well_idx, mu, WI):
    '''Boundary conditions for the slab problem'''
    auxx = 2.0 * dy / (dx)
    auxy = 2.0 * dx / (dy)
    ## Boundary elements
    ##########################################################
    # Right boundary
    i = nx - 1
    for j in range(1,ny-1):
        index = j * nx + i
        index_plus_x  = j * nx + i + 1
        index_minus_x = j * nx + i - 1
        index_plus_y  = (j + 1) * nx + i
        index_minus_y = (j - 1) * nx + i
        Ta = harmonic_mean(k[i, j] , k[i, j]) * auxx * 2.0
        Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
        Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
        Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
        A[index, index]         = -(Ta + Tb + Tc + Td)
        A[index, index_minus_x] = Tb
        A[index, index_plus_y]  = Tc
        A[index, index_minus_y] = Td
        b[index] = Q[i, j] - Pr * Ta
    ##########################################################
    ## Left boundary
    i = 0
    for j in range(1,ny-1):
        index = j * nx + i
        index_plus_x  = j * nx + i + 1
        index_minus_x = j * nx + i - 1
        index_plus_y  = (j + 1) * nx + i
        index_minus_y = (j - 1) * nx + i
        Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
        Tb = harmonic_mean(k[i, j] , k[i, j]) * auxx * 2.0
        Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
        Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
        A[index, index]         = -(Ta + Tb + Tc + Td)
        A[index, index_plus_x]  = Ta
        A[index, index_plus_y]  = Tc
        A[index, index_minus_y] = Td
        b[index] = Q[i, j] - Pe * Tb
    ##########################################################
    ## Top boundary
    j = ny - 1
    for i in range(1,nx-1):
        index = j * nx + i
        index_plus_x  = j * nx + i + 1
        index_minus_x = j * nx + i - 1
        index_plus_y  = (j + 1) * nx + i
        index_minus_y = (j - 1) * nx + i

        Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
        Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
        Tc = 0.0
        Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
        A[index, index]         = -(Ta + Tb + Tc + Td)
        A[index, index_plus_x]  = Ta
#        A[index, index_plus_y]  = Tc
        A[index, index_minus_x] = Tb
        A[index, index_minus_y] = Td
        b[index] = Q[i, j]
    ##########################################################
    ## Botton boundary
    j = 0
    for i in range(1,nx-1):
        index = j * nx + i
        index_plus_x = j * nx + i + 1
        index_minus_x = j * nx + i - 1
        index_plus_y = (j + 1) * nx + i

        Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
        Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
        Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
        Td = 0.0
        A[index, index]         = -(Ta + Tb + Tc + Td)
        A[index, index_plus_x]  = Ta
        A[index, index_minus_x] = Tb
        A[index, index_plus_y]  = Tc
#        A[index, index_minus_y] = Td
        b[index] = Q[i, j]
    ##########################################################
    # Right-top corner
    i = nx - 1
    j = ny - 1
    index = j * nx + i
    index_minus_x = j * nx + i - 1
    index_minus_y = (j - 1) * nx + i
    Ta = harmonic_mean(k[i, j] , k[i, j]) * auxx * 2.0
    Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
    Tc = 0.0
    Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
    A[index, index]         = -(Ta + Tb + Tc + Td)
#    A[index, index_plus_x]  = Ta
    A[index, index_minus_x] = Tb
#    A[index, index_plus_y]  = Tc
    A[index, index_minus_y] = Td
    b[index] = Q[i, j] - Pr * Ta
    ##########################################################
    # Left-top corner
    i = 0
    j = ny - 1
    index = j * nx + i
    index_plus_x = j * nx + i + 1
    index_minus_y = (j - 1) * nx + i
    Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
    Tb = harmonic_mean(k[i, j] , k[i, j]) * auxx * 2.0
    Tc = 0.0
    Td = harmonic_mean(k[i, j-1] , k[i, j]) * auxy
    A[index, index]         = -(Ta + Tb + Tc + Td)
    A[index, index_plus_x]  = Ta
#    A[index, index_minus_x] = Tb
#    A[index, index_plus_y]  = Tc
    A[index, index_minus_y] = Td
    b[index] = Q[i, j] - Pe * Tb
    ##########################################################
    # Left-botton corner
    i = 0
    j = 0
    index = j * nx + i
    index_plus_x = j * nx + i + 1
    index_plus_y = (j + 1) * nx + i
    Ta = harmonic_mean(k[i+1, j] , k[i, j]) * auxx
    Tb = harmonic_mean(k[i, j] , k[i, j]) * auxx * 2.0
    Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
    Td = 0.0
    A[index, index]         = -(Ta + Tb + Tc + Td)
    A[index, index_plus_x]  = Ta
#    A[index, index_minus_x] = Tb
    A[index, index_plus_y]  = Tc
#    A[index, index_minus_y] = Td
    b[index] = Q[i, j] - Pe * Tb
    ##########################################################
    # Right-botton corner
    i = nx - 1
    j = 0
    index = j * nx + i
    index_minus_x = j * nx + i - 1
    index_plus_y = (j + 1) * nx + i
    Ta = harmonic_mean(k[i, j] , k[i, j]) * auxx  * 2.0
    Tb = harmonic_mean(k[i-1, j] , k[i, j]) * auxx
    Tc = harmonic_mean(k[i, j+1] , k[i, j]) * auxy
    Td = 0.0
    A[index, index]         = -(Ta + Tb + Tc + Td)
#    A[index, index_plus_x]  = Ta
    A[index, index_minus_x] = Tb
    A[index, index_plus_y]  = Tc
#    A[index, index_minus_y] = Td
    b[index] = Q[i, j] - Pr * Ta
    ##########################################################
    return A, b
###############################################################################

###############################################################################
###############################################################################
def plot_pres(pres, Lx, Ly):
    ax = plt.imshow(pres.T, extent=[0, Lx, 0, Ly], origin='lower',
               cmap='jet')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
###############################################################################

###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":
    '''
    simul_setup ='fivespot2D'
    beta = 9.8692e-14
    rho  = 1.
    mu   = 1.0e-03
    Dom  = [100., 100., 1.]
    mesh = [4, 4, 3]
    BHP  = 101325.0
    PL   = 0.
    PR   = 0.
    rw   = 0.125
    q    = 100.
    pos  = [255, 755, 1255, 1755, 2255,  265,  765, 1265, 1765, 2265,  275, 775,
            1275, 1775, 2275,  285,  785, 1285, 1785, 2285,  295,  795, 1295, 1795, 2295]
    inputpar = simulationpar(simul_setup,beta,rho,mu,Dom,mesh,BHP,PR,PL,rw,q,pos,0)
    slab3D(inputpar, fieldY = 0.0)
    '''
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    simul_setup ='fivespot2D'
    beta = 9.8692e-14
    rho  = 1.
    mu   = 1.0e-03
    Dom  = [100., 100., 1.]
    mesh = [50, 50, 1]
    BHP  = 101325.0
    PL   = 0.
    PR   = 0.
    rw   = 0.125
    q    = 100.
    pos  = [255, 755, 1255, 1755, 2255,  265,  765, 1265, 1765, 2265,  275, 775,
            1275, 1775, 2275,  285,  785, 1285, 1785, 2285,  295,  795, 1295, 1795, 2295]
    inputpar = simulationpar(simul_setup,beta,rho,mu,Dom,mesh,BHP,PR,PL,rw,q,pos,0)
    nx = mesh[0]
    ny = mesh[1]
    filen = 'YS_ref.dat'
    Y  = get_permeability_field(filen)
    #Y = np.log(np.arange(1,nx*ny+1,dtype='float'))
    #Y    = np.ones((mesh[0]*mesh[1],))
    p    = five_spot_simulation(inputpar, fieldY = Y)

    d = np.prod(mesh)
    P_flat = np.arange(0,d,1, dtype=float)



# conduct = hydraulic_condutivity(Y, nx, ny, mu, rho, beta)
# plot_pres(conduct, N, 100)

    print(p)
    '''
    simul_setup ='slab2D'
    beta = 1.
    rho  = 1.
    mu   = 1.0e-0
    Dom  = [1., 1., 1.]
    mesh = [16, 16, 1]
    BHP  = 101325.0
    PL   = 1.
    PR   = 0.
    rw   = 0.125
    q    = 100.
    pos  = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 79, 80, 82, 84, 86, 88, 90, 92, 94, 97, 99, 101, 103, 105, 107, 109, 111, 112, 114, 116, 118, 120, 122, 124, 126, 129, 131, 133, 135, 137, 139, 141, 143, 144, 146, 148, 150, 152, 154, 156, 158, 161, 163, 165, 167, 169, 171, 173, 175, 176, 178,180, 182, 184, 186, 188, 190, 193, 195, 197, 199, 201, 203, 205, 207, 208, 210, 212, 214, 216, 218, 220, 222, 225, 227, 229, 231, 233, 235, 237, 239, 240, 242, 244, 246, 248, 250, 252, 254]
    inputpar = simulationpar(simul_setup,beta,rho,mu,Dom,mesh,BHP,PR,PL,rw,q,pos,0)
    nx = mesh[0]
    ny = mesh[1]
    filen = '../../simulDD_py/expref/fields/YS_ref.dat'
    Y  = get_permeability_field(filen)
    #Y = np.log(np.arange(1,nx*ny+1,dtype='float'))
    #Y    = np.ones((mesh[0]*mesh[1],))
    p    = slab_simulation(inputpar, fieldY = Y)

    d = np.prod(mesh)
    P_flat = np.arange(0,d,1, dtype=float)



    conduct = hydraulic_condutivity(Y, nx, ny, mu, rho, beta)
    plot_pres(conduct, 1, 1)

    print(p)



    P = np.zeros((nx,ny), dtype=float)
    start_timing = time.time()    
    for j in range(ny):
            aux = j * nx
            for i in range(nx):
                index = aux + i
                P[i, j] = P_flat[index]
    end_timing = time.time()
    seconds    = end_timing - start_timing
    hours, minutes, \
    seconds = format_seconds_to_hhmmss(seconds, 'Simulation')

    start_timing = time.time()  
    P2 = P_flat.reshape((nx,ny), order='F')
    end_timing = time.time()
    seconds    = end_timing - start_timing
    hours, minutes, \
    seconds = format_seconds_to_hhmmss(seconds, 'Simulation')

    print(P-P2)
    nx=8
    ny=2
    Y = np.arange(1,nx*ny+1,dtype='float')
    c = Y.reshape((nx,ny), order = 'F')
    plot_pres(c, 200, 100)
    '''
