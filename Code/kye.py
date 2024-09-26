# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:27:59 2023

@author: kyebchoo
"""
#%% 
'''
PLEASE READ THE SUBMITTED 'README' DOCUMENT BEFORE RUNNING THIS CODE

To run the code for the project, key in the following in your kernal:
    run_all()

For more information on other functions, refer to the README.
'''
#%%


'''
PRELIMINARY : Import required packages and declaring global constants
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.optimize as op
import scipy.integrate as int
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")

# given values for the project
n1 = 1.52
n2 = 1.495
a = 4.5e-6 # microns
wavelength = 1250e-9 # nanometers



#%%
'''
QUESTION 1 : Calculate V-parameter
'''
k0 = 2 * np.pi / wavelength
v = a * k0 * np.sqrt( n1*n1 - n2*n2)
mu0 = 4e-7 * np.pi 
epsilon0 = 8.85418782E-12
omega = 0

def calculate_V_parameter(n1, n2, wavelength, a):
    """
    Calculate the V-parameter of the waveguide.

    Parameters
    ----------
    n1 : FLOAT
        The refractive index of the core.
    n2 : FLOAT
        The refractive index of the cladding.
    wavelength : FLOAT
        The wavelength of the signal.
    a : FLOAT
        The core radius of the wwaveguide.

    Returns
    -------
    None.
    """
    global k0, v, omega
    k0 = 2 * np.pi / wavelength
    v = a * k0 * np.sqrt( n1*n1 - n2*n2)
    print(f'V-parameter: {v:.2f}')

    omega = k0/((epsilon0 * mu0)**0.5)
    pass




#%%
'''
QUESTION 2 : We decided that we will be analysing the {TE, TM, EH, HE} modes of the waveguide.
'''



#%%
'''
QUESTION 3 : Identify supported modes, and the associated propagation constant & effective index
'''


# rearranged versions of the characteristic equation
def f_plus(pa, m): 
    """
    The function returns the output of the rearranged characteristic equation by moving all terms to the left-hand-side. This is to be used for the EH modes.

    Parameters
    ----------
    pa : FLOAT
        The pa-value equivalent to the x-axis.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.

    Returns
    -------
    output : FLOAT
        Returns the value of the equation.
    """
    qa = np.sqrt(v*v - pa*pa)
    output = qa * sp.kv(m, qa) * sp.jv(m+1, pa) + pa * sp.jv(m, pa) * sp.kv(m+1, qa)
    return output



def f_minus(pa, m): 
    """
    The function returns the output of the rearranged characteristic equation by moving all terms to the left-hand-side. This is to be used for the HE modes.

    Parameters
    ----------
    pa : FLOAT
        The pa-value equivalent to the x-axis.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.

    Returns
    -------
    output : FLOAT
        Returns the value of the equation.
    """
    qa = np.sqrt(v*v - pa*pa)
    output = qa * sp.kv(m, qa) * sp.jv(m-1, pa) - pa * sp.jv(m, pa) * sp.kv(m-1, qa)
    return output



def f_TM(pa, m):
    """
    The function returns the output of the rearranged characteristic equation by moving all terms to the left-hand-side. This is to be used for the TM modes.
    This function does not make the assumption of n1 ~ n2 and thus would be more accurate than the previous functions.

    Parameters
    ----------
    pa : FLOAT
        The pa-value equivalent to the x-axis.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.

    Returns
    -------
    output : FLOAT
        Returns the value of the equation.
    """
    qa = np.sqrt(v**2 - pa**2)
    output = (n1**2) * qa * sp.kv(m, qa) * sp.jv(m + 1, pa) + (n2**2) * pa * sp.jv(m, pa) * sp.kv(m + 1, qa)
    return output
    
    

# separating the LHS and RHS to visually see the intersections
# THIS NEEDS THE POSITIVE AND NEGATIVE DIFFERENTIATION ADDING
def lhs(pa, m):
    """
    Output the left-hand-side of the general equation, while removing the x-axis of the asymptotes and returning it in a seperate array.

    Parameters
    ----------
    pa : FLOAT
        Input parameter. Could be a float or an array.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.

    Returns
    -------
    pa_clean : FLOAT
        An array containing the x-axis used in the plotting, with the asymptotes removed.
    lhs_array_clean : FLOAT
        An array containing the value of the function, with the asymptotes removed.
    asymptotes_pa : FLOAT
        An array containing all the removed asymptotes.
    """
    lhs_array = sp.jv(m+1, pa) / (pa * sp.jv(m, pa))
    
    # locate and remove the asymptotes based on sign changes
    # based on shape, asymptote requires there to be a sign change potive to negative
    sign_changes = np.where(np.diff(np.sign(lhs_array)))[0] # not sure what the where statement 
    asymptote_indices = [index+1 for index in sign_changes if lhs_array[index]>0 and lhs_array[index+1]<0]
    asymptotes_pa = pa[asymptote_indices]
    pa_clean = np.delete(pa, asymptote_indices)
    lhs_array_clean = np.delete(lhs_array, asymptote_indices)

    return pa_clean, lhs_array_clean, asymptotes_pa



def rhs(pa, m):
    """
    Output the right-hand-side of the general equation, while removing the x-axis of the asymptotes and returning it in a seperate array.

    Parameters
    ----------
    pa : FLOAT
        Input parameter. Could be a float or an array.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.

    Returns
    -------
    pa_clean : FLOAT
        An array containing the x-axis used in the plotting, with the asymptotes removed.
    rhs_array_clean : FLOAT
        An array containing the value of the function, with the asymptotes removed.
    asymptotes_pa : FLOAT
        An array containing all the removed asymptotes.
    """
    qa = np.sqrt(v*v - pa*pa)
    output = -sp.kv(m+1, qa) / (qa * sp.kv(m, qa))
    return output 



# function that creates nice plots with graphical solutions
def graphical_solve(m, pa, solutions = 0, sol_values = 0, labels = 0):
    """
    Routine to get intersection of the general equation, and to return the pa with asymptotes removed.

    Parameters
    ----------
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.
    pa : FLOAT
        The order of the Bessel and modified Bessel function of the second kind.
    solutions : INTEGER, optional
        Number of plotting solutions already found by solver. The default is 0.
    sol_values : FLOAT, optional
        Value of plotting solutions already found by solver. The default is 0.
    labels : STRING, optional
        Labelling for the solution found. The default is 0.

    Returns
    -------
    pa_clean : FLOAT
        An array containing the x-axis used in the plotting, with the asymptotes removed..

    """
    pa_clean, lhs_array_clean, asymptotes = lhs(pa, m)
    
    plt.plot(pa_clean, lhs_array_clean, 'x', color= 'black', label= 'first order')
    plt.plot(pa, rhs(pa, m), 'x', color = 'blue', label='second order')
    plt.axvline(x=v, color='black', linestyle='--')
    plt.text(v, 0.5, 'V = {:.2f}'.format(v), rotation=90, color='black', ha='right', va='bottom')

    for asymptote_pa in asymptotes:
        plt.axvline(x=asymptote_pa, color='red', linestyle='--')
        plt.text(asymptote_pa, 0.5, 'Asymptote ({:.2f})'.format(asymptote_pa), rotation=90, color='red', ha='right', va='bottom')
        pass
    if solutions == 1: 
        for i in range(len(sol_values)):
            plt.plot(sol_values[i], rhs(m,sol_values[i]),'.', color = 'orange',markersize=10) # plotting solutions found by solver
            plt.text(sol_values[i], -0.1, '${}$'.format(labels[i]),  color='black', ha='right', va='bottom')
            pass
        pass
    
    plt.xlabel('pa')
    plt.ylim([-5,5])
    plt.grid()
    plt.legend()
    plt.title('Mode: {}'.format(m))
    plt.show()

    return pa_clean



###############################################################################################
table_modes = pd.DataFrame(columns = ['mode', 'beta', 'n_eff', 'pa-zeroes', 'm', 'power_fraction'])

def construct_table():
    """
    Method to construct a table of all possible supported modes along with the corresponding values of propagation constant, effective index, zeroes of pa's and mode'

    Returns
    -------
    None.
    """
    # create a table in which to store the modes
    global table_modes, v
    # table_modes = pd.DataFrame(columns = ['mode', 'beta', 'n_eff', 'pa-zeroes', 'm', 'power_fraction'])
    
    # to find the TE and TM modes
    for m in range(0, 1):
        pa = np.linspace(0, v, 100) # limits for guided wave
        pa = pa[1:-1] # limits are non inclusive
        f0_TE = f_plus(pa, m)
        f0_TM = f_TM(pa, m)
        
        # find a rough estimate for the roots of the equation based on changes in sign
        binary_TE = np.sign(f0_TE[:-1]) - np.sign(f0_TE[1:])
        pa0_TE = pa[np.where(binary_TE != 0)[0]]
        if len(pa0_TE) != 0: 
            solution_TE = op.fsolve(f_plus, pa0_TE, args = m)
            for i in range(len(solution_TE)):
                mode = f'TE_{m}{i + 1}'
                beta = np.sqrt(n1*k0*n1*k0 - (solution_TE[i]/a)**2 )
                n_eff = beta / k0
                table_modes.loc[len(table_modes)] = [mode, beta, n_eff, solution_TE[i], m, '']
                pass
            pass
        
        binary_TM = np.sign(f0_TM[:-1]) - np.sign(f0_TM[1:])
        pa0_TM = pa[np.where(binary_TM != 0)[0]]
        if len(pa0_TM) != 0: 
            solution_TM = op.fsolve(f_TM, pa0_TM, args = m)
            for i in range(len(solution_TM)):
                mode = f'TM_{m}{i + 1}'
                beta = np.sqrt(n1*k0*n1*k0 - (solution_TM[i]/a)**2 )
                n_eff = beta / k0
                table_modes.loc[len(table_modes)] = [mode, beta, n_eff, solution_TM[i], m, '']
                pass
            pass
    
    
    
    # to find the rest of the modes
    for m in range(1, 10):
        pa = np.linspace(0, v, 100) # limits for guided wave
        pa = pa[1:-1] # limits are non inclusive
        f0_plus = f_plus(pa, m)
        f0_minus = f_minus(pa, m)
    
        # find a rough estimate for the roots of the equation based on changes in sign
        binary_plus = np.sign(f0_plus[:-1]) - np.sign(f0_plus[1:])
        pa0_plus = pa[np.where(binary_plus!=0)[0]]
        if len(pa0_plus) != 0: 
            solution_plus = op.fsolve(f_plus, pa0_plus, args=m)
            for i in range(len(solution_plus)):
                mode = f'EH_{m}{i+1}'
                beta = np.sqrt(n1*k0*n1*k0 - (solution_plus[i]/a)**2 )
                n_eff = beta / k0
                table_modes.loc[len(table_modes)] = [mode, beta, n_eff, solution_plus[i], m, '']
                pass
            pass
        pass
                
        binary_minus = np.sign(f0_minus[:-1]) - np.sign(f0_minus[1:])
        pa0_minus = pa[np.where(binary_minus!=0)[0]]
        if len(pa0_minus) != 0: 
            solution_minus = op.fsolve(f_minus, pa0_minus, args=m)
            for i in range(len(solution_minus)):
                mode = f'HE_{m}{i+1}'
                beta = np.sqrt(n1*k0*n1*k0 - (solution_minus[i]/a)**2 )
                n_eff = beta / k0
                table_modes.loc[len(table_modes)] = [mode, beta, n_eff, solution_minus[i], m, '']
                pass
            pass
        pass
    
    print(table_modes)
    pass
     


#%%
'''    
QUESTION 5 : Plot maps in the plane perpendicular to the fibre axis for amplitude of all three fields.
'''



def get_A_B_C_D(p, q, m, beta, set_A_value = 1):
    """
    Funtion to return the A, B, C, D coefficient of the optical fibre modes (page 7 - Lecture 3).

    Parameters
    ----------
    p : FLOAT
        p-parameter equivalent to x-axis.
    q : FLOAT
        q-parameter that could be calculated from the p-parameter.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.
    beta : FLOAT
        The propagation constant of the waveguide.
    set_A_value : FLOAT, optional
        To set the A-coefficient. The default is 1.

    Returns
    -------
    A : FLOAT
        Returns value of A-coefficient.
    B : FLOAT
        Returns value of B-coefficient.
    C : FLOAT
        Returns value of B-coefficient.
    D : FLOAT
        Returns value of D-coefficient.
    """
    
    # A value is set to unity by default
    A = set_A_value
    
    # calculating B:
    numerator = (n1**2) * (sp.jvp(m, p * a, n = 1) / (p * a * sp.jv(m, p * a))) + (n2**2) * (sp.kvp(m, q * a, n = 1) / (q * a * sp.kv(m, q * a)))
    denominator = (1j) * ((beta * m)/(omega * epsilon0)) * (((p * a)**(-2)) + ((q * a)**(-2)))

    B = -1 * A * numerator / denominator
    
    # calculating C:
    C = A * (sp.jv(m, p * a) / sp.kv(m, q * a))
    
    # calculating D:
    D = B * (sp.jv(m, p * a) / sp.kv(m, q * a))
    return A, B, C, D



def E_z(r, p, q, A, C, m):
    """
    To calculate the magnitude of the electric field vector in the z-direction.

    Parameters
    ----------
    r : FLOAT
        r-coordinate of the position.
    p : FLOAT
        p-parameter equivalent to x-axis.
    q : FLOAT
        q-parameter that could be calculated from the p-parameter.
    A : FLOAT
        A-coefficient of the solution.
    C : FLOAT
        C-coefficient of the solution.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.

    Returns
    -------
    out : FLOAT
        Magnitude of the electric field vector in the z-direction.
    """
    if r <= a:
        out = A * sp.jv(m, p * r)
        pass
    if r > a:
        out = C * sp.kv(m, q * r)
        pass
    return out



def E_r(r, p, q, A, B, C, D, m, beta):
    """
    To calculate the magnitude of the electric field vector in the r-direction.

    Parameters
    ----------
    r : FLOAT
        r-coordinate of the position.
    p : FLOAT
        p-parameter equivalent to x-axis.
    q : FLOAT
        q-parameter that could be calculated from the p-parameter.
    A : FLOAT
        A-coefficient of the solution.
    B : FLOAT
        B-coefficient of the solution.
    C : FLOAT
        C-coefficient of the solution.
    D : FLOAT
        D-coefficient of the solution.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.
    beta : FLOAT
        The propagation constant of the waveguide.

    Returns
    -------
    out : FLOAT
        Magnitude of the electric field vector in the r-direction.
    """
    if r <= a:
        out = -1 * (1j) * (beta / (p**2)) * (A * p * sp.jvp(m, p * r, n = 1) + (1j) * omega * ((mu0 * m)/(beta * r)) * B * sp.jv(m, p * r))
        pass
    if r > a:
        out = 1 * (1j) * (beta / (q**2)) * (C * q * sp.kvp(m, q * r, n = 1) + (1j) * omega * ((mu0 * m)/(beta * r)) * D * sp.kv(m, q * r))
        pass
    return out
        


def E_phi(r, p, q, A, B, C, D, m, beta):
    """
    To calculate the magnitude of the electric field vector in the phi-direction (or theta).

    Parameters
    ----------
    r : FLOAT
        r-coordinate of the position.
    p : FLOAT
        p-parameter equivalent to x-axis.
    q : FLOAT
        q-parameter that could be calculated from the p-parameter.
    A : FLOAT
        A-coefficient of the solution.
    B : FLOAT
        B-coefficient of the solution.
    C : FLOAT
        C-coefficient of the solution.
    D : FLOAT
        D-coefficient of the solution.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.
    beta : FLOAT
        The propagation constant of the waveguide.

    Returns
    -------
    out : FLOAT
        Magnitude of the electric field vector in the r-direction.
    """
    if r <= a:
        out = -1 * (1j) * (beta / (p**2)) * ((1j) * (m / r) * A * sp.jv(m, p * r) - omega * (mu0 / beta) * p * B * sp.jvp(m, p * r, n = 1))
        pass
    if r > a:
        out = 1 * (1j) * (beta / (q**2)) * ((1j) * (m / r) * C * sp.kv(m, q * r) - omega * (mu0 / beta) * q * D * sp.kvp(m, q * r, n = 1))
    return out



def E_r_phi_z(r, phi, z, m, beta):
    """
    Function to return the three vectors corresponding to the electric field in the r, z and phi direction.

    Parameters
    ----------
    r : FLOAT
        r-coordinate of the position.
    phi : FLOAT
        Angle along the circumference of the waveguide.
    z : FLOAT
        Parameter that goes along the axis of the waveguide.
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.
    beta : FLOAT
        The propagation constant of the waveguide.

    Returns
    -------
    E_r_vector : FLOAT 
        Magnitude of the electric field vector in the r-direction that varies along z and angle.
    E_phi_vector : FLOAT
        Magnitude of the electric field vector in the phi-direction that varies along z and angle.
    E_z_vector : FLOAT
        Magnitude of the electric field vector in the z-direction that varies along z and angle.
    """
    
    p = np.sqrt((n1 * k0)**2 - beta**2)
    q = np.sqrt(beta**2 - (n2 * k0)**2)
    
    suffix = np.exp((1j) * m * phi - (1j) * beta * z)
    A, B, C, D = get_A_B_C_D(p, q, m, beta, set_A_value = 1)
    
    E_z_vector = suffix * E_z(r, p, q, A, C, m)
    E_r_vector = suffix * E_r(r, p, q, A, B, C, D, m, beta)
    E_phi_vector = suffix * E_phi(r, p, q, A, B, C, D, m, beta)
    return E_r_vector, E_phi_vector, E_z_vector



def plot_E_field(m, beta, z = 0, radial_density = a / 3, plot_r_limit = 2 * a):
    """
    To plot the directional electric field lines.

    Parameters
    ----------
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.
    beta : FLOAT
        The propagation constant of the waveguide.
    z : FLOAT
        Parameter that goes along the axis of the waveguide.
    radial_density : FLOAT, optional
        Input to control how dense the plotting would be along the radial direction, i.e. how frequent the code would compute values. The final plot is an interpolation of these values. The default is a / 5.
    plot_r_limit : FLOAT, optional
        The extent of the polar plot in the r-direction. The default is 2 * a, corresponding to twice the radius of the core.

    Returns
    -------
    None.

    """
    
    azimuthal_density = 2 * np.pi * radial_density / 8
    y_array = np.arange(radial_density, plot_r_limit, radial_density)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120, subplot_kw={'projection': 'polar'})
    
    for i in range(0, len(y_array)):
        radius = y_array[i]
        circumference = 2 * np.pi * radius
        x_spacing = circumference / np.floor(circumference / azimuthal_density)
        for j in range(0, round(np.floor(circumference / azimuthal_density))):
            azimuthal = 2 * np.pi * ((x_spacing * j / circumference) - circumference / 2)
            E_r_vector, E_phi_vector, E_z_vector = E_r_phi_z(r = radius, phi = azimuthal, z = z, m = m, beta = beta)

            dr = np.real(E_r_vector)
            dt = np.real(E_phi_vector)        
            r = radius
            theta = azimuthal
            # ax.quiver(azimuthal, radius, np.real(E_phi_vector), np.real(E_r_vector), pivot = 'middle')
            ax.quiver(theta, r, dr * np.cos(theta) - dt * np.sin (theta), dr * np.sin(theta) + dt * np.cos(theta), pivot = 'middle')
            pass
        pass
    
    s = f'{mode}'
    s = s[:3] + '{' + s[3:] + '}'
    
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylim([0, plot_r_limit])
    ax.set_title(f'Cross-sectional plot of the E-fields of ${s}$')
    plt.draw()
    pass



def plot_E_component_field(m, beta, ax, fig, choice = 'E_phi', z = 0, plot_r_limit = 2 * a):
    """
    Function to plot the component electric fields in the radial, azimuthal, or z-direction.

    Parameters
    ----------
    m : INTEGER
        The order of the Bessel and modified Bessel function of the second kind.
    beta : FLOAT
        The propagation constant of the waveguide.
    choice : STRING, optional
        Choice of the electric fields to be plotted. May be chosen from 'E_phi', 'E_r', and 'E_z'. The default is 'E_phi'.
    z : FLOAT
        Parameter that goes along the axis of the waveguide.
    plot_r_limit : FLOAT, optional
        The extent of the polar plot in the r-direction. The default is 2 * a, corresponding to twice the radius of the core.

    Returns
    -------
    None.
    """
    
    theta = np.linspace(0, 2*np.pi, 200)
    r = np.linspace(0, plot_r_limit, 100)

    # Create a meshgrid for r and theta
    r, theta = np.meshgrid(r, theta)

    # Initialize the contour plot
    contour = np.empty_like(r)

    # Compute E_phi for each point in the meshgrid
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            E_r_vector, E_phi_vector, E_z_vector = E_r_phi_z(r=r[i, j], phi=theta[i, j], z=z, m=m, beta=beta)
            
            if choice == 'E_phi':
                vector = E_phi_vector
            elif choice == 'E_r':
                vector = E_r_vector
            elif choice == 'E_z':
                vector = E_z_vector
            elif choice == 'intensity':
                vector = np.abs(E_phi_vector * E_phi_vector) + np.abs(E_r_vector * E_r_vector)
            else: 
                print('something went wrong with field choice')
            contour[i, j] = np.real(vector)

    # Plot the polar contour
    if choice == 'intensity':
        c = ax.pcolor(theta, r, contour, cmap = 'Reds')
        pass
    else:
        c = ax.pcolor(theta, r, contour, cmap = 'RdGy', vmin = -3, vmax = 3)
        pass
    s = f'{mode}'
    s = s[:3] + '{' + s[3:] + '}'
    if choice == 'E_phi':
        ax.set_title(f'${s}$' + ': $E_{\phi}$')
        pass
    else:
        ax.set_title(f'${s}$: ${choice}$')
        pass
    ax.set_rticks([a]) 
    ax.grid(color='black', alpha=0.2)
    return c



def choose_E_component_fields(mode_input, ax, fig, choice = 'E_phi', z = 0, radial_density = a / 5, plot_r_limit = 2 * a):
    """
    To make life easier for the user, this function allows us to only choose the mode and the other parameters would be filled in automatically to get the component electric fields.

    Parameters
    ----------
    mode : STRING
        Choice of mode, corresponding to the elements entered into the table.
    choice : STRING, optional
        Choice of the electric fields to be plotted. May be chosen from 'E_phi', 'E_r', and 'E_z'. The default is 'E_phi'.
    z : FLOAT
        Parameter that goes along the axis of the waveguide.
    radial_density : FLOAT, optional
        Input to control how dense the plotting would be along the radial direction, i.e. how frequent the code would compute values. The final plot is an interpolation of these values. The default is a / 5.
    plot_r_limit : FLOAT, optional
        The extent of the polar plot in the r-direction. The default is 2 * a, corresponding to twice the radius of the core.

    Returns
    -------
    None.
    """
    global mode, table_modes
    mode = mode_input
    current_mode = table_modes.loc[table_modes['mode'] == mode_input]
    c = plot_E_component_field(m=current_mode['m'].values[0], beta=current_mode['beta'].values[0], ax=ax, fig=fig, choice = choice, z = 0, plot_r_limit = 2 * a)
    return c



def choose_E_vector_fields(mode_input, z = 0, radial_density = a / 3, plot_r_limit = 2 * a):
    """
    To make life easier for the user, this function allows us to only choose the mode and the other parameters would be filled in automatically to get the component electric fields.

    Parameters
    ----------
    mode : STRING
        Choice of mode, corresponding to the elements entered into the table.
    choice : STRING, optional
        Choice of the electric fields to be plotted. May be chosen from 'E_phi', 'E_r', and 'E_z'. The default is 'E_phi'.
    z : FLOAT
        Parameter that goes along the axis of the waveguide.
    radial_density : FLOAT, optional
        Input to control how dense the plotting would be along the radial direction, i.e. how frequent the code would compute values. The final plot is an interpolation of these values. The default is a / 5.
    plot_r_limit : FLOAT, optional
        The extent of the polar plot in the r-direction. The default is 2 * a, corresponding to twice the radius of the core.

    Returns
    -------
    None.
    """
    global mode, table_modes
    mode = mode_input
    current_mode = table_modes.loc[table_modes['mode'] == mode_input]
    plot_E_field(m = current_mode['m'].values[0], beta=current_mode['beta'].values[0], z = z, radial_density = radial_density, plot_r_limit = plot_r_limit)
    pass



def plot_1D_E_field(mode_input, choice = 'E_phi', plot_r_limit = 2 * a, phi = 0, z = 0):
    """
    To generate the electric field distributions with arrows to show direction. the size of the arrows are not representative of the intensities.

    Parameters
    ----------
    mode : STRING
        Choice of mode, corresponding to the elements entered into the table.
    choice : STRING, optional
        Choice of the electric fields to be plotted. May be chosen from 'E_phi', 'E_r', and 'E_z'. The default is 'E_phi'.
    plot_r_limit : FLOAT, optional
        The extent of the polar plot in the r-direction. The default is 2 * a, corresponding to twice the radius of the core.
    phi : FLOAT, optional
        Selection of what angle the 1D cut is taken from. The default is 0.
    z : FLOAT
        Parameter that goes along the axis of the waveguide.

    Returns
    -------
    None.
    """
    current_mode = table_modes.loc[table_modes['mode'] == mode_input]
    r_array = np.arange(0, plot_r_limit, plot_r_limit/1000)
    
    field_value_array_real = []
    field_value_array_imaginary = []
    for i in range(0, len(r_array)):
        m = current_mode['m'].values[0]
        beta = current_mode['beta'].values[0]
        E_r_vector, E_phi_vector, E_z_vector = E_r_phi_z(r_array[i], phi = phi, z = z, m = m, beta = beta)
        if choice == 'E_phi':
            vector = E_phi_vector
            pass
        elif choice == 'E_r':
            vector = E_r_vector
            pass
        elif choice == 'E_z':
            vector = E_z_vector
            pass
        elif choice == 'intensity':
            vector = np.abs(E_phi_vector * E_phi_vector) + np.abs(E_r_vector * E_r_vector)
            pass
        else: 
            print('something went wrong with field choice')
            pass
        
        field_value_array_real.append(np.real(vector))
        field_value_array_imaginary.append(np.imag(vector))
        pass
    
    # Formatting the input strings for plotting (non-essential for core)
    if choice == 'E_phi':
        choice = choice[:2] + '{' + choice[2:] + '}'
        pass
    s = f'{mode_input}'
    s = s[:3] + '{' + s[3:] + '}'
    
    # Plotting
    fig, ax = plt.subplots()
    plt.title(f'${s}$: ${choice}$')
    ax.plot(r_array, field_value_array_real, label = 'Real Component', color = 'blue')
    ax.plot(r_array, field_value_array_imaginary, label = 'Imaginary Component', color = 'red', linestyle = 'dashed')
    plt.grid()
    plt.legend()
    plt.xlabel('Radius/m')
    plt.ylabel('Magnitude')
    plt.show()        
    pass



def plot_intensities_of_vector_fields():
    """
    To plot the intensities electric fields of the various modes.

    Returns
    -------
    None.
    """
    modes = table_modes['mode'].values[4:] # skipping the TE and TM modes
    
    global mode
    for mode in modes:
        print(mode)
        choices = list(['E_phi', 'E_r', 'E_z'])
        fig, axes = plt.subplots(figsize = (9, 5), nrows=1, ncols=3, subplot_kw={'projection': 'polar'})
        fig.tight_layout()
        # contour_max = 0
        for ax,choice in zip(axes.ravel().tolist(),choices):
            c = choose_E_component_fields(mode, ax, fig, choice)
            pass
        fig.colorbar(c, ax=axes.ravel().tolist(), orientation='horizontal', location='bottom')
        plt.show()
        pass
    pass


def plot_HE_22():
    """
    Plotting for HE_22.

    Returns
    -------
    None.
    """
    mode = 'HE_22'
    choices = list(['E_phi', 'E_r', 'E_z'])
    fig, axes = plt.subplots(figsize = (9, 5), nrows=1, ncols=3, subplot_kw={'projection': 'polar'})
    fig.tight_layout()
    # contour_max = 0
    for ax,choice in zip(axes.ravel().tolist(),choices):
        c = choose_E_component_fields(mode, ax, fig, choice)
        pass
    fig.colorbar(c, ax=axes.ravel().tolist(), orientation='horizontal', location='bottom')
    plt.show()
    pass



def plot_remaining():
    """
    Plotting the rest of the modes and formatting it into a table of figures.

    Returns
    -------
    None.
    """
    modes_raw = table_modes['mode'].values[4:]   
    modes_raw = np.delete(modes_raw, np.where(modes_raw == 'HE_22'))    # skipping the TE and TM modes
    modes = np.repeat(modes_raw, 3)
    choices_raw = list(['E_phi', 'E_r', 'E_z'])
    choices = choices_raw * 7
    
    fig, axes = plt.subplots(figsize = (63, 4), nrows = 1, ncols = 21, subplot_kw={'projection': 'polar'})
    fig.tight_layout()
    for mode, ax, choice in zip(modes, axes.ravel().tolist(), choices):
        c = choose_E_component_fields(mode, ax, fig, choice)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pass
    fig.colorbar(c, ax=axes.ravel().tolist(), orientation='horizontal', location='bottom', aspect = 63)
    plt.show()
    pass
    
#%%
'''
QUESTION 6 : Spatial Distribution of Intensity
'''



def intensity(E_phi, E_r):
    """
    To calculate the intensity of the electric field using only the radial and azimuthal vectors.

    Parameters
    ----------
    E_phi : FLOAT
        The vector in the azimuthal direction.
    E_r : FLOAT
        The vector in the radial direction.

    Returns
    -------
    intensity : FLOAT
        The value of the intensity.
    """
    intensity = np.abs(E_phi * E_phi) + np.abs(E_r * E_r) 
    return intensity



def plot_intensity(mode_input, z = 0, radial_density = a / 5, plot_r_limit = 2 * a, plot = True):
    """
    Function to plot the intensity of the radial and azimuthal electric fields in the waveguide.

    Parameters
    ----------
    mode : STRING
        Choice of mode, corresponding to the elements entered into the table.
    z : FLOAT
        Parameter that goes along the axis of the waveguide.
    radial_density : FLOAT, optional
        Input to control how dense the plotting would be along the radial direction, i.e. how frequent the code would compute values. The final plot is an interpolation of these values. The default is a / 5.
    plot_r_limit : FLOAT, optional
        The extent of the polar plot in the r-direction. The default is 2 * a, corresponding to twice the radius of the core.

    Returns
    -------
    None.

    """
    global mode
    mode = mode_input
    
    # Define polar grid
    theta = np.linspace(0, 2*np.pi, 200)
    r = np.linspace(0, plot_r_limit, 100)

    # Create a meshgrid for r and theta
    r, theta = np.meshgrid(r, theta)
    
    # Plotting
    if plot == True:
        fig, ax = plt.subplots(figsize = (6.3, 6), subplot_kw={'projection': 'polar'})
        fig.tight_layout()
        c = choose_E_component_fields(mode, ax = ax, fig = fig, choice = 'intensity', z = z, radial_density = radial_density, plot_r_limit = plot_r_limit)
        fig.colorbar(c)
        plt.show()
        pass

    current_mode = table_modes.loc[table_modes['mode'] == mode]

    core_intensity = 0 
    total_intensity = 0 

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            E_r_vector, E_phi_vector, E_z_vector = E_r_phi_z(r=r[i, j], phi=theta[i, j], z=z, m=current_mode['m'].values[0], beta=current_mode['beta'].values[0])
            I = intensity(E_phi_vector, E_r_vector)
            # contour[i,j] = np.real(I)
            total_intensity += np.nan_to_num(I)
            if r[i,j] < a:
                core_intensity += np.nan_to_num(I)
                pass
            pass
        pass
    
    fractional_power = np.real(core_intensity) / np.real(total_intensity)
    table_modes['power_fraction'].loc[table_modes['mode'] == mode] = fractional_power
    pass


def plot_intensities_all(plot = False):
    """
    To generate plots of intensities for all the modes.

    Returns
    -------
    None.
    """
    for modes in table_modes['mode'].values[4:]:
        plot_intensity(mode_input = modes, z = 0, radial_density = a / 5, plot_r_limit = 2 * a, plot = plot)
        pass
    pass



#%%
'''
QUESTION 7 : Calculate the waveguide dispersion numerically
'''



def n_m_function(wavelengths, m, l, plus):
    """
    Return the n_m_function value for use by other methods.

    Parameters
    ----------
    wavelengths : FLOAT
        Wavelength of the signal given in meters.
    mode : STRING
        Choice of mode, corresponding to the elements entered into the table.
    l : INTEGER
        To input the 'l' of the waveguide.
    plus : INTEGER
        Choice of '1' or otherwise to choose the equation used.

    Returns
    -------
    n_m : FLOAT
        Effective index of the waveguide.
    """
    n_m = []
    for wavelength in wavelengths: 
        k0 = 2 * np.pi / wavelength
        v = a * k0 * np.sqrt( n1*n1 - n2*n2)

        pa = np.linspace(0, v, 100) # limits for guided wave
        pa = pa[1:-1] # limits are non inclusive
        
        # obtaining the filtered pa 
        paf, _ , _ = lhs(pa, m)
    
        if plus == 1:

            f0_plus = f_plus(paf,m)
            # find a rough estimate for the roots of the equation based on changes in sign
            binary_plus = np.sign(f0_plus[:-1]) - np.sign(f0_plus[1:])
            pa0_plus = paf[np.where(binary_plus!=0)[0]]
            pass
            
            if len(pa0_plus) != 0: 
                solution_plus = op.fsolve(f_plus, pa0_plus, args=(m))
                beta = np.sqrt(n1*k0*n1*k0 - (solution_plus[l-1]/a)**2 )
                n_eff = beta / k0
                n_m.append(n_eff)
                pass
        else: 
            f0_minus = f_minus(paf, m, v)
            binary_minus = np.sign(f0_minus[:-1]) - np.sign(f0_minus[1:])
            pa0_minus = paf[np.where(binary_minus!=0)[0]]
            pass

            if len(pa0_minus) != 0: 
                solution_minus = op.fsolve(f_minus, pa0_minus, args=(m))     
                beta = np.sqrt(n1*k0*n1*k0 - (solution_minus[l-1]/a)**2 )
                n_eff = beta / k0
                n_m.append(n_eff)
                pass
    return np.array(n_m)



def s2nd_derivative(f, x, dx, m, l, plus):
    """
    To calculate the second derivative of s.

    Parameters
    ----------
    f : FUNCTION
        Input function fpr derivative.
    x : FLOAT
        x-value where the derivative is to be found.
    dx : FLOAT
        Step size in the x-axis.
    m : STRING
        Choice of mode, corresponding to the elements entered into the table.
    l : INTEGER
        To input the 'l' of the waveguide.
    plus : Integer
        Choice of '1' or 'otherwise to choose the equation used.

    Returns
    -------
    d2y_dx2 : FLOAT
        The second derivative of the function.
    """
    # Calculate the second-order derivative using finite differences
    d2y_dx2 = (f(x + dx, m, l, plus) - 2 * f(x, m, l, plus) + f(x - dx, m, l, plus)) / (dx**2)

    return d2y_dx2  



def dispersion(true_wavelength, m, l, plus, range_input):
    """
    Generates the dispersion values.

    Parameters
    ----------
    true_wavelength : FLOAT
        The true wavelength of the signal.
    m : STRING
        Choice of mode, corresponding to the elements entered into the table.
    l : INTEGER
        To input the 'l' of the waveguide.
    plus : Integer
        Choice of '1' or 'otherwise to choose the equation used.
    range_input : FLOAT
        Range to plot the graph.

    Returns
    -------
    D_w : FLOAT
        Returns the dispersion value.
    """
    wavelengths = np.linspace(true_wavelength - range_input/2, true_wavelength + range_input/2, 10)
    # want to calculte n_m for various wavelengths 
    n_m = n_m_function(wavelengths, m, l, plus)

    # Perform linear fit
    coefficients = np.polyfit(n_m, wavelengths, 1)
    linear_fit = np.polyval(coefficients, n_m)

    # Plot the linear fit
    plt.figure()
    plt.plot(wavelengths,n_m,'.',color='black')
    plt.plot(linear_fit, n_m, '--', color='red', label='Linear Fit')
    # plt.plot(fitted_curve, n_m, '--', color='purple', label='Exponential Fit')
    plt.ylabel('N$_{m}$')
    plt.xlabel('Wavelengths (m)')
    plt.title('Graph Q7')
    plt.legend()
    plt.grid()
    plt.show()
      
    # defining the speed of light 
    c = 3e8
    dy_dx = np.gradient(n_m, wavelengths)
    
    plt.figure()
    plt.plot(wavelengths,dy_dx,'.',color='black')
    plt.ylabel('dN$_{m}$ / d$\lambda$')
    plt.xlabel('Wavelengths (m)')
    plt.title('Graph Q7')
    plt.grid()
    plt.show()

    # Calculating second derivative using finite difference 
    d2n_dlambda = s2nd_derivative(n_m_function, np.array([wavelength]), range_input, m, l, plus)

    #calculating the waveguide dipersion 
    D_w = - (true_wavelength / c) * d2n_dlambda

    return D_w 



def run_dispersion(wavelength):
    D_w = dispersion(wavelength,0 ,1 ,plus = 1 , range_input = 10e-9)
    print('D_w : ', D_w[0])
    pass



#%%
'''
QUESTION 8 : Approximating effective index and the fraction of P0_omega_ER in the core (approximate and actual)
'''



# b = ( 1.1428 - 0.996 / v ) ** 2

# Calculating effective indexes
q8_table = pd.DataFrame(columns = ['mode', 'n_eff', 'n_eff_approx', 'n_eff_percent_diff', 'gamma', 'gamma_approx', 'gamma_approx_percent_diff'])



def q8_values(mode_input):
    """
    This function created a table with the values of estimated and calculated effective index, plus the fraction of the power carried in the core calculated from the intensity distribution and the approximate formula.

    Parameters
    ----------
    mode : STRING
        Choice of mode, corresponding to the elements entered into the table.

    Returns
    -------
    None.
    """
    b = ( 1.1428 - 0.996 / v ) ** 2
    n_eff_approx = n2 + b * ( n1 - n2 )
    n_eff = table_modes.loc[table_modes['mode'] == mode_input]['n_eff'].values[0]
    n_eff_percent_diff = (n_eff_approx - n_eff) / n_eff * 100 

    ### fraction of power in the core
    ## approximate value
    omega_p_bar = 0.634 + 1.619 * v ** (-1.5) + 2.879 * v ** (-6) - 1.561 * v ** (-7)
    gamma_approx = 2 / (v * omega_p_bar) ** 2 + b
    gamma = table_modes.loc[table_modes['mode'] == mode_input]['power_fraction'].values[0]
    gamma_percent_diff = (gamma_approx - gamma) / gamma * 100 

    q8_table.loc[(len(q8_table))] = [mode_input, n_eff, n_eff_approx, n_eff_percent_diff, gamma, gamma_approx, gamma_percent_diff]
    pass



def run_q8():
    """
    Method called to run the code for Q8

    Returns
    -------
    None.
    """
    # creating the table 
    for mode in table_modes['mode'].values[4:]:
        q8_values(mode)
        pass
    print(q8_table.to_string())
    pass



#%%
'''
Defining code to run the entire project.
'''
def run_all():
    print('######################################################################')
    print('Optical Communications Project 2023-2024')
    print('Group   : 3')
    print('Members : 01849865, 01878398, 02150264')
    print('######################################################################')
    print('QUESTION 1: Calculate the V-parameter of the waveguide.')
    calculate_V_parameter(n1, n2, wavelength, a)
    print('----------------------------------------------------------------------')
    print('QUESTION 2: We have decided to consider the {TE, TM, HE, EH} modes.')
    print('----------------------------------------------------------------------')
    print('QUESTION 3: Identify all supported modes.')
    construct_table()
    print('----------------------------------------------------------------------')
    print('QUESTION 4: We have decided to consider the HE_22 mode.')
    print('----------------------------------------------------------------------')
    print('QUESTION 5: Plot amplitude of electric field.')
    plot_HE_22()
    print('Figure plotted (Figure 1)')
    print('----------------------------------------------------------------------')
    print('QUESTION 6: Spatial distribution of intensity.')
    plot_intensity('HE_22')
    print('Figure plotted (Figure 2)')
    print('----------------------------------------------------------------------')
    print('QUESTION 7: Calculate the waveguide dispersion.')
    run_dispersion(1250E-9)
    print('Figure plotted (Fugure 3 & 4)')
    print('----------------------------------------------------------------------')
    print('QUESTION 8: Calculate the n_eff and % of power carried in core.')
    plot_intensities_all(plot = False)
    run_q8()
    print('######################################################################')







