#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:49:50 2025

@author: augustin
"""


from openwind import InstrumentPhysics, FrequentialSolver, InstrumentGeometry, Player
from openwind.inversion import InverseFrequentialResponse
import matplotlib.pyplot as plt
import numpy as np

from openwind.impedance_tools import plot_reflection

plt.close('all')

plot_admittance = True
# %% Geometry and fingering chart

bore = [[0,75, 10,10,'linear'],
        [75,'400', 10, '6','linear']] # a conical pipe
holes = [['label','location','diameter','chimney'],
         ['embouchure',20,8,'1<~5'], # the ~ allow to modify the parameters without recomputing everything and to optimize it
         ['hole1',100,7,3],
         ['hole2',200,5,3],
         ['hole3',250,5,3],
         ['hole4',300,4,3],
         ['hole5',350,4,3],
         ]
chart = [['label', 'A', 'B', 'C', 'D'],
         ['hole1','x','x','x','x'],
         ['hole2','x','x','x','o'],
         ['hole3','x','x','o','o'],
         ['hole4','x','o','o','o'],
         ['hole5','x','o','o','o'],
         ]

my_geom = InstrumentGeometry(bore, holes, chart, unit='mm', diameter=True)
my_geom.plot_InstrumentGeometry()

param = my_geom.optim_params
print(param) # you print the data about the design parameter
# now you can easily modify the geometry by doing:
param.set_active_values([5e-3]) # the length must be given in meter

# %% expected sounding frequency
""" Here you can define any sounding frequency """

notes = my_geom.fingering_chart.all_notes()
diapason = 415
semitone = np.array([0,2,3,5])#,12])
f_sound = diapason*2**(semitone/12)

f_harmonics = np.array([f_sound, 2*f_sound, 3*f_sound])

# %% Simulation
"""We compute the impedance for the original geometry"""
freq_simu = np.arange(100,5002,2)
my_phy = InstrumentPhysics(my_geom, temperature=25, player=Player(),
                           source_location='embouchure', # set the source point at the embouchure hole
                           losses=True, humidity=.5,
                           radiation_category={'entrance':'closed', 'holes':'unflanged', 'bell':'unflanged'}
                           )

# Añade esta verificación:
if hasattr(my_phy.player, 'labels'): # Accede al player dentro de my_phy
    print(f"INFO en adjust_embouchure_tube_length.py: my_phy.player tiene 'labels': {my_phy.player.labels}")
else:
    print(f"ERROR CRÍTICO en adjust_embouchure_tube_length.py: my_phy.player NO TIENE 'labels'.")

# This is necessary if we want to optimize the length correction in order to reach perfectly the sounding frequency
Z_target = np.array([0]) # at the resonance freq, we want an impedance with a 0 phase (or imaginary part =0)

# choice between 2 observables:
obs = ['impedance_phase', # better define as we don't need to guess the magnitude of the peak, but can accidentaly adjust an anti-resonance
       'reflection'] # force to adjust an admittance peak resonance, but need to guess the peak magnitude (here infinite)
optim = InverseFrequentialResponse(my_phy, f_sound[0], [Z_target], observable=obs[1])

length_correction = list() # initiation of the length correction list

for k_note, note in enumerate(notes): # loop through the fingerings
    optim.update_frequencies_and_mesh(freq_simu) # set a frequency vector with a fine step to correctly estimate the resonance frequency
    optim.set_note(note) # set the right fingering
    optim.solve() # compute the impedance
    f_res_simu = optim.antiresonance_frequencies(k=1, display_warning=False)[0] # estimate the resonances

    if plot_admittance:
        fig = plt.figure()
        optim.plot_admittance(figure=fig, label='init')
        plt.axvline(f_sound[k_note], color='k', label='target')
        print(f_res_simu)
        plt.title(note)



    # %% Low freq approximation
    """Here we estimate the length correction from a low freq approximation"""

    celerity = my_phy.get_entry_coefs('c')[0]
    delta_L = celerity/2*(1/f_sound[k_note] - 1/f_res_simu)

    current_height = param.get_active_values()[0]
    new_height = (current_height + delta_L)*(8/10)**2 #the new desired height


    if new_height<=0:
        raise ValueError('The new height is negative, this should be wrong...')
    optim.modify_parts([new_height]) # we modify the geometry and update the mesh etc...


    if plot_admittance:
        optim.set_note(note) # we re-set the fingering (I have to correct this issue: currently modifying the geometry reset the fingering to all open)
        optim.solve() # we recompute the impedance
        optim.plot_admittance(figure=fig, label='low freq approx')
        f_res_simu = optim.antiresonance_frequencies(k=1, display_warning=False)[0]



    # %% optimmize
    """ Here we optimize the chimney length to have perfectly the right frequency of resonance"""
    optim.update_frequencies_and_mesh(f_sound[k_note]) # to do so we only need to compute the impedance at the desired freq of resonance
    optim.set_targets_list([Z_target], [note]) # and we want that, at this, freq, the phase of impedance is 0

    optim.optimize_freq_model(iter_detailed=True) # we perform the optimization

    if plot_admittance:
        optim.recompute_impedance_at(freq_simu)
        optim.plot_admittance(figure=fig, label='optimized')
        plt.legend()
        f_res_optim = optim.antiresonance_frequencies(k=2, display_warning=False)

        print(f'Final deviation: {f_res_optim[0]} vs {f_sound[k_note]}')
        print(f'Final harmonicity: {f_res_optim[1]/f_res_optim[0]}')


    length_correction.append(param.values[0])

    # %% display ac. fields
    """Here we display the ac fields"""
    optim.update_frequencies_and_mesh(f_harmonics[:,k_note]) # for each fingering we can compute only at the frequency of interest, here f1, f2, f3
    optim.solve(interp=True,interp_grid=1e-3) # we solve by interpolating the ac fields all mm.

    # We get the location, pressure and flow at all the computed frequencies
    # these fields are computed for an 1m³/s input flow... If you want raisonable values,
    # you can scale by sqrt(Zc)
    x = optim.x_interp
    p = optim.pressure/np.sqrt(optim.get_ZC_adim())
    u = optim.flow*np.sqrt(optim.get_ZC_adim())

    fig, axs = plt.subplots(2,1,sharex=True)
    fig.suptitle(f"Fingering: {note}")
    axs[0].plot(x*1e3, np.real(p).T)
    axs[0].grid()
    axs[0].set_ylabel('Pressure [Pa]')
    # axs[0].set_xlabel('Location on the main bore [mm]')
    axs[0].legend(['$f_1$', '$2*f_1$','$3*f_1$'])


    axs[1].plot(x*1e3, np.real(u).T)
    axs[1].grid()
    axs[1].set_ylabel('Flow [L/s]')
    axs[1].set_xlabel('Location on the main bore [mm]')
    # axs[1].legend(['$f_1$', '$2*f_1$','$3*f_1$'])
    plt.show()



# %% Length correction obtained for each fingerings

plt.figure()
plt.bar(range(len(notes)), np.array(length_correction)*1000)
plt.ylabel('Length correction [mm]')
plt.xticks(range(len(notes)), notes)
plt.xlabel('Notes')
