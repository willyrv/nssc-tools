import copy
import numpy as np
import matplotlib.pyplot as plt
import bisect
from plotmodel import get_psmc_history, calibrateTheta
from functions import readScenario
from model import NSSC

def generateScaledIICR(msCmd, x_vector, scaling_factor=1, thetaEstim=1, scalingNe=1e3):
    T2values = generate_MS_t2(msCmd)
    T2values = [2 * i for i in T2values] # Rescaling time values comming from ms
    (T2cdf, T2pdf) = compute_empirical_dist(T2values, x_vector)
    empirical_lambda = np.true_divide(len(T2values) - T2cdf, T2pdf)
    # Rescaling
    s = scaling_factor * thetaEstim
    rescaledIICRTime = [i * s for i in x_vector]
    s = scaling_factor * thetaEstim * scalingNe
    rescaledIICRLambda = [i * s for i in empirical_lambda]
    return(rescaledIICRTime, rescaledIICRLambda)

human_scenario_file = "./scenarios/HumansNeandScenario_samplingHumans.txt"
neand_scenario_file = "./scenarios/HumansNeandScenario_samplingNeanderthals.txt"

# For real data
# Reading psmc results
psmc_theta_hum, psmc_time_hum, psmc_lambda_hum = get_psmc_history('./psmc_results/real_data/human/French_upper.psmc')
psmc_theta_neand, psmc_time_neand, psmc_lambda_neand = get_psmc_history('./psmc_results/real_data/Altai/AltaiNea.psmc')
# Compute the theta per locus
s = 100
psmc_theta_hum = psmc_theta_hum / s
psmc_theta_neand = psmc_theta_neand / s

# Scaling psmc for humans
scalled_psmc_time_hum = [psmc_theta_hum * i for i in psmc_time_hum]
scalled_psmc_lambda_hum = [psmc_theta_hum * i * 1e3 for i in psmc_lambda_hum]

# Shifting the Neanderthal IICR. 
# Neanderthal time is shifted by 50 kya 
TimeY = 5.3e4
muY = 2*0.5e-9
time_shift = TimeY * 2*muY / psmc_theta_neand  # Converting to coalescent scale
shifted_psmc_time_neand = [i + time_shift for i in psmc_time_neand]
# Scaling psmc for neanderthals
scaled_shifted_psmc_time_neand = [psmc_theta_neand * i for i in shifted_psmc_time_neand]
scaled_psmc_lambda_neand = [psmc_theta_neand * i * 1e3 for i in psmc_lambda_neand]


start = 0
end = 100
n = 500
x_vector = [0.1*(np.exp(i * np.log(1+10*end)/n)-1) for i in range(n+1)]
#x_vector = np.arange(0, 100, 0.1)

# rescaling by 0.1
scaling_factor = 0.1

#######  Creating the NSSC models ################
# For humans
d_humans = readScenario(human_scenario_file)
current_d_humans = copy.deepcopy(d_humans)
model_humans = NSSC(d_humans)
IICR_humans = [model_humans.evaluateIICR(i) for i in x_vector]
s = scaling_factor * psmc_theta_hum
rescaled_IICR_time_hum = [i * s for i in x_vector]
rescaled_IICR_lambda_hum = [i * s * 1e3 for i in IICR_humans]
# For neanderthals
d_neanderthals = readScenario(neand_scenario_file)
current_d_neanderthals = copy.deepcopy(d_neanderthals)
model_neanderthals = NSSC(d_neanderthals)
IICR_neanderthals = [model_neanderthals.evaluateIICR(i) for i in x_vector]
rescaled_IICR_time_neand = [i * s for i in x_vector]
rescaled_IICR_lambda_neand = [i * s * 1e3 for i in IICR_neanderthals]

# Doing the plot
plt.ion()
fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax1.step(scalled_psmc_time_hum, scalled_psmc_lambda_hum, 'b', label='humans real data')
ax1.step(scaled_shifted_psmc_time_neand, scaled_psmc_lambda_neand, 'r', label='neanderthal real data')
pHumans = ax1.step(rescaled_IICR_time_hum, rescaled_IICR_lambda_hum, 'g', label='humans simulated')
pNeand = ax1.step(rescaled_IICR_time_neand, rescaled_IICR_lambda_neand, 'y', label='neanderthal simulated')
ax1.set_xlabel(r'time (in $2\mu T$ units)')
ax1.set_ylabel(r'IICR (in units of $4\mu N_{e} \times 10^{3}$)')
#ax1.legend(loc='upper right', bbox_to_anchor=(.5, .8))
ax1.legend(bbox_to_anchor=(1.4, 1), fontsize=11)
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_xlim(1e-5, 1e-2)
ax1.set_ylim(0, 3.5)
ax2.set_xlim(1e-5, 1e-2)
ax2.set_xticklabels(["", "5-10kya", "50-100kya", "0.5-1Mya", "5-10Mya"])
#plt.legend(loc='upper left')
plt.pause(2)



while True:
    #############     Reading the scenarios and simulating the data     #############
    # For humans
    d_humans = readScenario(human_scenario_file)
    if current_d_humans != d_humans:
        current_d_humans = copy.deepcopy(d_humans)
        model_humans = NSSC(d_humans)
        IICR_humans = [model_humans.evaluateIICR(i) for i in x_vector]
        s = scaling_factor * psmc_theta_hum
        rescaled_IICR_time_hum = [i * s for i in x_vector]
        rescaled_IICR_lambda_hum = [i * s * 1e3 for i in IICR_humans]
        for p in pHumans:
            p.remove()
        pHumans = ax1.step(rescaled_IICR_time_hum, rescaled_IICR_lambda_hum, 'g', label='humans simulated')

    # For neanderthals
    d_neanderthals = readScenario(neand_scenario_file)
    if current_d_neanderthals != d_neanderthals:
        current_d_neanderthals = copy.deepcopy(d_neanderthals)
        model_neanderthals = NSSC(d_neanderthals)
        IICR_neanderthals = [model_neanderthals.evaluateIICR(i) for i in x_vector]
        rescaled_IICR_time_neand = [i * s for i in x_vector]
        rescaled_IICR_lambda_neand = [i * s * 1e3 for i in IICR_neanderthals]
        for p in pNeand:
            p.remove()
        pNeand = ax1.step(rescaled_IICR_time_neand, rescaled_IICR_lambda_neand, 'y', label='neanderthal simulated')
    plt.pause(2)

