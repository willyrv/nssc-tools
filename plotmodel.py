import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import bisect
from scipy.optimize import minimize
#from lib.NI_model import StSICMR


##############  FUNCTIONS USED TO CALIBRATE THE IICR #############

def distance_PSMC_IICR(psmc_history, IICR, theta=1):
    """
    Compute the difference between the IICR inferred by PSMC
    and a given IICR
    psmc_history: a tuple of the form (time, lambda_t)
    IICR: a tuple of the form (time, IICR_t)
    theta: The scaled mutation rate per locus. Note that theta = Ne * 4\mu. 
    This value will be used for scaling the IICR
    """
    # Rescaling IICR by theta
    IICR_times = [i * 2 * theta for i in IICR[0]]
    IICR_values = [i * theta for i in IICR[1]]
    x_vector = list(set(psmc_history[0] + IICR_times))  # Remove duplicates
    x_vector.sort()
    difference = 0
    for i in range(1, len(x_vector)):
        psmcIx = bisect.bisect_right(psmc_history[0], x_vector[i]) - 1
        psmc_value = psmc_history[1][psmcIx]
        IICRIx = bisect.bisect_right(IICR_times, x_vector[i]) - 1
        IICR_value = IICR_values[IICRIx]
        difference += abs(psmc_value - IICR_value) * \
            (x_vector[i] - x_vector[i-1])
    return(difference)


def doIICRplots(psmc_history, IICR, theta=1):
    """
    Plots the psmc results and the IICR. Returns an ax object
    psmc_history: a tuple of the form (time, lambda_t) directly from
    the psmc file
    IICR: a tuple of the form (time, IICR_t)
    theta: The scaled mutation rate per locus. Note that theta = Ne * 4\mu. 
    This value will be used for scaling the IICR
    Returns: distance, the distance value
    """
    IICR_times = [i * theta for i in IICR[0]]
    IICR_values = [i * theta for i in IICR[1]]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(psmc_history[0], psmc_history[1], label='psmc')
    ax.step(IICR_times, IICR_values, label='IICR')
    ax.set_xscale('log')
    plt.legend(loc='best')
    plt.show()


def calibrateTheta(psmc_history, IICR):
    """
    Search for the best theta value that scale the IICR so
    it overlaps as best as possible with the given psmc_history
    psmc_history: a tuple of the form (time, lambda_t) directly from
    the psmc file
    IICR: a tuple of the form (time, IICR_t)
    Returns: theta, a float which is the best theta found
    """
    def distanceIICR(x): return distance_PSMC_IICR(psmc_history, IICR, x[0])
    # We try three different ways and get the best of them
    fValues_x = []
    bounded_SLSQP = minimize(distanceIICR, np.array([1]),
                             method='SLSQP', bounds=[(0, None)])
    fValues_x.append((bounded_SLSQP['fun'], bounded_SLSQP['x']))
    SLSQP = minimize(distanceIICR, np.array([1]), method='SLSQP')
    fValues_x.append((SLSQP['fun'], SLSQP['x']))
    NeldMead = minimize(distanceIICR, np.array([1]), method='Nelder-Mead')
    fValues_x.append((NeldMead['fun'], NeldMead['x']))
    fValues_x.sort()  # Will sort the list by the first element of the tuples
    return(fValues_x[0][1])


def distanceForM(psmc_history, StCoalModel, M):
    """
    Compute the distance from a psmc_history and a StCoalModel for
    a given M
    """
    StCoalModel.update(StCoalModel.n, StCoalModel.T, [M], StCoalModel.C)
    IICR = [psmc_history[0], [StCoalModel.lambda_s(i)
                              for i in psmc_history[0]]]
    return(distance_PSMC_IICR(psmc_history, IICR))

# def fitNislands(psmc_history, minMaxINb=(2, 50), minMaxMig=(0.001, 100),
#                 demeSize = 1):
#     """
#     Fits an n-islands model to a given psmc history
#     """
#     times = psmc_history[0]
#     n_min = minMaxINb[0]
#     n_max = minMaxINb[1]
#     StCoalModel = StSICMR(n_min, [0], [1], [demeSize])
#     distance_IICR = lambda x: distanceForM(psmc_history, StCoalModel, x[0])
#     fValuesTuples = []
#     for i in range(n_min, n_max):
#         bounded_SLSQP = minimize(distance_IICR, np.array([1]),
#                                  method='SLSQP', bounds=[minMaxMig])
#         fValuesTuples.append((bounded_SLSQP['fun'], bounded_SLSQP['x'], i))
#         SLSQP = minimize(distance_IICR, np.array([1]), method='SLSQP')
#         fValuesTuples.append((SLSQP['fun'], SLSQP['x'], i))
#         NeldMead = minimize(distance_IICR, np.array([1]),
#                             method='Nelder-Mead')
#         fValuesTuples.append((NeldMead['fun'], NeldMead['x'], i))
#     fValuesTuples.sort()
#     return(fValuesTuples[0])

########################


# Graph the PSMC versus the Model
def graph(model, times, lambdas):
    lambda_s = [model.lambda_s(t) for t in times]
    # Display the model
    plt.plot(times, lambda_s, label='Model', linewidth=6, color='red')
    # Display the target PSMC
    plt.step(times, lambdas, label='PSMC',
             linewidth=6, where='post', color='green')
    # Add the mutation times as vertical lines
    for t in model.T_list[1:]:
        plt.axvline(t, linestyle='--', color='k', alpha=0.5)
    # Set x scale to logarithmic time
    plt.xscale('log')
    # Annotate the graph
    plt.title('Infinitesimal Generator VS PSMC',
              fontsize=14, fontweight='bold')
    plt.legend(loc=4)
    plt.xlabel('Time going backwards (logscale)')
    plt.ylabel('Population size')
    # Add best model information
    information = 'Number of islands: ' + str(model.n) + '\n' + \
                  'Times of flow rate changes: ' + str(model.T_list) + \
                  '\n' + 'Flow rates: ' + str(model.M_list)
    plt.annotate(information, xy=(0, 1), xycoords='axes fraction',
                 fontsize=12, ha='left', va='top', xytext=(5, -5),
                 textcoords='offset points')
    plt.show()


def get_psmc_history(filename):
    # Returns the history infered by psmc in a tuple (theta, t_k, lambda_k)
    a = open(filename, 'r')
    text = a.read()
    a.close()

    # getting the time windows and the lambda values
    last_block = text.split('//\n')[-2]
    last_block = last_block.split('\n')
    time_windows = []
    estimated_lambdas = []
    for line in last_block:
        if line[:2] == 'RS':
            time_windows.append(float(line.split('\t')[2]))
            estimated_lambdas.append(float(line.split('\t')[3]))

    # getting the estimations of theta and N0
    # The 'PA' lines contain the estimated lambda values
    result = text.split('PA\t')
    result = result[-1].split('\n')[0]
    result = result.split(' ')
    theta = float(result[1])
    # N_ref = theta/(4*mut_rate)/bin_size

    return (theta, time_windows, estimated_lambdas)


def plot_results_from_folder(psmc_results_folder, mutation_rate, bin_size,
                             generation_time):
    # Plots all the .psmc files in a folder

    # Get the data from the psmc files
    psmc_files = os.path.join(psmc_results_folder, '*.psmc')
    files = glob.glob(psmc_files)
    infered_histories = [get_psmc_history(f, mutation_rate, bin_size)
                         for f in files]

    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for h in infered_histories:
        N_ref = h[0]
        x = 2 * N_ref * generation_time * np.array(h[1])
        y = N_ref * np.array(h[2])
        ax.step(x, y, '-g', where='post')
    ax.set_xlim(1e3, 1e7)
    ax.set_ylim(0, 5e4)
    ax.set_xscale('log')
    plt.show()


def plot_real_fitted_histories(psmc_results_folder, mutation_rate, bin_size,
                               generation_time, real_effective_size=False,
                               instants2mark=False, extra_info=False):
    """
    Plots all the .psmc files in a folder (i.e. the results of psmc).
    Plot also the psmc history recovered from the ms command in the 
    supplementary materials.
    The argument instants2mark is a list containing values (in years) that 
    indicate a place to put a vertical line (for example, migration rate 
    changes).
    The argument extra_info is for marking the periods when some known process
    have taken place. The format of the information is a dictionary where keys
    are tuples in the form (start, end) in years. The keys of the dicct are
    the names of the period.

    The history used comes from the ms-command was the sim-YH from the 
    Supplementary Materials.

    ////////////////////////////////////////////////////////////////////////
    The ms-command is:    

    ms 2 100 -t 65130.39 -r 10973.82 30000000 -eN 0.0055 0.0832 \
    -eN 0.0089 0.0489 -eN 0.0130 0.0607 -eN 0.0177 0.1072 -eN 0.0233 0.2093 \
    -eN 0.0299 0.3630 -eN 0.0375 0.5041 -eN 0.0465 0.5870 -eN 0.0571 0.6343 \
    -eN 0.0695 0.6138 -eN 0.0840 0.5292 -eN 0.1010 0.4409 -eN 0.1210 0.3749 \
    -eN 0.1444 0.3313 -eN 0.1718 0.3066 -eN 0.2040 0.2952 -eN 0.2418 0.2915 \
    -eN 0.2860 0.2950 -eN 0.3379 0.3103 -eN 0.3988 0.3458 -eN 0.4701 0.4109 \
    -eN 0.5538 0.5048 -eN 0.6520 0.5996 -eN 0.7671 0.6440 -eN 0.9020 0.6178 \
    -eN 1.0603 0.5345 -eN 1.4635 1.7931 

    ////////////////////////////////////////////////////////////////////////

    we used \mu=2.5e-8
    """
    N0 = 65130.39/30000000/(4*2.5e-8)
    liDurbin_tk = np.array([1e-5, 0.011, 0.0178, 0.026, 0.0354, 0.0466, 0.0598,
                            0.075, 0.093, 0.1142, 0.139, 0.168, 0.202, 0.242,
                            0.2888, 0.3436, 0.408, 0.4836, 0.572, 0.6758,
                            0.7976, 0.9402, 1.1076, 1.304, 1.5342, 1.804,
                            2.1206, 2.927, 3.5])

    liDurbin_lk = np.array([1, 0.0832, 0.0489, 0.0607, 0.1072, 0.2093, 0.363,
                            0.5041, 0.587, 0.6343, 0.6138, 0.5292, 0.4409,
                            0.3749, 0.3313, 0.3066, 0.2952, 0.2915, 0.295,
                            0.3103, 0.3458, 0.4109, 0.5048, 0.5996, 0.644,
                            0.6178, 0.5345, 1.7931, 1.7931])

    li_durbin_x = 2 * N0 * 25 * liDurbin_tk
    li_durbin_y = N0 * liDurbin_lk
    # Get the data from the psmc files
    psmc_files = os.path.join(psmc_results_folder, '*.psmc')
    files = glob.glob(psmc_files)
    infered_histories = [get_psmc_history(f, mutation_rate, bin_size)
                         for f in files]

    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    h = infered_histories[0]
    N_ref = h[0]
    x = 2 * N_ref * generation_time * np.array(h[1])
    y = N_ref * np.array(h[2])

    label = 'PSMC estim. 10 sim. scenarios'
    p1 = ax.step(x, y, '-g', where='post', label=label)
    for h in infered_histories[1:]:
        N_ref = h[0]
        x = 2 * N_ref * generation_time * np.array(h[1])
        y = N_ref * np.array(h[2])
        ax.step(x, y, '-g', where='post')

    label = 'PSMC estim. real data'
    p2 = ax.step(li_durbin_x, li_durbin_y, '-r', where='post', label=label)

    if real_effective_size:
        p3 = ax.plot([0.01, 1e7], [real_effective_size, real_effective_size],
                     ':k', label='Real pop. size sim. scenarios')

    # Mark some important epochs
    if instants2mark:
        n = 0
        for i in instants2mark:
            ax.axvline(x=i, color='k', linestyle='--', linewidth=0.5)
            n += 1
            ax.annotate('$M_{}$'.format(n), xy=(i, 2e4), xycoords='data',
                        xytext=(-7, 50), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"))

    x_events = []
    x_events_labels = []
    if extra_info:
        for k in extra_info.keys():
            ax.fill_between(k, [0, 0], [5e4, 5e4], color='blue',
                            facecolor='blue', alpha=0.5)
            # ax.axvline(x=k[0], color='r', alpha=1, linestyle='--',
            #           linewidth=0.3)
            # ax.axvline(x=k[1], color='r', alpha=1, linestyle='--',
            #           linewidth=0.3)
            #ax.annotate(extra_info[k], xy=(k[0], 3.1e4))
            x_events.append(k[0])
            x_events_labels.append(extra_info[k])

    ax.set_xlim(1e3, 1e7)
    ax.set_ylim(0, 3.5e4)
    ax.set_xscale('log')

    # The ticks and the labels I want to put in the plot
    ticks = [1e3, 1e4, 1e5, 1e6, 1e7]
    tlabels = [r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$']

    # Set the ticks and add the new ones
    ax.set_xticks(ticks+x_events)
    ax.set_xticklabels(tlabels+x_events_labels)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # Get the ticklabels we will rotate
    labels2modify = [i.label for i in ax.xaxis.get_major_ticks()[-3:]]
    # Rotate
    [i.set_rotation(30) for i in labels2modify]
    [i.set_position((1, -0.05)) for i in labels2modify]

    ax.set_xlabel('time (in years)')
    ax.set_ylabel('IICR (x$10^{4}$)')
    plt.legend(loc='upper left', prop={'size': 8})

    plt.tight_layout()
    plt.savefig('./figure.png', dpi=300)
    plt.show()
