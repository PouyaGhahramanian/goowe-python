import numpy as np
import matplotlib.pyplot as plt
import sys
import random

#data_stream = 'rbf_fast'
N_STREAMS = 10
data_stream_ = 1
streams = ['agrawal', 'hyperplane', 'led', 'rbf_slow', 'rbf_fast', 'sea']
#data_stream = streams[data_stream_]
print('\n\t----------------------------')
print('\tData Stream: === ' + streams[int(sys.argv[1])].upper() + ' === ')
print('\t----------------------------')
data_stream = streams[int(sys.argv[1])]

accuracies_all = np.load('results/multi_source/' + data_stream + '_accuracies_sources.npy') * 100.0
accuracies_goowe = np.load('results/multi_source/' + data_stream + '_accuracies_goowe.npy') * 100.0
accuracies_mv = np.load('results/multi_source/' + data_stream + '_accuracies_mv.npy') * 100.0
accuracies_av = np.load('results/multi_source/' + data_stream + '_accuracies_av.npy') * 100.0

print('\tAccuracy values: ')
for j in range(N_STREAMS):
    print('\tSource {}: {}'.format(j+1, np.round(np.mean(accuracies_all[j]), 3)))
print('\tMV: {}\n\tAV: {}\n\tGoowe: {}\n'.format(np.round(np.mean(accuracies_mv), 3),
            np.round(np.mean(accuracies_av)), np.round(np.mean(accuracies_goowe))))

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(20, 12))
#plt.ylim(ymin=0.6, ymax=0.95)
for j in range(N_STREAMS):
    r = random.random()
    b = random.random()
    g = random.random()
    color_random = (r, g, b)
    plt.plot(accuracies_all[j], color = color_random, linestyle = 'dashed', label='Source Stream {}'.format(j+1))
plt.plot(accuracies_goowe, color = 'c', label='Target Stream (Goowe)')
plt.plot(accuracies_mv, color = 'b', label='Target Stream (MV)')
plt.plot(accuracies_av, color = 'k', label='Target Stream (AV)')
plt.grid(True)
plt.legend()
fig.savefig('figs/multi_source/' + data_stream + '.png')
plt.show()
