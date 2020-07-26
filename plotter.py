import numpy as np
import matplotlib.pyplot as plt
import sys

#data_stream = 'rbf_fast'
data_stream_ = 1
streams = ['agrawal', 'hyperplane', 'led', 'rbf_slow', 'rbf_fast', 'sea']
#data_stream = streams[data_stream_]
print('\n\t----------------------------')
print('\tData Stream: === ' + streams[int(sys.argv[1])].upper() + ' === ')
print('\t----------------------------')
data_stream = streams[int(sys.argv[1])]

accuracies_1 = np.load('results/' + data_stream + '_accuracies_1.npy') * 100.0
accuracies_2 = np.load('results/' + data_stream + '_accuracies_2.npy') * 100.0
accuracies_3_goowe = np.load('results/' + data_stream + '_goowe_accuracies_3.npy') * 100.0
accuracies_3_mv = np.load('results/' + data_stream + '_mv_accuracies_3.npy') * 100.0
accuracies_3_av = np.load('results/' + data_stream + '_av_accuracies_3.npy') * 100.0

print('\tAccuracy values:\n\tSource 1: {}\n\tSource 2: {}\n\tGoowe: {}\n\tMV: {}\n\tAV: {}\n'
      .format(np.mean(accuracies_1), np.mean(accuracies_2), np.mean(accuracies_3_goowe),
              np.mean(accuracies_3_mv), np.mean(accuracies_3_av)))

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(20, 12))
#plt.ylim(ymin=0.6, ymax=0.95)
plt.plot(accuracies_1, color = 'r', linestyle = 'dashed', label='Source Stream 1')
plt.plot(accuracies_2, color='orange', linestyle = 'dashed', label='Source Stream 2')
plt.plot(accuracies_3_goowe, color = 'c', label='Target Stream (Goowe)')
plt.plot(accuracies_3_mv, color = 'b', label='Target Stream (MV)')
plt.plot(accuracies_3_av, color = 'k', label='Target Stream (AV)')
plt.grid(True)
plt.legend()
fig.savefig('figs/' + data_stream + '.png')
plt.show()
