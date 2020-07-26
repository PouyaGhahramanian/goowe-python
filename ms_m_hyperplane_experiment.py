from skmultiflow.data.file_stream import FileStream
import numpy as np
from Goowe import Goowe
#from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import HyperplaneGenerator
import logging
from GooweMSS import GooweMS
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Prepare the data stream
streams = []
N_STREAMS = 10
instances_num = 10000

for i in range(N_STREAMS):
    stream = HyperplaneGenerator(random_state=None, n_features=10, n_drift_features=2,
                                   mag_change=0.1, noise_percentage=0.0, sigma_percentage=0.1)
    streams.append(stream)
    stream.prepare_for_use()

stream_t = HyperplaneGenerator(random_state=None, n_features=10, n_drift_features=2,
                               mag_change=0.1, noise_percentage=0.0, sigma_percentage=0.1)
stream_t = streams[0]
stream_t.prepare_for_use()

instances_counter = 0

### Arrays for storing accuracy values for Streams
accuracies_all = []
for i in range(N_STREAMS):
    accuracies = []
    accuracies_all.append(accuracies)
accuracies_mv = []
accuracies_av = []
accuracies_goowe = []
accuracies_all.append(accuracies_av)
accuracies_all.append(accuracies_mv)
accuracies_all.append(accuracies_goowe)

num_features = stream_t.n_features
num_targets = stream_t.n_targets
num_classes = 2
target_values = [0., 1.]
logging.info("\n\tStreams are generated and prepared for use.\n\tNumber of features: {0} - Number of targets: {1} - Number of classes: {2} - Target values: {3}"
             .format(num_features, num_targets, num_classes, target_values))

N_MAX_CLASSIFIERS = 15
N_MAX_CLASSIFIERS_TARGET = 30
CHUNK_SIZE = 500        # User-specified
WINDOW_SIZE = 100       # User-specified

### Probability of drift in streams
p_tresholds = []
for i in range(N_STREAMS):
    p_treshold = 0.8
    p_tresholds.append(p_treshold)
pt_threshold = 0.8

# Initialize the ensemble
goowes = []
for i in range(N_STREAMS):
    goowe = Goowe(n_max_components=N_MAX_CLASSIFIERS,
                  chunk_size=CHUNK_SIZE,
                  window_size=WINDOW_SIZE,
                  logging = False)
    goowes.append(goowe)
    # Initialize the ensemble
    goowe.prepare_post_analysis_req(num_features, num_targets, num_classes, target_values)

goowe_t = GooweMS(goowes, num_streams = 10, n_max_components=N_MAX_CLASSIFIERS_TARGET,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE,
              logging = False)
goowe_t.prepare_post_analysis_req(num_features, num_targets, num_classes, target_values)

# For the first chunk, there is no prediction.
for i in range(N_STREAMS):
    X_init, y_init = streams[i].next_sample(CHUNK_SIZE)
    goowes[i].partial_fit(X_init, y_init)

X_init, y_init = stream_t.next_sample(CHUNK_SIZE)
goowe_t.update(X_init, y_init, [])

accuracies_tmp = np.zeros(N_STREAMS)
accuracy_mv = 0.0
accuracy_av = 0.0
accuracy_goowe = 0.0
totals = np.zeros(N_STREAMS)
true_predictions = np.zeros(N_STREAMS)
true_predictions_t = 0.0
true_predictions_t_mv = 0.0
true_predictions_t_av = 0.0
true_predictions_t_goowe = 0.0
total = 0.

for i in range(CHUNK_SIZE):
    total += 1
    Xs = []
    ys = []
    for j in range(N_STREAMS):
        curr = streams[j].next_sample()
        X, y = curr[0], curr[1]
        Xs.append(X)
        ys.append(y)
        preds = goowes[j].predict(X)
        true_predictions[j] += np.sum(preds == y)
        accuracies_tmp[j] = true_predictions[j] / total

    curr_t = stream_t.next_sample()
    X_t, y_t = curr_t[0], curr_t[1]
    preds_t_mv = goowe_t.predict(X_t, ensemble_type='mv')
    preds_t_av = goowe_t.predict(X_t, ensemble_type='av')
    preds_t_goowe = goowe_t.predict(X_t, ensemble_type='goowe')
    true_predictions_t_mv += np.sum(preds_t_mv == y_t)
    true_predictions_t_av += np.sum(preds_t_av == y_t)
    true_predictions_t_goowe += np.sum(preds_t_goowe == y_t)
    accuracy_mv = true_predictions_t_mv / total
    accuracy_av = true_predictions_t_av / total
    accuracy_goowe = true_predictions_t_goowe / total
    for j in range(N_STREAMS):
        print('\tSTREAM {} :: Data instance: {} - Accuracy: {}'.format(str(j+1), int(total), round(accuracies_tmp[j]*100.0, 3)))
    print('\tTARGET STREAM :: Data instance: {} - Accuracies: MV: {} - AV: {} - Goowe: {}'.format(int(total),
        round(accuracy_mv*100.0, 3), round(accuracy_av*100.0, 3), round(accuracy_goowe*100.0, 3)))
    print('\t==========================================================================')
    for j in range(N_STREAMS):
        goowes[j].partial_fit(Xs[j], ys[j])
    goowe_t.update(X_t, y_t,[])

# Now, for the remaining instances, do ITTT (Interleaved Test Then Train).
while(stream_t.has_more_samples() and instances_counter < instances_num):

    if(instances_counter % CHUNK_SIZE == 0):
        accuracies_tmp = np.zeros(N_STREAMS)
        accuracy_mv = 0.0
        accuracy_av = 0.0
        accuracy_goowe = 0.0
        totals = np.zeros(N_STREAMS)
        true_predictions = np.zeros(N_STREAMS)
        true_predictions_t = 0.0
        true_predictions_t_mv = 0.0
        true_predictions_t_av = 0.0
        true_predictions_t_goowe = 0.0
        total = 0.

    ### Generating drifts by generating random values for each Stream
    ps = []
    for j in range(N_STREAMS):
        p = random.random()
        ps.append(p)
        #if p > p_tresholds[j]:
            #streams[j].generate_drift()
            #logging.info('\n\tDrift generatoed for STREAM {}'.format(str(j+1)))
    p_t = np.random.random()
    #if p_t > pt_threshold:
        #stream_t.generate_drift()
        #logging.info('\n\tDrift generatoed for TARGET STREAM')
    total += 1
    Xs = []
    ys = []
    for j in range(N_STREAMS):
        curr = streams[j].next_sample()
        X, y = curr[0], curr[1]
        Xs.append(X)
        ys.append(y)
        preds = goowes[j].predict(X)
        true_predictions[j] += np.sum(preds == y)
        accuracies_tmp[j] = true_predictions[j] / total

    curr_t = stream_t.next_sample()
    X_t, y_t = curr_t[0], curr_t[1]
    preds_t_mv = goowe_t.predict(X_t, ensemble_type='mv')
    preds_t_av = goowe_t.predict(X_t, ensemble_type='av')
    preds_t_goowe = goowe_t.predict(X_t, ensemble_type='goowe')
    true_predictions_t_mv += np.sum(preds_t_mv == y_t)
    true_predictions_t_av += np.sum(preds_t_av == y_t)
    true_predictions_t_goowe += np.sum(preds_t_goowe == y_t)
    accuracy_mv = true_predictions_t_mv / total
    accuracy_av = true_predictions_t_av / total
    accuracy_goowe = true_predictions_t_goowe / total
    for j in range(N_STREAMS):
        print('\tSTREAM {} :: Data instance: {} - Accuracy: {}'.format(str(j+1), int(total), round(accuracies_tmp[j]*100.0, 3)))
    print('\tTARGET STREAM :: Data instance: {} - Accuracies: MV: {} - AV: {} - Goowe: {}'.format(int(total),
        round(accuracy_mv*100.0, 3), round(accuracy_av*100.0, 3), round(accuracy_goowe*100.0, 3)))
    print('\tCurrent classifier indices: {}'.format(goowe_t.get_classifer_indices()))
    print('\t==========================================================================')
    for j in range(N_STREAMS):
        goowes[j].partial_fit(Xs[j], ys[j])
    goowe_t.update(X_t, y_t, [])
    for j in range(N_STREAMS):
        accuracies_all[j].append(accuracies_tmp[j])
    accuracies_mv.append(accuracy_mv)
    accuracies_av.append(accuracy_av)
    accuracies_goowe.append(accuracy_goowe)
    #np.save('results/agrawal_'+ENSEMBLE_TYPE+'_accuracies_1.npy', np.asarray(accuracies_1))
    #np.save('results/agrawal_'+ENSEMBLE_TYPE+'_accuracies_2.npy', np.asarray(accuracies_2))
    #np.save('results/agrawal_'+ENSEMBLE_TYPE+'_accuracies_3.npy', np.asarray(accuracies_3))
    instances_counter += 1
np.save('results/multi_source/hyperplane_accuracies_sources.npy', np.asarray(accuracies_all))
np.save('results/multi_source/hyperplane_accuracies_mv.npy', np.asarray(accuracies_mv))
np.save('results/multi_source/hyperplane_accuracies_av.npy', np.asarray(accuracies_av))
np.save('results/multi_source/hyperplane_accuracies_goowe.npy', np.asarray(accuracies_goowe))
# TODO: Create new goowe_3 by using components of the other two goowes with highest weights (5 from each i.e.)
# TODO: AND update goowe_3 at each chunk (each while step)
