from skmultiflow.data.file_stream import FileStream
import numpy as np
from Goowe import Goowe
from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import AGRAWALGenerator
import logging
from GooweMS import GooweMS
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Prepare the data stream
stream_1 = ConceptDriftStream(stream=AGRAWALGenerator(balance_classes=False, classification_function=1, perturbation=0.0, random_state=112),
            drift_stream=AGRAWALGenerator(balance_classes=False, classification_function=2, perturbation=0.0, random_state=112),
            position=3000, width=1000, random_state=None, alpha=0.0)
stream_2 = ConceptDriftStream(stream=AGRAWALGenerator(balance_classes=False, classification_function=3, perturbation=0.0, random_state=21),
            drift_stream=AGRAWALGenerator(balance_classes=False, classification_function=1, perturbation=0.0, random_state=22),
            position=7000, width=200, random_state=None, alpha=0.0)
stream_3 = ConceptDriftStream(stream=AGRAWALGenerator(balance_classes=False, classification_function=1, perturbation=0.0, random_state=11),
            drift_stream=AGRAWALGenerator(balance_classes=False, classification_function=2, perturbation=0.0, random_state=12),
            position=6000, width=500, random_state=None, alpha=0.0)
stream_1.prepare_for_use()
stream_2.prepare_for_use()
stream_3.prepare_for_use()

instances_num = 10000
instances_counter = 0
ENSEMBLE_TYPE = 'av'

### Arrays for storing accuracy values for Streams
accuracies_1 = []
accuracies_2 = []
accuracies_3_mv = []
accuracies_3_av = []
accuracies_3_goowe = []

num_features = stream_1.n_features
num_targets = stream_1.n_targets
num_classes = 2
target_values = [0., 1.]
logging.info("\n\tStreams are generated and prepared for use.\n\tNumber of features: {0} - Number of targets: {1} - Number of classes: {2} - Target values: {3}"
             .format(num_features, num_targets, num_classes, target_values))

N_MAX_CLASSIFIERS = 15
CHUNK_SIZE = 500        # User-specified
WINDOW_SIZE = 100       # User-specified

### Probability of drift in streams
p1_threshold = 0.8
p2_threshold = 0.9
p3_threshold = 0.85

# Initialize the ensemble
goowe_1 = Goowe(n_max_components=N_MAX_CLASSIFIERS,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE,
              logging = False)
goowe_1.prepare_post_analysis_req(num_features, num_targets, num_classes, target_values)

# Initialize the ensemble
goowe_2 = Goowe(n_max_components=N_MAX_CLASSIFIERS,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE,
              logging = False)
goowe_2.prepare_post_analysis_req(num_features, num_targets, num_classes, target_values)


goowe_3 = GooweMS(goowe_1, goowe_2, n_max_components=N_MAX_CLASSIFIERS,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE,
              logging = False)
goowe_3.prepare_post_analysis_req(num_features, num_targets, num_classes, target_values)
# For the first chunk, there is no prediction.

X_init, y_init = stream_1.next_sample(CHUNK_SIZE)
goowe_1.partial_fit(X_init, y_init)

X_init, y_init = stream_2.next_sample(CHUNK_SIZE)
goowe_2.partial_fit(X_init, y_init)

X_init, y_init = stream_3.next_sample(CHUNK_SIZE)
#a = goowe_3.predict(X_init)
#print('==============', a)
goowe_3.update(X_init, y_init, 1, 1)
# TODO: update_from(goowe_1, goowe_2) :: updates existing goowe by selecting N_MAX_CLASSIFIERS / 2 components from each of them.

accuracy_1 = 0.0
total_1 = 0.0
true_predictions_1 = 0.0

accuracy_2 = 0.0
total_2 = 0.0
true_predictions_2 = 0.0

accuracy_3 = 0.0
total_3 = 0.0
true_predictions_3_mv = 0.0
true_predictions_3_av = 0.0
true_predictions_3_goowe = 0.0

total = 0.

for i in range(CHUNK_SIZE):
     total += 1
     cur_1 = stream_1.next_sample()
     cur_2 = stream_2.next_sample()
     cur_3 = stream_3.next_sample()
     X_1, y_1 = cur_1[0], cur_1[1]
     X_2, y_2 = cur_2[0], cur_2[1]
     X_3, y_3 = cur_3[0], cur_3[1]
     preds_1 = goowe_1.predict(X_1)
     preds_2 = goowe_2.predict(X_2)
     preds_3_mv = goowe_3.predict(X_3, ensemble_type='mv')
     preds_3_av = goowe_3.predict(X_3, ensemble_type='av')
     preds_3_goowe = goowe_3.predict(X_3, ensemble_type='goowe')
     true_predictions_1 += np.sum(preds_1 == y_1)
     true_predictions_2 += np.sum(preds_2 == y_2)
     true_predictions_3_mv += np.sum(preds_3_mv == y_3)
     true_predictions_3_av += np.sum(preds_3_av == y_3)
     true_predictions_3_goowe += np.sum(preds_3_goowe == y_3)
     accuracy_1 = true_predictions_1 / total
     accuracy_2 = true_predictions_2 / total
     accuracy_3_mv = true_predictions_3_mv / total
     accuracy_3_av = true_predictions_3_av / total
     accuracy_3_goowe = true_predictions_3_goowe / total
     print('\tSTREAM 1 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_1*100.0, 3)))
     print('\tSTREAM 2 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_2*100.0, 3)))
     print('\tSTREAM 3 :: Data instance: {} - Accuracies: MV: {} - AV: {} - Goowe: {}'.format(int(total),
         round(accuracy_3_mv*100.0, 3), round(accuracy_3_av*100.0, 3), round(accuracy_3_goowe*100.0, 3)))
     print('\t==========================================================================')
     goowe_1.partial_fit(X_1, y_1)
     goowe_2.partial_fit(X_2, y_2)
     goowe_3.update(X_3, y_3, 1, 1)

# Now, for the remaining instances, do ITTT (Interleaved Test Then Train).
while(stream_1.has_more_samples() and stream_2.has_more_samples() and instances_counter < instances_num):

    if(instances_counter % CHUNK_SIZE == 0):
        accuracy_1 = 0.0
        total_1 = 0.0
        true_predictions_1 = 0.0
        accuracy_2 = 0.0
        total_2 = 0.0
        true_predictions_2 = 0.0
        accuracy_3_mv = 0.0
        accuracy_3_av = 0.0
        accuracy_3_goowe = 0.0
        total_3 = 0.0
        true_predictions_3_mv = 0.0
        true_predictions_3_av = 0.0
        true_predictions_3_goowe = 0.0
        total = 0.

    ### Generating drifts by generating random values for each Stream
    p1 = random.random()
    p2 = random.random()
    p3 = random.random()
    if p1 > p1_threshold:
        #stream_1.generate_drift()
        logging.info('\n\tDrift generatoed for STREAM 1')
    if p2 > p2_threshold:
        #stream_2.generate_drift()
        logging.info('\n\tDrift generatoed for STREAM 2')
    if p3 > p3_threshold:
        #stream_3.generate_drift()
        logging.info('\n\tDrift generatoed for STREAM 3')
    total += 1
    cur_1 = stream_1.next_sample()
    cur_2 = stream_2.next_sample()
    cur_3 = stream_3.next_sample()
    X_1, y_1 = cur_1[0], cur_1[1]
    X_2, y_2 = cur_2[0], cur_2[1]
    X_3, y_3 = cur_3[0], cur_3[1]
    preds_1 = goowe_1.predict(X_1)
    preds_2 = goowe_2.predict(X_2)
    preds_3_goowe = goowe_3.predict(X_3, ensemble_type='goowe')
    preds_3_mv = goowe_3.predict(X_3, ensemble_type='mv')
    preds_3_av = goowe_3.predict(X_3, ensemble_type='av')
    true_predictions_1 += np.sum(preds_1 == y_1)
    true_predictions_2 += np.sum(preds_2 == y_2)
    true_predictions_3_mv += np.sum(preds_3_mv == y_3)
    true_predictions_3_av += np.sum(preds_3_av == y_3)
    true_predictions_3_goowe += np.sum(preds_3_goowe == y_3)
    accuracy_1 = true_predictions_1 / total
    accuracy_2 = true_predictions_2 / total
    accuracy_3_mv = true_predictions_3_mv / total
    accuracy_3_av = true_predictions_3_av / total
    accuracy_3_goowe = true_predictions_3_goowe / total
    accuracies_1.append(accuracy_1)
    accuracies_2.append(accuracy_2)
    accuracies_3_mv.append(accuracy_3_mv)
    accuracies_3_av.append(accuracy_3_av)
    accuracies_3_goowe.append(accuracy_3_goowe)
    #np.save('results/agrawal_'+ENSEMBLE_TYPE+'_accuracies_1.npy', np.asarray(accuracies_1))
    #np.save('results/agrawal_'+ENSEMBLE_TYPE+'_accuracies_2.npy', np.asarray(accuracies_2))
    #np.save('results/agrawal_'+ENSEMBLE_TYPE+'_accuracies_3.npy', np.asarray(accuracies_3))
    np.save('results/agrawal_accuracies_1.npy', np.asarray(accuracies_1))
    np.save('results/agrawal_accuracies_2.npy', np.asarray(accuracies_2))
    np.save('results/agrawal_mv_accuracies_3.npy', np.asarray(accuracies_3_mv))
    np.save('results/agrawal_av_accuracies_3.npy', np.asarray(accuracies_3_av))
    np.save('results/agrawal_goowe_accuracies_3.npy', np.asarray(accuracies_3_goowe))
    print('\tSTREAM 1 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_1*100.0, 3)))
    print('\tSTREAM 2 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_2*100.0, 3)))
    print('\tSTREAM 3 :: Data instance: {} - Accuracies: MV: {} - AV: {} - Goowe: {}'.format(int(total),
        round(accuracy_3_mv*100.0, 3), round(accuracy_3_av*100.0, 3), round(accuracy_3_goowe*100.0, 3)))
    print('\t==========================================================================')
    goowe_1.partial_fit(X_1, y_1)             # Then train
    goowe_2.partial_fit(X_2, y_2)             # Then train
    goowe_3.update(X_3, y_3, 1, 1)
    instances_counter += 1

# TODO: Create new goowe_3 by using components of the other two goowes with highest weights (5 from each i.e.)
# TODO: AND update goowe_3 at each chunk (each while step)
