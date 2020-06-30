from skmultiflow.data.file_stream import FileStream
import numpy as np
from Goowe import Goowe
from skmultiflow.data import SEAGenerator
import logging
from GooweMS import GooweMS

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Prepare the data stream
stream_1 = SEAGenerator()
stream_2 = SEAGenerator()
stream_3 = SEAGenerator()
stream_1.prepare_for_use()
stream_2.prepare_for_use()
stream_3.prepare_for_use()

num_features = stream_1.n_features
num_targets = stream_1.n_targets
num_classes = 2
target_values = [0., 1.]
logging.info("\n\tStreams are generated and prepared for use.\n\tNumber of features: {0} - Number of targets: {1} - Number of classes: {2} - Target values: {3}"
             .format(num_features, num_targets, num_classes, target_values))

N_MAX_CLASSIFIERS = 15
CHUNK_SIZE = 500        # User-specified
WINDOW_SIZE = 100       # User-specified

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
true_predictions_3 = 0.0

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
     preds_3 = goowe_3.predict(X_3)
     true_predictions_1 += np.sum(preds_1 == y_1)
     true_predictions_2 += np.sum(preds_2 == y_2)
     true_predictions_3 += np.sum(preds_3 == y_3)
     accuracy_1 = true_predictions_1 / total
     accuracy_2 = true_predictions_2 / total
     accuracy_3 = true_predictions_3 / total
     print('\tSTREAM 1 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_1*100.0, 3)))
     print('\tSTREAM 2 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_2*100.0, 3)))
     print('\tSTREAM 3 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_3*100.0, 3)))
     print('\t==========================================================================')
     goowe_1.partial_fit(X_1, y_1)
     goowe_2.partial_fit(X_2, y_2)
     goowe_3.update(X_3, y_3, 1, 1)

# Now, for the remaining instances, do ITTT (Interleaved Test Then Train).
while(stream_1.has_more_samples() and stream_2.has_more_samples()):
    total += 1
    cur_1 = stream_1.next_sample()
    cur_2 = stream_2.next_sample()
    cur_3 = stream_3.next_sample()
    X_1, y_1 = cur_1[0], cur_1[1]
    X_2, y_2 = cur_2[0], cur_2[1]
    X_3, y_3 = cur_3[0], cur_3[1]
    preds_1 = goowe_1.predict(X_1)            # Test
    preds_2 = goowe_2.predict(X_2)            # Test
    preds_3 = goowe_3.predict(X_3)            # Test
    true_predictions_1 += np.sum(preds_1 == y_1)
    true_predictions_2 += np.sum(preds_2 == y_2)
    true_predictions_3 += np.sum(preds_3 == y_3)
    accuracy_1 = true_predictions_1 / total
    accuracy_2 = true_predictions_2 / total
    accuracy_3 = true_predictions_3 / total
    print('\tSTREAM 1 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_1*100.0, 3)))
    print('\tSTREAM 2 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_2*100.0, 3)))
    print('\tSTREAM 3 :: Data instance: {} - Accuracy: {}'.format(int(total), round(accuracy_3*100.0, 3)))
    print('\t==========================================================================')
    goowe_1.partial_fit(X_1, y_1)             # Then train
    goowe_2.partial_fit(X_2, y_2)             # Then train
    goowe_3.update(X_3, y_3, 1, 1)

# TODO: Create new goowe_3 by using components of the other two goowes with highest weights (5 from each i.e.)
# TODO: AND update goowe_3 at each chunk (each while step)
