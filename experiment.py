from skmultiflow.data.file_stream import FileStream
import numpy as np
from Goowe import Goowe


# Prepare the data stream
stream = FileStream('./datasets/sea_stream.csv')
stream.prepare_for_use()

num_features = stream.n_features
num_targets = stream.n_targets
print(stream.get_target_values())
num_classes = len(stream.get_target_values())
target_values = stream.get_target_values()

N_MAX_CLASSIFIERS = 15
CHUNK_SIZE = 500        # User-specified
WINDOW_SIZE = 100       # User-specified

# Initialize the ensemble
goowe = Goowe(n_max_components=N_MAX_CLASSIFIERS,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE,
              logging = False)
goowe.prepare_post_analysis_req(num_features, num_targets, num_classes, target_values)

# For the first chunk, there is no prediction.

X_init, y_init = stream.next_sample(CHUNK_SIZE)
print(X_init)
print(y_init)
goowe.partial_fit(X_init, y_init)

accuracy = 0.0
total = 0.0
true_predictions = 0.0

for i in range(CHUNK_SIZE):
     total += 1
     cur = stream.next_sample()
     X, y = cur[0], cur[1]
     preds = goowe.predict(X)
     true_predictions += np.sum(preds == y)
     accuracy = true_predictions / total
     print('\tData instance: {} - Accuracy: {}'.format(total, accuracy))
     goowe.partial_fit(X, y)

# Now, for the remaining instances, do ITTT (Interleaved Test Then Train).
while(stream.has_more_samples()):
    total += 1
    cur = stream.next_sample()
    X, y = cur[0], cur[1]
    preds = goowe.predict(X)            # Test
    true_predictions += np.sum(preds == y)
    accuracy = true_predictions / total
    print('\tData instance: {} - Accuracy: {}'.format(int(total), round(accuracy*100.0, 3)))
    goowe.partial_fit(X, y)             # Then train
