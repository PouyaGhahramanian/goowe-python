import numpy as np
from skmultiflow.core.base import StreamModel
from skmultiflow.trees import HoeffdingTree
from skmultiflow.utils.data_structures import InstanceWindow, FastBuffer


class Goowe(StreamModel):
    """ GOOWE (Geometrically Optimum Online Weighted Ensemble), as it is
    described in Bonab and Can (2017). Common notation in the code is
    as follows:
        K for maximum number of classifiers in the ensemble.
        N for data instances.
        A, d as they are, in the aforementioned paper.


    Parameters
    ----------
    n_max_components: int
        Ensemble size limit. Maximum number of component classifiers.
    chunk_size: int
        The amount of instances necessary for ensemble to learn concepts from.
        At each chunk_size many instances, some training is done.
    window_size: int
        Size of sliding window, which keeps record of the last k instances
        that are encountered in the data stream.
    """

    def __init__(self, n_max_components: int = 10,
                 chunk_size: int = 500, window_size: int = 100):
        super().__init__()
        self._num_of_max_classifiers = n_max_components
        self._chunk_size = chunk_size

        self._num_of_current_classifiers = 0
        self._num_of_processed_instances = 0
        self._classifiers = np.array((self._num_of_max_classifiers))
        self._weights = np.zeros((self._num_of_max_classifiers))

        # What to save from current Data Chunk --> will be used for
        # adjusting weights, pruning purposes and so on.
        # Individual predictions of components, overall prediction of ensemble,
        # and ground truth info.
        self._chunk_comp_preds = FastBuffer(max_size=chunk_size)
        self._chunk_ensm_preds = FastBuffer(max_size=chunk_size)

        # chunk_data has instances in the chunk and their ground truth.
        # To be initialized after receiving n_features, n_targets
        self._chunk_data = None
        # self._chunk_truths = FastBuffer(max_size=chunk_size)

        # TODO: Implement Sliding Window Continuous Evaluator.
        # What to save at Sliding Window (last n instances) --> will be
        # used for continuous evaluation.
        # self._sliding_window_ensemble_preds =FastBuffer(max_size=window_size)
        # self._sliding_window_truths = FastBuffer(max_size=window_size)

    def prepare_post_analysis_req(self, num_features, num_targets):
        # Need to get the dataset information but we do not want to
        # take it as an argument to the classifier itself, nor we do want to
        # ask it at each data instance. Hence we take dataset info from user
        # explicitly to create _chunk_data entries.
        self._chunk_data = InstanceWindow(n_features=num_features,
                                          n_targets=num_targets)
        return

    def _get_components_predictions_for_instance(self, inst):
        """ For a given data instance, takes predictions of
        individual components from the ensemble as a matrix.

        Parameters
        ----------
        inst: data instance for which votes of components are delivered.

        Returns
        ----------
        numpy.array
            A 2-d numpy array where each row corresponds to predictions of
            each classifier.
        """
        preds = np.ndarray(shape=(self._num_of_current_classifiers,
                                  self._num_of_classes))
        for k in range(len(preds)):
            kth_comp_pred = self._classifiers[k].get_votes_for_instance(inst)
            print("Component {}'s Prediction: {}".format(k, kth_comp_pred))
            preds[k] = kth_comp_pred

        return preds

    def _adjust_weights(self):
        """ Weight adustment by solving linear least squares, as it is
        described in Bonab and Can (2017).
        """
        # Prepare variables for Weight Adjustment
        A = np.zeros(shape=(self._num_of_current_classifiers,
                            self._num_of_current_classifiers))
        d = np.zeros(shape=(self._num_of_current_classifiers,))

        # Go over all the data chunk, calculate values of (S_i x S_j) for A.
        # (S_i x O) for d.
        y_all = self._chunk_data.get_targets_matrix()
        for i in range(len(y_all)):
            class_index = y_all[i]
            A = A + self._chunk_comp_preds[i].dot(self._chunk_comp_preds[i].T)
            d = d + self._chunk_comp_preds[i, class_index]

        # A and d are filled. Now, the linear system Aw=d to be solved
        # to get our desired weights. w is of size K.
        w = np.linalg.solve(A, d)

        # _weights has maximum size but what we found can be
        # smaller. Therefore, need to put the values of w to global weights
        if(self._num_of_current_classifiers < self._num_of_max_classifiers):
            for i in range(len(w)):
                self._weights[i] = w[i]
        else:                             # If full size, there is no problem.
            self._weights = w
        return

    def _normalize_weights(self):
        """ Normalizes the weights of the ensemble to (0, 1) range.
        Performs (x_i - min(x)) / (max(x) - min(x)) on the nonzero elements
        of the weight vector.
        """
        min = np.amin(self._weights[:self._num_of_current_classifiers])
        max = np.amax(self._weights[:self._num_of_current_classifiers])
        self._weights = (self._weights - min) / (max - min)
        return

    def _process_chunk(self):
        """ A subroutine that runs at the end of each chunk, allowing
        the components to be trained and ensemble weights to be adjusted.
        Until the first _process_chunk call, the ensemble is not yet ready.
        At first call, the first component is learned.
        At the rest of the calls, new components are formed, and the older ones
        are trained by the given chunk.
        If the ensemble size is reached, then the lowest weighted component is
        removed from the ensemble.
        """
        new_clf = HoeffdingTree()  # with default parameters for now
        new_clf.reset()

        # Case 1: No classifier in the ensemble yet, first chunk:
        if(self._num_of_current_classifiers == 0):
            self._classifiers[0] = new_clf
            self._weights[0] = 1.0  # weight is 1 for the first clf
            self._num_of_current_classifiers += 1
        else:
            # First, adjust the weights of the old component classifiers
            # according to what happened in this chunk.
            self._adjust_weights()
            self._normalize_weights()       # maybe useful. we'll see.

            # Case 2: There are classifiers in the ensemble but
            # the ensemble size is still not capped.
            if(self._num_of_current_classifiers < self._num_of_max_classifiers):
                # Put the new classifier to ensemble with the weight of 1
                self._classifiers[self._num_of_current_classifiers] = new_clf
                self._weights[self._num_of_current_classifiers] = 1.0
                self._num_of_current_classifiers += 1

            # Case 3: Ensemble size is capped. Need to replace the component
            # with lowest weight.
            else:
                assert (self._num_of_current_classifiers
                        == self._num_of_max_classifiers), "Ensemble not full."
                index_of_lowest_weight = np.argmin(self._weights)
                self._classifiers[index_of_lowest_weight] = new_clf
                self._weights[index_of_lowest_weight] = 1.0

        # Ensemble maintenance is done. Now train all classifiers
        # in the ensemble from the current chunk.
        # Can be parallelized.
        data_features = self._chunk_data.get_attributes_matrix()
        data_truths = self._chunk_data.get_targets_matrix()

        print("Starting training the components with the current chunk...")
        for k in range(len(self._num_of_current_classifiers)):
            for i in range(len(self._chunk_size)):
                # Classifier (e.g. HT) must have partial_fit() for training
                X = data_features[i]
                y = data_truths[i]
                self._classifiers[k].partial_fit(X, y)
        print("Training the components with the current chunk completed...")
        return

    # --------------------------------------------------
    # Overridden methods from the parent (StreamModel)
    # --------------------------------------------------
    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError("For now, only the stream version "
                                  "is implemented. Use partial_fit()")

    def partial_fit(self, X, y, classes=None, weight=None):
        # If still filling the chunk, then just add the instance to the
        # current data chunk, wait for it to be filled.
        self._num_of_processed_instances += 1

        # Save X and y to train classifiers later.
        self._chunk_data.add_element(X, y)

        # If at the end of a chunk, start training components
        # and adjusting weights using information in this chunk.
        if(self._num_of_processed_instances % self._chunk_size == 0):
            self._process_chunk(self._current_chunk)

    def predict(self, X):
        """ For a given data instance, yields the binary prediction values.

        Parameters
        ----------
        X: data instance for which prediction is done.

        Returns
        ----------
        numpy.array
            A vector with number_of_classes elements where only the
            class that is predicted as correct is 1 and the rest is 0.
        """
        relevance_scores = self.predict_proba(X)
        single_label_prediction = np.zeros_like(relevance_scores)

        # Now, do a single-label prediction as
        # the highest value of those scores to be the correct class.
        single_label_prediction[np.argmax(relevance_scores)] = 1
        return single_label_prediction

    def predict_proba(self, X):
        """ For a given data instance, takes WEIGHTED combination
        of components to get relevance scores for each class.

        Parameters
        ----------
        X: data instance for which weighted combination is delivered.

        Returns
        ----------
        numpy.array
            A vector with number_of_classes elements where each element
            represents class score of corresponding class for this instance.
        """
        weights = np.array(self._weights)
        components_preds = self._get_components_predictions_for_instance(X)

        # Save individual component predictions and ensemble prediction
        # for later analysis.
        self._chunk_comp_preds.add_element(components_preds)

        weighted_ensemble_vote = np.dot(components_preds.T, weights)
        self._chunk_ensm_preds.add_element(weighted_ensemble_vote)

        return weighted_ensemble_vote

    def reset(self):
        pass

    def score(self, X, y):
        pass

    def get_info(self):
        pass

    def get_class_type(self):
        pass

    # Some getters..
    def get_number_of_current_classifiers(self):
        return self._num_of_current_classifiers

    def get_number_of_max_classifiers(self):
        return self._num_of_max_classifiers