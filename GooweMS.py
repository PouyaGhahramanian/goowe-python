
import numpy as np
from Goowe import Goowe

class GooweMS(Goowe):

    # TODO: Number of Base Streams is 2.
    # Implementation for more than 2 base streams can be done
    # by changing self._goowe_1 and self._goowe_2 with a list of Goowe objects.
    def __init__(self, goowe_1, goowe_2, n_max_components: int = 10,
                 chunk_size: int = 500, window_size: int = 100, logging = True, num_streams = 2):
        super().__init__(n_max_components, chunk_size, window_size, logging)
        self._num_streams = num_streams
        self._goowe_1 = goowe_1
        self._goowe_2 = goowe_2

    '''
    def __init__(self, n_max_components: int = 10,
                 chunk_size: int = 500, window_size: int = 100, logging = True, num_streams = 2):
        super().__init__(n_max_components, chunk_size, window_size, logging)
        self._num_streams = num_streams
        self._goowe_1 = Goowe(n_max_components, chunk_size, window_size, logging)
        self._goowe_2 = Goowe(n_max_components, chunk_size, window_size, logging)

    def __init__(self, goowe_1, goowe_2, num_streams = 2):
        self._num_streams = num_streams
        self._goowe_1 = goowe_1
        self._goowe_2 = goowe_2
    '''

    def update(self, X, y, clf_num_1 = 1, clf_num_2 = 1):
        if(len(X) == 1):
            y_i = np.array([y])
            self._chunk_data.add_element(X, y_i)
            self._num_of_processed_instances += 1
            #self._update_classifiers(clf_num_1 = 1, clf_num_2 = 1)

            if(self._num_of_processed_instances % self._chunk_size == 0):
                print("Instance {}".format(self._num_of_processed_instances))
                self._update_classifiers(clf_num_1 = 1, clf_num_2 = 1)

        elif(len(X) > 1):
            for i in range(len(X)):
                X_i = np.array([X[i]])
                y_i = np.array([[y[i]]])
                self._chunk_data.add_element(X_i, y_i)
                self._num_of_processed_instances += 1
                #self._update_classifiers(clf_num_1 = 1, clf_num_2 = 1)

                if(self._num_of_processed_instances % self._chunk_size == 0):
                    print("Instance {}".format(self._num_of_processed_instances))
                    self._update_classifiers(clf_num_1 = 1, clf_num_2 = 1)

        else:
            print("Something wrong with the data...")
            print("len(X) is: {}".format(len(X)))
            raise ValueError

    def _update_classifiers(self, clf_num_1 = 1, clf_num_2 = 1):
        classifiers = []
        weights_1 = np.asarray(self._goowe_1.get_weights())
        weights_2 = np.asarray(self._goowe_2.get_weights())
        crr_clfs_1 = self._goowe_1.get_number_of_current_classifiers()
        crr_clfs_2 = self._goowe_2.get_number_of_current_classifiers()
        p_1 = clf_num_1 / (clf_num_1 + clf_num_2)
        p_2 = clf_num_2 / (clf_num_1 + clf_num_2)
        if(self._num_of_current_classifiers > 0):
            self._adjust_weights()
        indices_1 = weights_1.argsort()[(-1) * round(p_1 * crr_clfs_1):][::-1]
        indices_2 = weights_2.argsort()[(-1) * round(p_2 * crr_clfs_2):][::-1]
        clfs_1 = self._goowe_1.get_classifiers()[indices_1]
        clfs_2 = self._goowe_2.get_classifiers()[indices_2]
        clfs_1 = np.asarray(clfs_1[clfs_1 != np.array(None)]).tolist()
        clfs_2 = np.asarray(clfs_2[clfs_2 != np.array(None)]).tolist()
        classifiers = clfs_1 + clfs_2
        classifiers = classifiers[:self._num_of_max_classifiers]
        self._classifiers = np.asarray(classifiers)
        self._num_of_current_classifiers = len(self._classifiers)
        if(self._num_of_current_classifiers > 0):
            #print('DONE!')
            self._normalize_weights_softmax()       # maybe useful. we'll see.
            #print(self.get_weights())
        #self._adjust_weights()
        #self._normalize_weights_softmax()
        if(self._Logging):
            print("After normalizing weights: ")
            print(self._weights)
        '''
        print('===========================')
        print(crr_clfs_1)
        print(crr_clfs_2)
        print(round(p_1 * crr_clfs_1))
        print(round(p_2 * crr_clfs_1))
        print(len(self._classifiers))
        print(self._num_of_current_classifiers)
        print(self._classifiers)
        print('===========================')
        '''
    def prepare_post_analysis_req(self, num_features, num_targets, num_classes, target_values, record=False):
        super().prepare_post_analysis_req(num_features, num_targets, num_classes, target_values, record=False)

    def _get_components_predictions_for_instance(self, inst):
        wep = super()._get_components_predictions_for_instance(inst)
        #print('++++++++++++++++++++++++++++++++++')
        #print(wep)
        #print('++++++++++++++++++++++++++++++++++')

    def _get_components_predictions_for_instance_2(self, inst):
        preds = np.zeros((self._num_of_current_classifiers, self._num_classes))
        for k in range(len(preds)):
            kth_comp_pred = self._classifiers[k].predict_proba(inst)
            preds[k, :] = kth_comp_pred[0]
        if(self._Logging):
            print('Component Predictions:')
            print(preds)
        '''
        print('///////////////////////////////////')
        print(preds)
        print('///////////////////////////////////')
        '''
        return preds

    def _adjust_weights(self):
        super()._adjust_weights()

    def _normalize_weights(self):
        super()._normalize_weights()

    def _normalize_weights_softmax(self):
        super()._normalize_weights_softmax()

    def _process_chunk(self):
        super()._process_chunk()

    def _record_truths_this_chunk(self):
        super()._record_truths_this_chunk();

    def _record_comp_preds_this_chunk(self):
        super()._record_comp_preds_this_chunk()

    def _record_weights_this_chunk(self):
        super()._record_weights_this_chunk()

    def fit(self, X, y, classes=None, weight=None):
        super().fit(X, y, classes=None, weight=None)

    def partial_fit(self, X, y, classes=None, weight=None):
        super().partial_fit(X, y, classes=None, weight=None)

    def predict(self, X):
        #super().predict(X)
        predictions = []
        if(len(X) == 1):
            predictions.append(np.argmax(self.predict_proba(X)))
        elif(len(X) > 1):
            for i in range(len(X)):
                relevance_scores = self.predict_proba(X[i])
                predictions.append(np.argmax(relevance_scores))
        if(self._Logging):
            print('Ensemble Prediction:')
            print(np.array(predictions))
        return np.array(predictions)

    def predict_proba(self, X):
        #super().predict_proba(X)
        weights = np.array(self._weights)
        weights = weights[:self._num_of_current_classifiers]
        components_preds = self._get_components_predictions_for_instance_2(X)
        #print('*****************************')
        #print(components_preds)
        #print('*****************************')
        self._chunk_comp_preds.add_element([components_preds])
        weighted_ensemble_vote = np.dot(weights, components_preds)
        self._chunk_ensm_preds.add_element(weighted_ensemble_vote)
        return weighted_ensemble_vote

    def reset(self):
        super().reset()

    def score(self, X, y):
        super().score(X, y)

    def get_info(self):
        super().get_info()

    def get_class_type(self):
        super().get_class_type()

    def get_number_of_current_classifiers(self):
        super().get_number_of_current_classifiers()

    def get_number_of_max_classifiers(self):
        super().get_number_of_max_classifiers()
