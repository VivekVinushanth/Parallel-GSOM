# MAL_GSOM =  Multiple Aspect Learning GSOM
import threading
import time
from os.path import join
from datetime import datetime
import numpy as np

import os
import sys

sys.path.append('../../')
import copy
from util import utilities as Utils
from util import display as Display_Utils
from core4 import AspectLearnerGSOM as GSOM_Core
from params import params as Params

from gsomClassifier import GSOMClassifier


# class MAL_GSOM():
#     def __init__(
#             self,
#             aspects=2,
#             hierarchy=2,
#             aspect_gsom_array=[],
#             feature_array=[]
#
#     ):
#         self.aspects=aspects
#         self.hierarchy=hierarchy
#         self.aspect_gsom_array=aspect_gsom_array
#         self.feature_array=feature_array


# def gen_aspect_GSOMs(aspects):
#     if (aspects>0):
#         for i in range(aspects):
#             gsom = GSOMClassifier()
#             aspect_gsom_array.append(gsom)
#
# def gen_feature_array(aspects):
#     if (aspects > 0):
#         for i in range(aspects):
#             feature_array.append([5])
#
#
# def feature_producer():

emotion_feature = np.random((50,1024))
behaviour_feature = np.random((50,1024))


emo_lock = threading.Lock()
behav_lock = threading.Lock()

emotion_feature_list = []
behavior_feature_list = []
INPUT_SIZE = emotion_feature.shape[0]

class GSOM_Factory():
    def __init__(
            self,
            SF=0.83,
            forget_threshold=60,
            temporal_contexts=1,
            learning_itr=100,
            smoothing_irt=50,
            plot_for_itr=4,
    ):
        self.SF = SF
        self.forget_threshold = forget_threshold
        self.temporal_contexts = temporal_contexts
        self.learning_itr = learning_itr
        self.smoothing_irt = smoothing_irt
        self.plot_for_itr = plot_for_itr
        self.output_loc = None
        self.params = None
        self.gsom = None

    def generate_output_config(self, SF, forget_threshold):
        # File Config
        dataset = 'Classifier'
        experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        output_save_location = join('output/', experiment_id)

        # Output data config
        output_save_filename = '{}_data_'.format(dataset)
        filename = output_save_filename + str(SF) + '_T_' + str(self.temporal_contexts) + '_mage_' + str(
            forget_threshold) + 'itr'
        plot_output_name = join(output_save_location, filename)

        # Generate output plot location
        output_loc = plot_output_name
        output_loc_images = join(output_loc, 'images/')
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
        if not os.path.exists(output_loc_images):
            os.makedirs(output_loc_images)

        return output_loc, output_loc_images

    def generate_params(self, dim):
        # Init GSOM Parameters
        gsom_params = Params.GSOMParameters(self.SF, self.learning_itr, self.smoothing_irt,
                                            distance=Params.DistanceFunction.EUCLIDEAN,
                                            temporal_context_count=self.temporal_contexts,
                                            forget_itr_count=self.forget_threshold)
        generalise_params = Params.GeneraliseParameters(gsom_params)

        # Setup the age threshold based on the input vector length
        generalise_params.setup_age_threshold(dim)

        return generalise_params

    def fit(self, input_vector_database, classes):

        #  Initiate parameters
        self.params = self.generate_params(input_vector_database.shape[0])

        # Process the input files
        self.output_loc, output_loc_images = self.generate_output_config(self.SF, self.forget_threshold)

        X_train = input_vector_database

        result_dict = []
        start_time = time.time()

        self.gsom = GSOM_Core.GSOM(self.params.get_gsom_parameters(), X_train, X_train.shape[1],
                                   plot_for_itr=self.plot_for_itr,
                                   activity_classes=classes, output_loc=output_loc_images)
        self.gsom.grow()
        self.gsom.smooth()
        self.gsom_nodemap = self.gsom.assign_hits()

        print('Batch', 0)
        print('Neurons:', len(self.gsom_nodemap))
        print('Duration:', round(time.time() - start_time, 2), '(s)\n')

        result_dict.append({
            'gsom': self.gsom_nodemap,
            'aggregated': None
        })

        return result_dict, classes

    def save(self):
        result_dict = self.gsom.finalize_gsom_label()
        saved_name = Utils.Utilities.save_object(result_dict,
                                                 join(self.output_loc, 'gsom_nodemap_SF-{}'.format(self.SF)))

    def predict(self, x_test):
        y_pred = self.gsom.predict(x_test)
        return y_pred

    ####### predict from loaded model ########
    def predict_x(self, X_test, nodemap):
        y_pred = []
        gsom_nodemap = copy.deepcopy(nodemap)

        for cur_input in X_test:
            winner = Utils.Utilities.select_winner(gsom_nodemap, np.array([cur_input]))
            node_index = Utils.Utilities.generate_index(winner.x, winner.y)
            y_pred.append(winner.get_mapped_labels())
        return y_pred

    ####### Display #############
    def dispaly(self, result_dict, classes):
        gsom_nodemap = result_dict[0]['gsom']
        # Display
        display = Display_Utils.Display(result_dict[0]['gsom'], None)
        display.setup_labels_for_gsom_nodemap(classes, 2, 'Latent Space of {} : SF={}'.format("Data", self.SF),
                                              join(self.output_loc, 'latent_space_' + str(self.SF) + '_hitvalues'))
        display.setup_labels_for_gsom_nodemap(classes, 2, 'Latent Space of {} : SF={}'.format("Data", self.SF),
                                              join(self.output_loc, 'latent_space_' + str(self.SF) + '_labels'))
        print('Completed.')
