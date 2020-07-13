import threading
import time
from os.path import join
from datetime import datetime
import numpy as np

import os
import sys

sys.path.append('../../')
from core4.AspectLearnerGSOM import AspectLearnerGSOM
from core4.AssociativeGSOM import AssociativeGSOM

from params import params as Params
import Lock


def generate_output_config(SF, forget_threshold):
    # File Config
    dataset = 'Classifier'
    experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    output_save_location = join('output/', experiment_id)

    # Output data config
    output_save_filename = '{}_data_'.format(dataset)
    filename = output_save_filename + str(SF) + '_T_' + str(temporal_contexts) + '_mage_' + str(
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


if __name__ == "__main__":
    SF = 0.83
    forget_threshold = 60
    temporal_contexts = 1
    learning_itr = 100
    smoothing_irt = 50
    plot_for_itr = 4

    # Init GSOM Parameters
    gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt,
                                        distance=Params.DistanceFunction.EUCLIDEAN,
                                        temporal_context_count=temporal_contexts,
                                        forget_itr_count=forget_threshold)
    generalise_params = Params.GeneraliseParameters(gsom_params)

    # Setup the age threshold based on the input vector length
    generalise_params.setup_age_threshold(Lock.INPUT_SIZE)

    # Process the input files
    output_loc, output_loc_images = generate_output_config(SF, forget_threshold)

    X_train_emotion = Lock.emotion_feature
    X_train_behaviour = Lock.behaviour_feature
    y_train_emotion = Lock.emotion_label
    y_train_behaviour = Lock.behaviour_label

    result_dict = []
    start_time = time.time()

    EmotionGSOM = AspectLearnerGSOM(generalise_params.get_gsom_parameters(), "emotion", X_train_emotion,
                                    X_train_emotion.shape[1],
                                    plot_for_itr=plot_for_itr,
                                    activity_classes=y_train_emotion, output_loc=output_loc_images)

    BehaviourGSOM = AspectLearnerGSOM(generalise_params.get_gsom_parameters(), "behaviour", X_train_behaviour,
                                      X_train_behaviour.shape[1],
                                      plot_for_itr=plot_for_itr,
                                      activity_classes=y_train_behaviour, output_loc=output_loc_images)

    ThreatGSOM = AssociativeGSOM(generalise_params.get_gsom_parameters(),
                                 X_train_emotion.shape[1] + X_train_behaviour.shape[1],
                                 plot_for_itr=plot_for_itr,
                                 activity_classes=y_train_behaviour, output_loc=output_loc_images)

    EmotionGSOM.start()
    BehaviourGSOM.start()
    ThreatGSOM.start()
