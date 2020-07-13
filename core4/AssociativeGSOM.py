import math
import numpy as np
import copy
import time

import scipy
import pandas as pd
from tqdm import tqdm
from core4 import growth_handler as Growth_Handler
from core4 import elements as Elements
from util import utilities as Utils
from util import display as Display_Utils
import threading
import Lock

np.random.seed(8)


class AssociativeGSOM(threading.Thread):

    def __init__(self, params, dimensions, plot_for_itr=0, activity_classes=None, output_loc=None):
        threading.Thread.__init__(self)
        self.parameters = params
        self.growth_handler = Growth_Handler.GrowthHandler()
        self.dimensions = dimensions
        # self.learn_smooth_sample_size = self.parameters.get_learn_smooth_sample_size(len(inputs))
        self.gsom_nodemap = {}
        self.plot_for_itr = plot_for_itr
        self.display = Display_Utils.Display(None, None)
        self.activity_classes = activity_classes
        self.output_save_location = output_loc

        # Parameters for recurrent gsom
        self.globalContexts = np.zeros((self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS, self.dimensions))
        self.globalContexts_evaluation = np.zeros((self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS, self.dimensions))
        self.alphas = Utils.Utilities.get_decremental_alphas(self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS)
        self.previousBMU = np.zeros((1, self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS, self.dimensions))
        self.previousBMU_evaluation = np.zeros((1, self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS, self.dimensions))

    def run(self):
        self.grow()
        Lock.emo_lock.acquire()
        Lock.emo_lock.notify()
        Lock.emo_lock.release()

        Lock.behav_lock.acquire()
        Lock.behav_lock.notify()
        Lock.behav_lock.release()

        self.smooth()

        Lock.emo_smooth_lock.acquire()
        Lock.emo_smooth_lock.notify()
        Lock.emo_smooth_lock.release()

        Lock.behav_smooth_lock.acquire()
        Lock.behav_smooth_lock.notify()
        Lock.behav_smooth_lock.release()

        self.assign_hits()

    def grow(self):

        self._initialize_network(self.dimensions)
        param = self.parameters

        # Optimise python references: that are reevaluated each time through the loop
        grow_in = self._grow_for_single_iteration_and_single_input

        learning_rate = param.START_LEARNING_RATE
        # start_time = time.time()
        pbar = tqdm(range(0, param.LEARNING_ITERATIONS),
                    desc='Consumer Learning ' + str(param.LEARNING_ITERATIONS) + ' iterations')
        for i in pbar:

            if i != 0:
                learning_rate = self._get_learning_rate(param, learning_rate, len(self.gsom_nodemap))

            neighbourhood_radius = self._get_neighbourhood_radius(param.LEARNING_ITERATIONS, i,
                                                                  param.MAX_NEIGHBOURHOOD_RADIUS)

            # Consuming combined_weights
            for k in range(0, Lock.INPUT_SIZE):
                # Consume one item

                Lock.emo_lock.acquire()
                # print("Consumer thread acquired emotion lock -----", k, "\n")
                while k > len(Lock.emotion_feature_list) - 1:
                    # print("Consumer thread waiting becoz k is greater (emo)----", k, "\n")
                    Lock.emo_lock.notify()

                    Lock.emo_lock.wait()
                emotion = Lock.emotion_feature_list[k]

                if k == Lock.INPUT_SIZE - 1:
                    Lock.emotion_feature_list = []

                Lock.emo_lock.notify()
                Lock.emo_lock.release()

                Lock.behav_lock.acquire()
                # print("Consumer thread acquired behavior lock -----", k, "\n")
                while k > len(Lock.behavior_feature_list) - 1:
                    # print("Consumer thread waiting becoz k is greater (behav)----", k, "\n")
                    Lock.behav_lock.notify()

                    Lock.behav_lock.wait()
                behaviour = Lock.behavior_feature_list[k]
                if k == Lock.INPUT_SIZE - 1:
                    Lock.behavior_feature_list = []

                Lock.behav_lock.notify()
                Lock.behav_lock.release()

                data = np.hstack((emotion, behaviour))
                grow_in(data, learning_rate, neighbourhood_radius)

            # Remove all the nodes above the age threshold
            Utils.Utilities.remove_older_nodes(self.gsom_nodemap, self.parameters.AGE_THRESHOLD)

        # END of learning iterations
        return self.gsom_nodemap

    def smooth(self):

        learning_rate = self.parameters.START_LEARNING_RATE * self.parameters.SMOOTHING_LEARNING_RATE_FACTOR
        reduced_neighbourhood_radius = self.parameters.MAX_NEIGHBOURHOOD_RADIUS * self.parameters.SMOOTHING_NEIGHBOURHOOD_RADIUS_FACTOR

        smooth = self._smooth_for_single_iteration_and_single_input

        pbar = tqdm(range(0, self.parameters.SMOOTHING_ITERATIONS),
                    desc='Consumer Smoothing ' + str(self.parameters.SMOOTHING_ITERATIONS) + ' iterations')
        for i in pbar:

            if i != 0:
                learning_rate = self._get_learning_rate(self.parameters, learning_rate, len(self.gsom_nodemap))

            neighbourhood_radius = self._get_neighbourhood_radius(self.parameters.SMOOTHING_ITERATIONS, i,
                                                                  reduced_neighbourhood_radius)

            for k in range(0, Lock.INPUT_SIZE):
                # Consume one item

                Lock.emo_smooth_lock.acquire()
                # print("Consumer thread acquired emotion smoothing lock -----", k, "\n")
                while k > len(Lock.emotion_smooth_list) - 1:
                    # print("Consumer thread waiting becoz k is greater (emo)----", k, "\n")
                    Lock.emo_smooth_lock.notify()

                    Lock.emo_smooth_lock.wait()
                emotion = Lock.emotion_smooth_list[k]

                if k == Lock.INPUT_SIZE - 1:
                    Lock.emotion_smooth_list = []

                Lock.emo_smooth_lock.notify()
                Lock.emo_smooth_lock.release()

                Lock.behav_smooth_lock.acquire()
                # print("Consumer thread acquired behavior smoothing lock -----", k, "\n")
                while k > len(Lock.behavior_smooth_list) - 1:
                    # print("Consumer thread waiting becoz k is greater (behav)----", k, "\n")
                    Lock.behav_smooth_lock.notify()

                    Lock.behav_smooth_lock.wait()
                behaviour = Lock.behavior_smooth_list[k]
                if k == Lock.INPUT_SIZE - 1:
                    Lock.behavior_smooth_list = []

                Lock.behav_smooth_lock.notify()
                Lock.behav_smooth_lock.release()

                smoothing_data = np.hstack((emotion, behaviour))

                smooth(smoothing_data, learning_rate, neighbourhood_radius)

        # End of smoothing iterations
        # return self.gsom_nodemap

    """
    This function to be called for a current dataset that used to train the gsom, to evaluate the hit nodes.
    """

    def assign_hits(self):

        param = self.parameters
        gsom_nodemap = self.gsom_nodemap
        curr_count = 0

        for k in range(0, Lock.INPUT_SIZE):
            # Consume one item

            Lock.emo_assign_lock.acquire()
            print("Consumer thread acquired emotion assign lock -----", k, "\n")
            while k > len(Lock.emotion_assign_list) - 1:
                print("Consumer thread waiting becoz k is greater (emo)----", k, "\n")
                Lock.emo_assign_lock.notify()

                Lock.emo_assign_lock.wait()
            emotion = Lock.emotion_assign_list[k]

            # if k == Lock.INPUT_SIZE - 1:
            #     Lock.emotion_assign_list = []

            Lock.emo_assign_lock.notify()
            Lock.emo_assign_lock.release()

            Lock.behav_assign_lock.acquire()
            print("Consumer thread acquired behavior assign lock -----", k, "\n")
            while k > len(Lock.behavior_assign_list) - 1:
                print("Consumer thread waiting becoz k is greater (behav)----", k, "\n")
                Lock.behav_assign_lock.notify()

                Lock.behav_assign_lock.wait()
            behaviour = Lock.behavior_assign_list[k]
            # if k == Lock.INPUT_SIZE - 1:
            #     Lock.behavior_assign_list = []

            Lock.behav_assign_lock.notify()
            Lock.behav_assign_lock.release()

            assign_data = np.hstack((emotion, behaviour))


            self.globalContexts[0] = assign_data

            # Update global context
            for z in range(1, param.NUMBER_OF_TEMPORAL_CONTEXTS):
                self.globalContexts[z] = (param.BETA * self.previousBMU[0, z]) + (
                        (1 - param.BETA) * self.previousBMU[0, z - 1])

            winner = Utils.Utilities.select_winner_recurrent(gsom_nodemap, self.globalContexts, self.alphas)
            winner.hit()

            # Recurrent learning set previous BMU
            self.previousBMU[0] = winner.recurrent_weights

            node_index = Utils.Utilities.generate_index(winner.x, winner.y)
            self.gsom_nodemap[node_index].map_label_indexes(curr_count)
            self.gsom_nodemap[node_index].map_label(self.activity_classes[curr_count])
            curr_count += 1

        # return the finalized map
        # return self.gsom_nodemap

    """
    This function to be called for a separate dataset, to evaluate the hit nodes.
    """

    def evaluate_hits(self):

        param = self.parameters
        gsom_nodemap = copy.deepcopy(self.gsom_nodemap)
        curr_count = 0

        for cur_input in inputs:

            self.globalContexts_evaluation[0] = cur_input

            # Update global context
            for z in range(1, param.NUMBER_OF_TEMPORAL_CONTEXTS):
                self.globalContexts_evaluation[z] = (param.BETA * self.previousBMU_evaluation[0, z]) + (
                            (1 - param.BETA) * self.previousBMU_evaluation[0, z - 1])

            winner = Utils.Utilities.select_winner_recurrent(gsom_nodemap, self.globalContexts_evaluation, self.alphas)
            winner.hit()

            # Recurrent learning set previous BMU
            self.previousBMU_evaluation[0] = winner.recurrent_weights

            node_index = Utils.Utilities.generate_index(winner.x, winner.y)
            gsom_nodemap[node_index].map_label(curr_count)
            curr_count += 1

        # return the finalized map
        return gsom_nodemap

    def finalize_gsom_label(self):
        # all_coordinates = self.node_labels.iloc[:, 4:]
        # all_coordinates = all_coordinates.astype(int)
        results = []
        removable_indexes = []
        for key, value in self.gsom_nodemap.items():
            key_split = key.split(':')
            x = int(key_split[0])
            y = int(key_split[1])

            if value.get_hit_count() > 0:
                count_0 = 0
                count_1 = 0

                labels = value.get_mapped_labels()

                # f_label = max(set(labels), key=labels.count)
                df = pd.DataFrame({'Number': labels})
                # df = df.astype('int32')
                hello = df['Number'].value_counts()
                max_occurances = df['Number'].mode()

                if (len(max_occurances) == 1):
                    f_label = max_occurances[0]
                    self.gsom_nodemap[key].change_label(f_label)

                else:
                    # cripkeu=np.empty((len(value.get_mapped_labels_indexes()),55))
                    X_weights = []
                    label_indexes = value.get_mapped_labels_indexes()
                    for i in label_indexes:
                        X_weight = inputs[i, :]
                        # np.append(cripkeu,sdgg,axis=0)
                        X_weights.append(X_weight)
                    X_weights = np.asarray(X_weights)
                    neutral_node = value.recurrent_weights.reshape(1, value.dimensions)
                    out = scipy.spatial.distance.cdist(X_weights, neutral_node, 'euclidean')

                    nearest_index = out.argmin()
                    nearest_neighbor_index = label_indexes[nearest_index]
                    nearest_neighbor = self.activity_classes[nearest_neighbor_index]

                    self.gsom_nodemap[key].change_label(nearest_neighbor)
                    # neutral_indexes.append(key)
            else:
                removable_indexes.append(key)
        # print(neutral_indexes)
        for key in removable_indexes:
            del self.gsom_nodemap[key]
        # print(neutral_indexes)
        results.append({
            'gsom': self.gsom_nodemap,
            'aggregated': None
        })
        return results

    def predict(self, X_test):
        y_pred = []
        param = self.parameters
        gsom_nodemap = copy.deepcopy(self.gsom_nodemap)

        for cur_input in X_test:
            self.globalContexts_evaluation[0] = cur_input

            # Update global context
            for z in range(1, param.NUMBER_OF_TEMPORAL_CONTEXTS):
                self.globalContexts_evaluation[z] = (param.BETA * self.previousBMU_evaluation[0, z]) + (
                        (1 - param.BETA) * self.previousBMU_evaluation[0, z - 1])

            winner = Utils.Utilities.select_winner_recurrent(gsom_nodemap, self.globalContexts_evaluation, self.alphas)
            node_index = Utils.Utilities.generate_index(winner.x, winner.y)
            y_pred.append(winner.get_mapped_labels())
        return y_pred

    def _smooth_for_single_iteration_and_single_input(self, input_vector, learning_rate, neigh_radius):

        param = self.parameters
        gsom_nodemap = self.gsom_nodemap
        self.globalContexts[0] = input_vector

        # Update global context
        for z in range(1, param.NUMBER_OF_TEMPORAL_CONTEXTS):
            self.globalContexts[z] = (param.BETA * self.previousBMU[0, z]) + (
                        (1 - param.BETA) * self.previousBMU[0, z - 1])

        winner = Utils.Utilities.select_winner_recurrent(gsom_nodemap, self.globalContexts, self.alphas)

        # Recurrent learning set previous BMU
        self.previousBMU[0] = winner.recurrent_weights

        # Adjust the weight of the winner
        winner.adjust_weights(self.globalContexts, 1, learning_rate)

        # habituate the neuron
        winner.habituate_neuron(self.parameters.TAU_B)

        left = Utils.Utilities.generate_index(winner.x - 1, winner.y)
        right = Utils.Utilities.generate_index(winner.x + 1, winner.y)
        top = Utils.Utilities.generate_index(winner.x, winner.y + 1)
        bottom = Utils.Utilities.generate_index(winner.x, winner.y - 1)

        if left in gsom_nodemap:
            self._adjust_weights_for_neighbours(gsom_nodemap[left], winner, neigh_radius, learning_rate)
        elif right in gsom_nodemap:
            self._adjust_weights_for_neighbours(gsom_nodemap[right], winner, neigh_radius, learning_rate)
        elif top in gsom_nodemap:
            self._adjust_weights_for_neighbours(gsom_nodemap[top], winner, neigh_radius, learning_rate)
        elif bottom in gsom_nodemap:
            self._adjust_weights_for_neighbours(gsom_nodemap[bottom], winner, neigh_radius, learning_rate)
        # dhtfjhfgjgf

    def _grow_for_single_iteration_and_single_input(self, input_vector, learning_rate, neigh_radius):

        # Set local references to self variables
        param = self.parameters
        gsom_nodemap = self.gsom_nodemap
        self.globalContexts[0] = input_vector

        # Update global context
        for z in range(1, param.NUMBER_OF_TEMPORAL_CONTEXTS):
            self.globalContexts[z] = (param.BETA * self.previousBMU[0, z]) + (
                        (1 - param.BETA) * self.previousBMU[0, z - 1])

        # Select the winner
        winner = Utils.Utilities.select_winner_recurrent(gsom_nodemap, self.globalContexts, self.alphas)
        winner_key = Utils.Utilities.generate_index(winner.x, winner.y)

        # Recurrent learning set previous BMU. i.e., b(t-1)
        self.previousBMU[0] = winner.recurrent_weights

        # Update the age of the winner and increment age for all.
        winner.fired_in_growing()
        # TODO: Future improvement - increment node ages only for non-neighbours. (Think of time complexity)
        Utils.Utilities.increment_node_ages(gsom_nodemap)

        # Adjust the weight of the winner
        winner.adjust_weights(self.globalContexts, 1, learning_rate)

        # Update the error value of the winner node
        winner.cal_and_update_error(self.globalContexts, self.alphas)

        # habituate the neuron
        winner.habituate_neuron(self.parameters.TAU_B)

        # Weight adaptation for winner's neighborhood
        adjust = self._adjust_weights_for_neighbours
        for key, node in gsom_nodemap.items():
            # Exclude winner from the nodemap since winner's weight has already been updated in the previous step
            if key != winner_key:
                adjust(node, winner, neigh_radius, learning_rate)

        # Evaluate winner's weights and grow network it it's above Growth Threshold (GT)
        if winner.error >= param.get_gt(len(input_vector)):
            self._adjust_winner_error(winner, len(input_vector))

    def _adjust_winner_error(self, winner, dimensions):

        left = Utils.Utilities.generate_index(winner.x - 1, winner.y)
        right = Utils.Utilities.generate_index(winner.x + 1, winner.y)
        top = Utils.Utilities.generate_index(winner.x, winner.y + 1)
        bottom = Utils.Utilities.generate_index(winner.x, winner.y - 1)

        if left in self.gsom_nodemap and right in self.gsom_nodemap and top in self.gsom_nodemap and bottom in self.gsom_nodemap:
            # If the network has adequate neurons to process the input data, the weight vectors of those neurons are
            #  adapted as such the distribution of the weight vectors will represent the input vector distribution.
            self._distribute_error_to_neighbours(winner, left, right, top, bottom, dimensions)
        else:
            # If the network does not have sufficient neurons, the weight will be accumulated on a single neuron.
            self.growth_handler.grow_nodes(self.gsom_nodemap, winner, self.globalContexts)

    def _distribute_error_to_neighbours(self, winner, left, right, top, bottom, dimensions):

        winner.error = self.parameters.get_gt(dimensions)
        self.gsom_nodemap[left].error = self._calc_error_for_neighbours(self.gsom_nodemap[left])
        self.gsom_nodemap[right].error = self._calc_error_for_neighbours(self.gsom_nodemap[right])
        self.gsom_nodemap[top].error = self._calc_error_for_neighbours(self.gsom_nodemap[top])
        self.gsom_nodemap[bottom].error = self._calc_error_for_neighbours(self.gsom_nodemap[bottom])

    def _calc_error_for_neighbours(self, node):
        return node.error * (1 + self.parameters.FD)

    def _adjust_weights_for_neighbours(self, node, winner, neigh_radius, learning_rate):

        node_dist_sqr = math.pow(winner.x - node.x, 2) + math.pow(winner.y - node.y, 2)
        neigh_radius_sqr = neigh_radius * neigh_radius

        if node_dist_sqr < neigh_radius_sqr:
            # update the weight vector of the neighbour
            influence = math.exp(- node_dist_sqr / (2 * neigh_radius_sqr))
            node.adjust_weights(self.globalContexts, influence, learning_rate)

            # habituate the neuron
            node.habituate_neuron(self.parameters.TAU_N)

    def _initialize_network(self, dimensions):

        # Generate the node map for initial GSOM layer - for all the inputs
        self.gsom_nodemap = {
            '0:0': Elements.GSOMNode(0, 0, np.random.rand(dimensions),
                                     np.zeros((self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS - 1, self.dimensions))),
            '0:1': Elements.GSOMNode(0, 1, np.random.rand(dimensions),
                                     np.zeros((self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS - 1, self.dimensions))),
            '1:0': Elements.GSOMNode(1, 0, np.random.rand(dimensions),
                                     np.zeros((self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS - 1, self.dimensions))),
            '1:1': Elements.GSOMNode(1, 1, np.random.rand(dimensions),
                                     np.zeros((self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS - 1, self.dimensions))),
        }

    def _get_learning_rate(self, parameters, prev_learning_rate, nodemap_size):
        return parameters.ALPHA * (1 - (parameters.R / nodemap_size)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration, max_neighbourhood_radius):
        time_constant = total_iteration / math.log(max_neighbourhood_radius)
        return max_neighbourhood_radius * math.exp(- iteration / time_constant)
