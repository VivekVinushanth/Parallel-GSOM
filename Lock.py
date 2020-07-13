import threading
import numpy as np

emo_lock = threading.Condition(threading.Lock())
behav_lock = threading.Condition(threading.Lock())

emo_smooth_lock = threading.Condition(threading.Lock())
behav_smooth_lock = threading.Condition(threading.Lock())

# Feature array
emotion_feature_list = []
behavior_feature_list = []

# Smoothing weight array
emotion_smooth_list = []
behavior_smooth_list = []

INPUT_SIZE = 50

emotion_feature = np.random.rand(INPUT_SIZE,10)
behaviour_feature = np.random.rand(INPUT_SIZE,10)
emotion_label = np.random.randint(2, size=INPUT_SIZE)
behaviour_label = np.random.randint(2, size=INPUT_SIZE)





