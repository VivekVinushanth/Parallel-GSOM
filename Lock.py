import threading
import numpy as np

emo_lock = threading.Condition(threading.Lock())
behav_lock = threading.Condition(threading.Lock())

emotion_feature_list = []
behavior_feature_list = []

INPUT_SIZE = 100

emotion_feature = np.random.rand(INPUT_SIZE,1024)
behaviour_feature = np.random.rand(INPUT_SIZE,1024)
emotion_label = np.random.randint(5, size=INPUT_SIZE)
behaviour_label = np.random.randint(5, size=INPUT_SIZE)





