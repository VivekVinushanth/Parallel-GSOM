import time
import os
import random
from multiprocessing import Process, Queue, Lock

emotion = np.zeros((50,1024))
behaviour = np.zeros(50,1024)

for i in iteration:
    for k in len(emotion):
        gsom1.wake()
        gsom2.wake()

        result1 = gsom1.grow(emotion[k])
        result2 = gsom2.grow(behaviour[k])

        gsom1.sleep()
        gsom2.sleep()

        ConsumerGSOM.wake()
        ConsumerGSOM.grow(result1+result2)
        ConsumerGSOM.sleep()
