import ftp.ftpClient as ftp
import numpy as np


p = np.zeros(3)

p[1] = 0.8

best_predicted_arm = np.argmax(p)

print(best_predicted_arm)

