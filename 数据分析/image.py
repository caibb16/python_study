import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r"e:\pythonproject\test\数据分析")
image = plt.imread('Assassin\'s Creed.jpg')
print(image.shape)
plt.imshow(image)
#plt.show() 
plt.imshow(image[::-1])
#plt.show() 
plt.imshow(image[:, 1280-300:1280+300])
plt.show() 