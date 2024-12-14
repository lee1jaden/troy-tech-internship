#Python File for Testing Concepts and Modules

"""
#Does Python input work in the Mac Terminal?
ans = input("Would you like to talk? (Yes/No) ")
if ans=="Yes":
    print ("Hello there.")
else:
    print ("Goodbye!")
"""

"""
---#%%
#Testing MatPlotLib abilities like printing images, grids
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("A_ZHandwrittenDataCopy.csv")
data.head()
all_images = data.copy()
all_labels = all_images.pop(all_images.columns[0])
all_images = np.array(all_images)
fig, axes = plt.subplots(2,2)
axes = axes.flatten()
for i, ax in enumerate(axes):
    img = all_images[i].reshape((28,28))
    ax.imshow(img, cmap="Greys")
    ax.grid()
    ax.set_title("Hi")
"""


"""
----#%%
#learning matplotlib in jupyter notebook
from matplotlib import pyplot as plt
x = [1,3,5,4,2]
y = [5,2,1,3,4]
plt.plot(x, y)
plt.show()
"""


print ("done")
# %%
