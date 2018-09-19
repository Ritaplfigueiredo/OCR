import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

# Load MNIST data.
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home = 'datasets/')

# Convert sklearn 'datasets bunch' object to Pandas DataFrames.
y = pd.Series(mnist.target).astype('int').astype('category')
X = pd.DataFrame(mnist.data)

# Print some stats on this dataset 

print("Shape of Training dataset: ", X.shape,  "\n Shape of labels for training dataset: ", y.shape)

''' There are 70k rows in this dataset, meaning 70k images for train. Each image in X has 784 columns, 
corresponding to 784 pixel values in a 28x28 image. The target y is a single column representing 
the true digit labels (0-9) for each training image. '''

# Change column-names in X to reflect that they are pixel values.
num_images = X.shape[1]
X.columns = ['pixel_'+str(x) for x in range(num_images)]

# Print first row of X.
print(X.head(1))

# Min, max, mean and most-common pixel-intensity values for the images. 
X_values = pd.Series(X.values.ravel())
print(" min: {}, \n max: {}, \n mean: {}, \n median: {}, \n most common value: {}".format(X_values.min(), 
                                                                                          X_values.max(), 
                                                                                          X_values.mean(),
                                                                                          X_values.median(), 
                                                                                          X_values.value_counts().idxmax()))
''' Plotting tha images. First, the plot of the first line (which corresponds to the 1st image. 
Afterwards, we will be plotting a few randomly chosen examples of digits in the dataset.'''

#First row is the first image.
first_image = X.loc[0,:]
first_label = y[0]

# 784 columns correspond to 28x28 images
plottable_image = np.reshape(first_image.values, (28,28))

# Plot the fimage
plt.imshow(plottable_image, cmap = 'gray_r')
plt.title('DigitLabel:{}'.format(first_label))
plt.show()

# Plot randomly a few more images.
images_to_plot = 9
random_indices = random.sample(range(70000), images_to_plot)

sample_images = X.loc[random_indices, :]
sample_labels = y.loc[random_indices]

plt.clf()
plt.style.use('seaborn-muted')

fig, axes = plt.subplots(3,3, 
                         figsize=(5,5),
                         sharex=True, sharey=True,
                         subplot_kw=dict(adjustable='box-forced', aspect='equal')) 

for i in range(images_to_plot):
    
    # axes (subplot) objects are stored in 2d array, accessed with axes[row,col]
    subplot_row = i//3 
    subplot_col = i%3  
    ax = axes[subplot_row, subplot_col]

    # plot image on subplot
    plottable_image = np.reshape(sample_images.iloc[i,:].values, (28,28))
    ax.imshow(plottable_image, cmap='gray_r')
    
    ax.set_title('Digit Label: {}'.format(sample_labels.iloc[i]))
    ax.set_xbound([0,28])

plt.tight_layout()
plt.show()

''' Finally, before moving forward with any classification task, we need to make sure
our data is well balanced.'''

print(y.value_counts(normalize=True))

