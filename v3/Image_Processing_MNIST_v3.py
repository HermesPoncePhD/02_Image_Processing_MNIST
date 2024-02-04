## IMAGE PROCESSING USING MNIST DATABSE ##
# Created by: Hermes Ponce
#===================================================================
# APPROACH:
# For this competition, we will be using Keras (with TensorFlow as our backend) as 
# the main package to create a simple neural network to predict, as accurately as we can, 
# digits from handwritten images. In particular, we will be calling the Functional Model API 
# of Keras, and creating a 4-layered and 5-layered neural network.

# Also, we will be experimenting with various optimizers: the plain vanilla Stochastic Gradient 
# Descent optimizer and the Adam optimizer. However, there are many other parameters, such as 
# training epochs which will we will not be experimenting with.

# In addition, the choice of hidden layer units are completely arbitrary and may not be optimal.
# This is yet another parameter which we will not attempt to tinker with. Lastly, we introduce 
# dropout, a form of regularisation, in our neural networks to prevent overfitting.

# RESULT:
# Following our simulations on the cross validation dataset, it appears that a 4-layered neural 
# network, using 'Adam' as the optimizer along with a learning rate of 0.01, performs best. 
# We proceed to introduce dropout in the model, and use the model to predict for the test set.

# The test predictions (submitted to Kaggle) generated by our model predicts with an accuracy 
# score of 97.600%, which places us at the top 55 percentile of the competition.

#===================================================================
from IPython import get_ipython
get_ipython().magic('reset -sf')








































