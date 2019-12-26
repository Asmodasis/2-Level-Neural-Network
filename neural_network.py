
import numpy as np
import random
import matplotlib as mp
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

#X, y = make_moons(200, noise=0.20)
# Helper function to evaluate the total loss on the data set 1
# model is the current version of the model { ’W1’:W1, ’b1’:b1 , ’W2’:W2, ’b2’:b2 ’} It’s a dictionary .
# X is all the training data
# y is the training labels
def calculateloss(model, X, y):
    #print("calculate loss called")  #REMOVE

    totalLoss = 0

    for n in X:                                                         # loop through the data set
      for c in y:                                                       # loop through the actual labels
        totalLoss +=  c * np.log(predict(model, n))                     # predict at the data entry and compute the loss
    totalLoss = -(1/X.size()) * totalLoss                               # divide the total loss by the sample size and negate it
    return totalLoss
    

# Helper function to predict an output (0 or 1)
# model is the current version of the model {’W1’:W1, ’b1’:b1, ’W2’:W2,’ b2’ : b2 ’}
# It’s a dictionary .
# x is one sample ( without the label )
#accepts Numpy Arrays
def predict(model, x):
    #TODO: Might be using np.zeros and .size wrong

   # print("{TEST} predict called \n\n") #remove
    K = len(model['W2'])                                                # K is the # of hidden units

    prediction = list([0 for _ in range(0, K)])                         # predict a value for all hidden units

    #print("{TEST} model[w1] in predict", model['W1']) #REMOVE
    
    for count in range(0, K-1):                                           # loop through hidden units
    
       # print("{TEST} x at position count is ", x) #REMOVE

       # print("{TEST} weight 1 at count", model['W1'][count]) #REMOVE
        #print("{TEST} weight 2", model['W2']) #REMOVE
        #print("{TEST} x in predict", x) #REMOVE
        #print("{TEST} model size in predict", len(model['W1'][count])) #REMOVE
       # print("{TEST} X size in predict", len(x)) #REMOVE
       # print("{TEST} model size in tanh", len((list(np.tanh(
        #                                                  dotProduct(x, model['W1'][count])# + model['b1'][count]
        #                                                ) 
         #                                       )))) #REMOVE
        
        
        prediction[count] = (softmax((
                                                dotProduct((list(np.tanh(
                                                          dotProduct(x, model['W1'][count]) + model['b1'][count]
                                                        ) 
                                                )),
                                                model['W2'][count])
                                                 
                                                + model['b2'][count]),
                                    K
                                    )
                            )    
                                                                        # compute all the dot products and add it up
                                                                        # add the bias to the activation
    #print("{TEST} predicition list before return is ", list(prediction))                                                                  
    return prediction

# Accepts list
def dotProduct(A, B):                                                   # computes the dot product
  #print("{TEST} dot product called")
  #print("{TEST}length of A is ", len(A))
 # print("{TEST}length of B is ", len(B))
  if len(A) != len(B):                                                  # A and B have to be same size to dot product
    raise Exception("Can't Compute the Dot Product between varying size vectors")
                                                                        # raise error if they are not
  else:                                                             
    amount = 0
    for i, j  in A, B:                                                         # loop through A and B and multiply 
      amount += i*j                                            # add amount together
  return amount                                                         # return the sum of the multipliers

def softmax(z, K):                                                      # computes the softmax of z
    yHat = 0
    for _ in range(1, K):                                               # loop up to K elements
        yHat += np.exp(z)                                               # sum over all e^z
    yHat += np.exp(z) // yHat                                           # divide e^z by  the sum
    return yHat                                                         # return the softmax


def makeRandom():
    returnVector = list([None for _ in range(2)])
    upperBound = 100

    for i in range(0, 2):                                        # fill the vector with random elements  
        returnVector[i] = random.random() % upperBound 
    return returnVector                                                  # return the random vector


# This function learns parameters for the neural network and returns the model.
# − X is the training data
# − y is the training labels
# − nn_hdim : Number of nodes in the hidden layer
# − num_passes : Number of passes through the training data for gradient descent
# − print_loss : If True , print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    #print("build model called")  #REMOVE
    
    W1, W2 = list([[0 for _ in range(0, 1)] for _ in range(len(X))]), list([[None for _ in range(2)] for _ in range(len(X))])

    b1, b2 = list([[0 for _ in range(1)] for _ in range(len(X))]), list([[None for _ in range(1)] for _ in range(len(X))])    
  
  
    for count in range(0, len(X)):
      W1[count], W2[count] = makeRandom(), makeRandom()
      
    for count in range(0, len(y)):
      b1[count], b2[count] =  makeRandom(), makeRandom()
                                                                        # initalize the weights and bias to a random value
    
    currentModel = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    #print("{TEST} printing W1 ", W1) # REMOVE

    # eta is the learning rate
    # K is the amount of hidden units we have
    eta = 0.1
    K = nn_hdim
    activations = list([[0 for _ in range(0, 1)] for _ in range(len(X))])
    inputGradient = list([[0 for _ in range(0, 1)] for _ in range(len(X))])
    outputGradient = list([[0 for _ in range(0, 1)] for _ in range(len(X))])
    #output = np.zeros(y.size, )
    error = list([0 for _ in range(0, 1)])
    
    #activations = np.zeros((len(X), 2))
    #inputGradient = np.zeros(X.size, )
    #outputGradient = np.zeros(X.size, )
    #output = np.zeros(y.size, )
    #error = np.zeros(y.size, )

    for loop in range(0, num_passes-1):
      #if loop == 1000:
      #TODO: Print the loss
      inputGradient = backPropagation(currentModel['W1'], X, y)
      outputGradient = backPropagation(currentModel['W2'], X, y)
      activations[0] = forwardPropagation(currentModel['W1'], X) 
      activations[1] = forwardPropagation(currentModel['W2'], X)

      #print("{TEST} current model in build model is ", currentModel['W1'])

      for element in range(0, len(X)-1):
          
        

          #print("{TEST} X size in build model", len(X[element])) #REMOVE
          #print("{TEST} predict in build model", predict(currentModel, X[element])) #REMOVE
          #print("{TEST} y at elemetn in build model", y[element]) #REMOVE
          print("{TEST} element count is ", element)
          error[element] = y[element] - predict(currentModel, X[element])
          
          inputGradient[element][0] += (-1)*(error[element] * activations[element][0])
          inputGradient[element][1] += (-1)*(error[element] * activations[element][1])

          for i in range(0, K):
            """
            outputGradient[element][0] += ((-1)
                                              *error[element]
                                              *(currentModel['W2'][element])
                                              *(1 - ((np.tanh(activations[i][0])**2)))
                                              *X[i])
            
            outputGradient[element][1] += ((-1)
                                              *error[element]
                                              *(currentModel['W2'][element])
                                              *(1 - ((np.tanh(activations[i][1])**2)))
                                              *X[i])
            
            outputGradient[element][0] = 0
            outputGradient[element][1] = 0

          W1[0] += (-1) * eta * inputGradient[element][0]
          W1[1] += (-1) * eta * inputGradient[element][1]
          W2[0] += (-1) * eta * outputGradient[element][0]
          W2[1] += (-1) * eta * outputGradient[element][1]
"""
      currentModel = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
      return currentModel



# The model will contain the weight and the bias
def forwardPropagation(weightList, X):
    #print("{TEST}forwardPropogation called") #REMOVE

    activations = list([0 for _ in range(len(X))])   
    returnList = list([[0 for _ in range(2)] for _ in range(len(X))])

    #print("{TEST} weight list in forwardPropagation is ", weightList)

    if len(X) != len(weightList):
        raise Exception("Can't propagate through the set!")

    #for node in range(0, weightList.size):
    for i in range(0, len(weightList)):
        #print("{TEST} weightList at i ", weightList[0], "\n\n\n") #REMOVE

        #print("{TEST} x at i ", X[i])
        activations[i] = dotProduct(weightList[i], X[i])                                                                   # The activation is the dot product of the weight and the data point for the layer
        activations[i] = np.tanh(activations[i])

        #returnList[i] = list(activations[i])

    #print("{TEST} returnList ", returnList)

    return activations

# The model will contain the weight and the bias
def backPropagation(weightList, X, y):
    #print("{TEST}backPropagation called") #REMOVE

    #convertToList = [0 for _ in range(len(weightList))]

    #print("{TEST} printing weightList ", weightList) #REMOVE

    #for num in range(0, len(weightList)):
    #convertToList[i] = [float(i) for i in weightList]

    #print("{TEST} printing convertToList ", convertToList) #REMOVE

    activations = forwardPropagation(weightList, X)
 
    networkError = list([[0 for _ in range(2)] for _ in range(len(activations))])
    gradient = list([[0 for _ in range(2)] for _ in range(len(activations))])

    for element in range(0, len(activations)):

      networkError[element] = y[element] - activations[element]


    for item in range(0, len(X)):
      for j in range(0, 1):
        #print("{TEST} network error is ", networkError[item])
        #print("{TEST} X at item is ", X[item])
        #gradient = (-1) * dotProduct(networkError[item], X[item])
        gradient[item][j] = (-1) * networkError[item] * X[item][j]
        networkError[item] = networkError[item] + ((networkError[item] * weightList[item][j] ) * (1-np.tanh(activations[item])))
    
  
    #print("{TEST} Gradient is ", gradient)
    return gradient
    
#Display decision boundary

def plot_decision_boundary(pred_func, X, y):
  #Set min and max values and give it some padding
  x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
  y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
  h = 0.01
  #Generate a grid of points with distance h between them
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  #Predict the function value for the whole gid
  Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

#Testing 

np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:,0] , X[:,1], s =40, c=y , cmap=plt.cm.Spectral)

plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
  plt.subplot(5, 2, i+1)
  plt.title('HiddenLayerSize%d' % nn_hdim)
  #print("{TEST} in main X is ", X) #REMOVE
  ##print("{TEST} in main y is ", y) #REMOVE
  
  model = build_model(X, y, nn_hdim)
  #print("{TEST} in main model is  ", model) #REMOVE
  plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()
