import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import matplotlib 
 
# Display plots inline and change default figure size 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
 

np.random.seed(3) 
X, y = sklearn.datasets.make_moons(200, noise=0.20) 
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral) 
 

# Train the logistic rgeression classifier 
clf = sklearn.linear_model.LogisticRegressionCV() 
clf.fit(X, y) 
 

# Helper function to plot a decision boundary. 
# If you don't fully understand this function don't worry, it just generates the contour plot below. 
def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 

# Plot the decision boundary 
plt.subplot(121)
plot_decision_boundary(lambda x: clf.predict(x)) 
plt.title("Logistic Regression") 


def weight_b(m, n, dtype='w'):
    if dtype=='w':
        return np.random.normal(0,0.05, (m,n))
    elif dtype=='b':
        return np.zeros((m,n), dtype=np.float64)

hidden_layers = 10
model = {'W1': weight_b(2,hidden_layers,'w'), 'b1':weight_b(1,hidden_layers,'b'),
         'W2': weight_b(hidden_layers,2,'w'), 'b2':weight_b(1,2,'b')}

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def train(input_data, labels, W1, b1, W2, b2, num_passes, reg_lambda, epsilon):
    for i in range(num_passes):
        z1 = input_data.dot(W1) + b1 #(200, 4)
        a1 = np.tanh(z1) #(200,4)
        z2 = a1.dot(W2) + b2 #(200,1)
        exp_scores = np.exp(z2) #(200,1)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #(200,1)
        
        delta3 = probs - labels
#        delta3[range(num_examples),y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1-np.power(a1,2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

catagories = list(np.unique(y))
labels = np.zeros((y.shape[0], len(catagories)), dtype=np.float64)
for i in range(labels.shape[0]):
    labels[i, catagories.index(y[i])] = 1

train(X, labels, model['W1'], model['b1'], model['W2'], model['b2'],
      200000, 0.0001, 0.1)

plt.subplot(122)
plot_decision_boundary(lambda x: predict(model, x)) 
plt.title("Neural Network") 

plt.show()