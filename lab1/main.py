import numpy as np
import matplotlib.pyplot as plt


def generate_liner(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)


def sigmoid(x):
    '''
    Sigmoid function, the basic activation funciton behind each layer
    Make liner regression to logistic regression 
    x.shape: batch_size * n
    '''
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x, grad_Y):
    '''
    Derivative of d Loss / d x
    formula: sigmoid(x) 
    x.shape: batch_size * n
    ret.shape: batch_size * n
    '''
    return grad_Y * sigmoid(x) * (1 - sigmoid(x))

def matrix_right_mul_W_derivative(X, grad_Y):
    '''
    Derivative of d Loss / d W
    formula Y = XW + b
    W.shape: n1 * n2
    X.shape: batch * n1
    Y.shape: batch * n2
    grad_Y.shape = batch * n2
    ret.shape: n1 * n2
    '''
    return  X.T @ grad_Y    

def matrix_right_mul_X_derivative(W, grad_Y):
    '''
    Derivative of d Loss / d W
    formula Y = XW + b
    W.shape: n1 * n2
    X.shape: batch * n1
    Y.shape: batch * n2
    grad_Y.shape = batch * n2
    ret.shape: n1 * n2
    '''
    return  grad_Y @ W.T  

def matrix_plus_derivative(grad_Y, batch_size):
    '''
    Derivative of d Loss / d b
    formula: Y = XW + b
    XW.shape: batch * n2
    Y.shape: batch * n2
    grad_Y.shape: batch * n2
    b.shape: batch * n2 (we should stack same b to the matrix)
    ret.shape: n2
    because ret is matrix, we should gain the mean grad for b to update (add each batch / batch_size) 
    '''
    return np.sum(grad_Y, axis=0) / batch_size

class SimpleNN():
    def __init__(self, batch_size, hidden_layer_size, learning_rate=0.05):
        '''
        form layers
        batch_size mean for each step, use how many samples
        hidden_layer_size.shape: 1 * 2, to decide the neurons in ith hidden_layer 
        '''
        self.batch_size = batch_size
        self.lr = learning_rate
        
        self.W1 = np.random.uniform(0,2,size=(2,hidden_layer_size[0]))
        self.b1 = np.random.uniform(0,1,size=hidden_layer_size[0])
        self.Z1 = np.zeros((batch_size, hidden_layer_size[0]))
        self.a1 = np.zeros((batch_size, hidden_layer_size[0]))
        
        self.W2 = np.random.uniform(0,2,size=(hidden_layer_size[0], hidden_layer_size[1]))
        self.b2 = np.random.uniform(0,1,size=hidden_layer_size[1])
        self.Z2 = np.zeros((batch_size, hidden_layer_size[1]))
        self.a2 = np.zeros((batch_size, hidden_layer_size[1]))
        
        self.W3 = np.random.uniform(0,2,size=(hidden_layer_size[1], 1))
        self.b3 = np.random.uniform(0,1,size=1)
        self.Z3 = np.zeros((batch_size, 1))
        self.a3 = np.zeros((batch_size, 1))

    def forward(self, X):
        
        if X.shape != (self.batch_size, 2):
            print('forward input error')
            return 0
        
        # Layer 1
        self.Z1 = X@self.W1 + np.tile(self.b1, (self.batch_size, 1))
        self.a1 = sigmoid(self.Z1)

        # Layer 2
        self.Z2 = self.a1@self.W2 + np.tile(self.b2, (self.batch_size, 1))
        self.a2 = sigmoid(self.Z2)

        # Layer 3
        self.Z3 = self.a2@self.W3 + np.tile(self.b3, (self.batch_size, 1))
        self.a3 = sigmoid(self.Z3)

        return self.a3

    def criterion(self, y, pred_y):
        '''
            y, pred_y shape: batch * 1
            With sigmoid, it's 2 classification problem
            Use cross-entropy as loss function
            for y = sigma(y * ln(y_pred))
        '''
        # pred_y may be zero
        return  - (y * np.log(pred_y) + (1-y) * np.log(1 - pred_y))
    
if __name__ == '__main__':

    X, Y = generate_liner()
    batch_size = 1
    n_epoch = 10
    model = SimpleNN(batch_size, [5,3])
    inputs = X[0:1,:]
    labels = Y[0:1,:]
    print(inputs)
    print(labels)  
    print(inputs.shape)
    print(labels.shape)
    ret = model.forward(inputs)
    print(ret)
    loss = model.criterion(labels, ret)
    print(loss)
    


