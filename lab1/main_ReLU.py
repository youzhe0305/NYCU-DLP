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

def generate_XOR_easy(n=10):
    # outupt 2n+1 samples
    inputs = []
    labels = []
    for i in range(n+1):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(2*n+1,1)


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

def ReLU(x):
    '''
    ReLU function, additional activation funciton
    x.shape: batch_size * n
    '''
    return np.maximum(0,x)

def ReLU_derivative(x, grad_Y):
    '''
    Derivative of d Loss / d x
    formula: ReLu(x) 
    x.shape: batch_size * n
    grad_Y.shape: batch_size * n
    ret.shape: batch_size * n
    '''
    return (x >= 0).astype(int) * grad_Y

def Leaky_ReLU(x, alpha = 0.01):
    '''
    ReLU function, additional activation funciton
    x.shape: batch_size * n
    '''
    return np.where(x>0, x, alpha * x)

def Leaky_ReLU_derivative(x, grad_Y, alpha=0.01):
    '''
    Derivative of d Loss / d x
    formula: Leakly_ReLu(x) 
    x.shape: batch_size * n
    grad_Y.shape: batch_size * n
    ret.shape: batch_size * n
    '''
    return np.where(x>0, 1, alpha) * grad_Y

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
    def __init__(self, batch_size, hidden_layer_size, learning_rate=0.1):
        '''
        form layers
        batch_size mean for each step, use how many samples
        hidden_layer_size.shape: 1 * 2, to decide the neurons in ith hidden_layer 
        '''
        self.batch_size = batch_size
        self.lr = learning_rate
        self.EPS = 1e-8
        
        self.W1 = np.random.uniform(0,1,(2,hidden_layer_size[0]))
        self.b1 = np.zeros(hidden_layer_size[0])
        self.Z1 = np.zeros((batch_size, hidden_layer_size[0]))
        self.a1 = np.zeros((batch_size, hidden_layer_size[0]))
        
        self.W2 = np.random.uniform(0,1,(hidden_layer_size[0], hidden_layer_size[1]))
        self.b2 = np.zeros(hidden_layer_size[1])
        self.Z2 = np.zeros((batch_size, hidden_layer_size[1]))
        self.a2 = np.zeros((batch_size, hidden_layer_size[1]))
        
        self.W3 = np.random.uniform(0,1,(hidden_layer_size[1], 1))
        self.b3 = np.zeros(1)
        self.Z3 = np.zeros((batch_size, 1))
        self.a3 = np.zeros((batch_size, 1))

    def forward(self, X, batch_size = None):
        
        if batch_size == None:
            batch_size = self.batch_size
        # Layer 1
        self.Z1 = X@self.W1 + np.tile(self.b1, (batch_size, 1))
        self.a1 = ReLU(self.Z1)

        # Layer 2
        self.Z2 = self.a1@self.W2 + np.tile(self.b2, (batch_size, 1))
        self.a2 = ReLU(self.Z2)

        # Layer 3
        self.Z3 = self.a2@self.W3 + np.tile(self.b3, (batch_size, 1))
        self.a3 = sigmoid(self.Z3)

        return self.a3

    def criterion(self, y_hat, pred_y):
        '''
            y, pred_y shape: batch * 1
            With sigmoid, it's 2 classification problem
            Use cross-entropy as loss function
            for y = sigma(y * ln(y_pred))
        '''
        loss =  - (y_hat.T @ np.log(pred_y + self.EPS) + (1 - y_hat).T @ np.log(1 - pred_y + self.EPS)) / self.batch_size 
        return loss[0][0]
    
    def backproapgation(self, X, y_hat, pred_y):
        
        '''
            grad_A_B mean the gradient of the A, denoated on the edge A-B
            same as d Loss / d A
        '''
        grad_a3_yhat = - (y_hat / (pred_y + self.EPS) - (1 - y_hat) / (1 - pred_y + self.EPS)) # Loss: - (y_hat * np.log(pred_y) + (1 - y_hat) * np.log(1 - pred_y))
        grad_Z3_a3 = sigmoid_derivative(self.Z3, grad_a3_yhat) # sigmoid(Z3)
        grad_a2_Z3_W3 = matrix_right_mul_W_derivative(self.a2, grad_Z3_a3) # XW + b, cacualte W
        grad_a2_Z3_b3 = matrix_plus_derivative(grad_Z3_a3, self.batch_size) # XW + b, cacualte b
        grad_a2_Z3 = matrix_right_mul_X_derivative(self.W3, grad_Z3_a3) # XW + b, cacualte X

        self.W3 -= grad_a2_Z3_W3 * self.lr
        self.b3 -= grad_a2_Z3_b3 * self.lr

        grad_Z2_a2 = ReLU_derivative(self.Z2, grad_a2_Z3) # sigmoid(Z2)
        grad_a1_Z2_W2 = matrix_right_mul_W_derivative(self.a1, grad_Z2_a2) # a1*W2 + b2, cacualte W 
        grad_a1_Z2_b2 = matrix_plus_derivative(grad_Z2_a2, self.batch_size) # a1*W2 + b2, cacualte b
        grad_a1_Z2 = matrix_right_mul_X_derivative(self.W2, grad_Z2_a2) # a1*W2 + b2, cacualte X

        # print(grad_a1_Z2_W2)
        self.W2 -= grad_a1_Z2_W2 * self.lr
        self.b2 -= grad_a1_Z2_b2 * self.lr

        grad_Z1_a1 = ReLU_derivative(self.Z1, grad_a1_Z2) # sigmoid(Z1)
        grad_X_Z1_W1 = matrix_right_mul_W_derivative(X, grad_Z1_a1) # X*W1 + b1, cacualte W 
        grad_X_Z1_b1 = matrix_plus_derivative(grad_Z1_a1, self.batch_size) # X*W1 + b1, cacualte b

        self.W1 -= grad_X_Z1_W1 * self.lr
        self.b1 -= grad_X_Z1_b1 * self.lr

    def para(self):
        print(self.W1)
        print(self.b1)
        print(self.W2)
        print(self.b2)
        print(self.W3)
        print(self.b3)


    def train(self, X_train, Y_train, n_epoch, sample_size, batch_size, train_name):
        loss_check_point_x = []
        loss_check_point_y = []
        for epoch in range(n_epoch):
            for batch in range( np.ceil(sample_size / batch_size).astype(int) ):
                inputs = X_train[batch*batch_size : min(sample_size, (batch+1)*batch_size), :]
                labels = Y_train[batch*batch_size : (batch+1)*batch_size, :]
                prediction = self.forward(inputs)
                loss = self.criterion(labels, prediction)
                self.backproapgation(inputs, labels, prediction)
            if (epoch+1) % 500 == 0 or epoch==0:
                loss_check_point_x.append(epoch+1)
                loss_check_point_y.append(loss)
                print(f'epoch: {epoch+1}/{n_epoch}, loss: {loss}')
                # model.para()

        plt.clf()
        plt.figure(figsize=(6.4, 4.8))
        plt.plot(loss_check_point_x, loss_check_point_y, marker='o')
        plt.savefig(f'output/train_ReLU_{train_name}.jpg')

    def test(self, X_test, Y_test, sample_size, test_name):
        prediction = self.forward(X_test, sample_size)
        prediction_cls = (prediction > 0.5).astype(int)
        eval = np.equal(prediction_cls, Y_test)
        accuracy = np.sum(eval) / sample_size
        print(f'test, sample_size: {sample_size}, accuracy: {accuracy * 100}%')

        plt.clf()
        fig, axes = plt.subplots(1,2, figsize=(16,6))
        axes[0].set_title('ground truth')
        axes[1].set_title('prediction')
        for i in range(X_test.shape[0]):
            if Y_test[i] == 0:
                axes[0].scatter(X_test[i,0], X_test[i,1], color='r')
            elif Y_test[i] == 1:
                axes[0].scatter(X_test[i,0], X_test[i,1], color='g', marker='^')

            if prediction_cls[i] == 0:
                axes[1].scatter(X_test[i,0], X_test[i,1], color='r')
            elif prediction_cls[i] == 1:
                axes[1].scatter(X_test[i,0], X_test[i,1], color='g', marker='^')

        plt.savefig(f'output/test_ReLU_{test_name}_scatter.jpg')


def XOR_easy():
    sample_size = 21
    X_data, Y_data = generate_XOR_easy(10)
    batch_size = 3
    n_epoch = 10000
    model = SimpleNN(batch_size, [10,10])
    model.train(X_data, Y_data, n_epoch, sample_size, batch_size, 'XOR_easy')
    model.test(X_data, Y_data, sample_size, 'XOR_easy')

def liner():
    sample_size = 100
    X_data, Y_data = generate_liner(100)
    batch_size = 4
    n_epoch = 10000
    model = SimpleNN(batch_size, [10,10])
    model.train(X_data, Y_data, n_epoch, sample_size, batch_size, 'liner_data')
    model.test(X_data, Y_data, sample_size, 'liner_data')


if __name__ == '__main__':

    liner()
    XOR_easy()
    pass


