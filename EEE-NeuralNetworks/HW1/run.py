import numpy as np


### Activation Functions ###
class Sigmoid:
    @staticmethod
    def forward(z):
        return 1./(1 + np.exp(-z))
    
    @staticmethod
    def derivative(a):
        return a * (1-a)

class Tanh:
    @staticmethod
    def forward(z):
        return np.tanh(z)
    
    @staticmethod
    def derivative(a):
        return 1 - a**2
    
class ReLu:
    @staticmethod
    def p(z):
        return np.maximum(z, 0)

    @staticmethod
    def derivative(a):
        return np.where(a > 0, 1, 0)
    
def mini_batch(xt, yt, mini_batch_size, shuffle = None):
        """
        It is a generator functions that generates&yields mini batches

        """
        # Shuffle dataset
        def create_permutation(x, y):
            perm = np.random.permutation(len(x))
            return x[perm], y[perm]

        if shuffle is not None:
            x_shuffle, y_shuffle = create_permutation(xt, yt)
        else:
            x_shuffle, y_shuffle = xt[:], yt[:] 

        # m: number of examples
        m = yt.shape[1]

        number_of_mini_batches = int(m/mini_batch_size)

        for k in range(number_of_mini_batches): 
            mini_batch_X = x_shuffle[:, k*mini_batch_size : (k+1)*mini_batch_size]
            mini_batch_Y = y_shuffle[:, k*mini_batch_size : (k+1)*mini_batch_size]
            yield (mini_batch_X, mini_batch_Y)

        if (m % number_of_mini_batches) != 0:
            mini_batch_X = x_shuffle[:, number_of_mini_batches*mini_batch_size : m]
            mini_batch_Y = y_shuffle[:, number_of_mini_batches*mini_batch_size : m]
            yield (mini_batch_X, mini_batch_Y)


def MSE_cost_onehot_encode(y_pred, y_true, m, l2_reg_cost):
        """ The cost function is sum of meansquared errors \\
            normalized by the number of samples
        """
        assert y_true.shape == y_pred.shape
        assert (y_true.shape[1] == m)
        
        y_diff = y_pred - y_true
        
        cost = (1/(y_true.shape[1])) * np.sum(np.square(y_diff))
        
        # ! if lambda_l2 is set to 0 regularization cost will be 0
        total_cost = cost + l2_reg_cost
        
        assert isinstance(total_cost, float)
        return total_cost


class NN:
    def __init__(self, N, output_size):
            # Initialize training data and labels
        self.x = None
        self.y = None
        
            # Initialize neural network layer size
        self.N = N
        self.output_size = output_size
        self.m = None  # number of training samples
        
            # Initialize training batch size
        self.batch_size = None 
        
        # Initialize learning rate
        self.lr = None
            
            # Initialize lambda for L2 regularization
        # if not used it will be set to 0 at NN.fit
        self.lambda_l2 = None 
        
            # Initialize weights and biases.
                # Gaussian distribution
        self.is_parameters_initialized = False
        
        # Layer 1
        self.W1 = None
        self.b1 = None
        
        # Layer 2
        self.W2 = None
        self.b2 = None
        
            # Initialize activation functions
        self.act1 = None
        self.act2 = None
    
    def init_parameters(self, input_size):
        np.random.seed(42)

        # Gaussian distribution
        mean = 0
        std = 0.01 
        
        self.mean = mean
        self.std = std
        
        # Layer 1
        self.W1 = np.random.normal(mean, std, (self.N, input_size))
        self.b1 = np.zeros((N, 1))
        
        # Layer 2
        self.W2 = np.random.normal(0, std, (self.output_size, self.N))
        self.b2 = np.zeros((self.output_size, 1))
        
        self.is_parameters_initialized = True
    
    def forward_pass(self, xb):
        # Layer1
        z1 = np.dot(self.W1, xb) + self.b1
        a1 = self.act1.forward(z1)
        
        # Layer2
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.act2.forward(z2)
        
        return a1, a2
    
    def backward_propagation(self, xb, a1, a2, yb):
        # Layer2
        
        # MSE Derivative 2*(a-y)
        loss_deriv = 2 * (a2-yb)
        dZ_2 = np.multiply(loss_deriv, self.act2.derivative(a2))
        dW2 = (
                 (1.0/self.m) * np.dot(dZ_2, a1.T) 
                 + (self.lambda_l2/self.m) * self.W2
              )
        db2 = (1.0/self.m) * np.sum(dZ_2, keepdims=True)
        
        # Layer1
        dZ_1 = np.multiply(np.dot(self.W2.T, dZ_2), self.act1.derivative(a1))
        dW1 = (
                (1.0/self.m) * np.dot(dZ_1, xb.T) 
                + (self.lambda_l2/self.m) * self.W1
            )
        db1 = (1.0/self.m) * np.sum(dZ_1, keepdims=True)
        
            # Update Weights and Biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


    def fit(self, X, Y, act1, act2, lr, epochs, lambda_l2 = 0, batch_size=32):
        """
        N : hidden layer(layer1) size
        """
        self.x = X.copy()
        self.y = Y.copy()
        

        # X.shape = (feature_size, # of samples)
        input_size = self.x.shape[0]
              
        # assing activations
        self.act1 = act1
        self.act2 = act2
        
        # assign learning rate
        self.lr = lr
        
        # assign reg. value. if not set it will be 0 which
        # simulates the condition where there is no L2 Regularization
        self.lambda_l2 = lambda_l2
        
        # assing weights. if initialized when fit called training can continue
        if not self.is_parameters_initialized:
            self.init_parameters(input_size)
            self.is_parameters_initialized = True
            print(f"\nWeights are initialized with Gaussian Distribution mean:{self.mean}, std:{self.std}")
            print(f"\t{self.W1.shape=} \n\t{self.b1.shape=} \n\t{self.W2.shape=} \n\t{self.b2.shape=}")
            print("--------------------")
            print("--------------------\n")


        self.batch_size = batch_size
        print("Training Starts...\n")
        for epoch in range(epochs):
            batch_no = 0
            total_cost = 0
            print(f"\nEpoch {epoch} ->")

            for xb, yb in mini_batch(X, Y, mini_batch_size = self.batch_size):
                
                # m : number of samples
                m = xb.shape[1]    
                self.m = m

                # Forward Propagation                
                a1, a2 = self.forward_pass(xb)
            
                batch_preds = self.prediction(xb, categorical=False)
            
                # ! if self.lambda_l2 is set to 0 
                # regularization cost will be 0
                l2_reg_cost = 0
                if self.lambda_l2 != 0:
                    l2_reg_cost = (
                            (self.lambda_l2/(2*self.m)) 
                            * (np.sum(np.square(self.W1)) 
                               + np.sum(np.square(self.W2)))
                        )

                batch_cost = MSE_cost_onehot_encode(batch_preds, yb, self.m, l2_reg_cost)
                print(f"\t\tbatch no: {batch_no} :\t batch loss: {batch_cost}")
                batch_no += 1
                total_cost += batch_cost
                
                # Backward Propagation
                self.backward_propagation(xb, a1, a2, yb)
            
            print(f"\tAverage Epoch Loss: {total_cost/batch_no}")
            print("\tTrain Acc: ", nn1.get_accuracy(x_train, y_train)) 
            print("\tTest Acc : ", nn1.get_accuracy(x_test, y_test), "\n")



    def prediction(self, X, categorical=False):
        # Layer1
        z1 = np.dot(self.W1, X) + self.b1
        a1 = self.act1.forward(z1)
        
        # Layer2
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.act2.forward(z2)
        
        cat = None
        if categorical:
            cat = np.argmax(a2, axis=0)
            return cat
        return a2
    
    def get_accuracy(self, X, Y):
        preds = self.prediction(X, categorical=True).reshape(1,-1)
        #print(X.shape, preds.shape, Y.shape)
        return np.mean(preds == Y)
    
    
def create_onehot(Y, num_class, m):
    y_onehot = np.zeros((m, num_class))
    for i,label in enumerate(Y[0], start=0):
        #print(f"{i=}, {label=}\n")
        y_onehot[i][label] = 1
    return y_onehot.T

def create_onehot_minus1(Y, num_class, m):
    y_onehot = np.full((m, num_class), -1)
    for i,label in enumerate(Y[0], start=0):
        #print(f"{i=}, {label=}\n")
        y_onehot[i][label] = 1
    return y_onehot.T

# input type (feature, sample)
#input size 28 * 28 = 784

x_train = np.load("mnist/mnist_data_x_train.npy")
y_train = np.load("mnist/mnist_data_y_train.npy")
x_test = np.load("mnist/mnist_data_x_test.npy")
y_test = np.load("mnist/mnist_data_y_test.npy")

x_train = x_train.reshape(60000, 28*28).T
x_test = x_test.reshape(10000, 28*28).T
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)

# Normalize the input
x_train = x_train / 255.0
x_test = x_test / 255.0


        # Onehot encode the Labels
case1 = True
case2 = False
# CASE1
if case1 is True:
    y_train_onehot = create_onehot_minus1(y_train, 10, y_train.shape[1])
    y_test_onehot = create_onehot_minus1(y_test, 10, y_test.shape[1])

# CASE2
if case2 is True:
    y_train_onehot = create_onehot(y_train, 10, y_train.shape[1])
    y_test_onehot = create_onehot(y_test, 10, y_test.shape[1])

# Print the shapes of the NumPy arrays
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

N = 300
nn1 = NN(N, 10)
nn1.fit(x_train, y_train_onehot, 
        act1=Tanh, act2=Tanh,
        lr=0.01, epochs=5, batch_size=10000)

# free memory
import gc
gc.collect()
