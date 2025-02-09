#THEANO_FLAGS='device=gpu' python2 classify-cnn.py
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import pandas as pd
from sklearn.cross_validation import train_test_split
import datetime
import time



srng = RandomStreams()
labels = ["0","1","2","3","4","5","6","7","8","9"]


def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx


df = pd.read_csv('train.csv', skiprows=1, header=None)

X_kaggel = df.iloc[:, 1:].values
X_kaggel = X_kaggel/255.0

y_kaggel = df.iloc[:, 0].values
y_kaggel = one_hot(y_kaggel,10)


#trX, teX, trY, teY = train_test_split(X_kaggel,y_kaggel, test_size=0.01, random_state=1)
trX = X_kaggel
trY = y_kaggel


df_validate = pd.read_csv('test.csv', skiprows=1, header=None)

X_validate = df_validate.iloc[:, :].values
X_validate = X_validate/255.0



trX = trX.reshape(-1, 1, 28, 28)
#teX = teX.reshape(-1, 1, 28, 28)

#X_validate = X_validate.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 525))
w_o = init_weights((525, 10))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(120):
    ts = time.time()
    print str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    print i
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    ts = time.time()
    print str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    #print np.mean(np.argmax(teY, axis=1) == predict(teX))
    if i == 2:
        break
'''
with open("predictions-cnn-theano-gpu-100train-n120-wfour525.txt", "w") as f:
    f.write('\"ImageId\",\"Label\"\n')
    for i in range(0,len(X_validate)):
        sample = X_validate[i,:].reshape(-1, 1, 28, 28)
        prediction = predict(sample)
        line = str(i+1) + "," + '\"' + str(labels[int(prediction[0])]) + '\"' + "\n"
        f.write(line)
'''
