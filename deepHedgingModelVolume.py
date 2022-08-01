from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, \
                Lambda, Add, Dot, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal, Zeros, he_uniform, TruncatedNormal
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

  
class OnePeriodModel(tf.keras.layers.Layer):
    def __init__(self, d = None, m = None, useBatchNorm = None, \
        weightInit = "he_uniform", biasInit=None, \
        denseActivation = "relu", outputActivation = "linear", period = None):
        super().__init__(name = "delta_" + str(period))
        self.d = d
        self.m = m
        self.useBatchNorm = useBatchNorm
        self.denseActivation = denseActivation
        self.outputActivation = outputActivation
        self.weightInit = weightInit
        self.biasInit = biasInit
        
        #hidden layers
        self.intermediateDense = [None for _ in range(d)]
        #batch normalization layers
        self.intermediateBN = [None for _ in range(d)]
        
        for i in range(d):
           self.intermediateDense[i] = Dense(self.m,    
                        kernel_initializer=self.weightInit,
                        bias_initializer=self.biasInit,
                        use_bias=(not self.useBatchNorm))
           if self.useBatchNorm:
               self.intermediateBN[i] = BatchNormalization()#momentum = 0.99, trainable=True)
           
        self.outputDense = Dense(4, 
                      kernel_initializer=self.weightInit,
                      bias_initializer = self.biasInit,
                      use_bias=True)
        
    def call(self, input):
        #runs the input through the strategy layer
            for i in range(self.d):
                if i == 0:
                    output = self.intermediateDense[i](input)
                else:
                    output = self.intermediateDense[i](output)                  
                    
                if self.useBatchNorm:
     			    # Batch normalization.
                    output = self.intermediateBN[i](output, training=True)
                    
                if self.denseActivation == "leaky_relu":
                    output = LeakyReLU()(output)
                else:
                    output = Activation(self.denseActivation)(output)
             
            output = self.outputDense(output)
    					 
            if self.outputActivation == "leaky_relu":
                output = LeakyReLU()(output)
            else:
                output = Activation(self.outputActivation)(output)
            # elif self.outputActivation == "sigmoid" or self.outputActivation == "tanh" or \
            #     self.outputActivation == "hard_sigmoid":
            #     output = Activation(self.outputActivation)(output)
            
            return output

def deepModelVolume(N=None, d=2, m=8, eps=0.0, useBatchNorm=False, \
            weightInit='random_normal', biasInit='random_normal', \
            denseActivation='tanh', outputActivation='linear'):
    #params to add: initialwealth, r, costType
    
    #input params
    price = Input(shape=(2,))
    informationSet = Input(shape=(2,))
    inputs = [price, informationSet]
    
    for j in range(N+1):
        if j<N:
            if j==0:
                #we start with 0 assets (setting it 0 in other ways doesnt work )
                strategy = Lambda(lambda x: 0.0*x)(price)  
                nonMarkov = Lambda(lambda x: 0.0*x)(price) #information that we hope wille encompass knowledge about previous periods
                
            helper = Concatenate()([informationSet, strategy, nonMarkov])
            onePeriodModel = OnePeriodModel(d=d, m=m, useBatchNorm=useBatchNorm,\
                            weightInit=weightInit, biasInit=biasInit,\
                            denseActivation=denseActivation, outputActivation=outputActivation,\
                            period=j)
            strategyHelper = onePeriodModel(helper)
            
            #calculate difference in strategy
            if j==0:
                deltaStrategy = strategyHelper[:,:2]
                tradingVolume = Lambda(lambda x: 0.0*x)(strategyHelper[:,:2])
                tradingVolume = Dot(axes=1)([tradingVolume, price])
            else:
                deltaStrategy = Subtract()([strategyHelper[:,:2], strategy])
            
            #use difference in strategy to calculate trading cost and purchasing costs
            absoluteChanges = Lambda(lambda x : K.abs(x))(deltaStrategy)
            tradingCosts = Dot(axes=1)([absoluteChanges, price])
            tradingVolume = Add()([tradingVolume, tradingCosts])
            tradingCosts = Lambda(lambda x : eps*x)(tradingCosts) 
            
            if j==0:
                wealth =  Lambda(lambda x : - x)(tradingCosts)
            else:
                wealth = Subtract()([wealth, tradingCosts])
            
            mult = Dot(axes=1)([deltaStrategy, price])
            wealth = Subtract()([wealth, mult])
    
            #next iteration of inputs
            price = Input(shape=(2,))
            informationSet = Input(shape=(2,))
            strategy = strategyHelper[:,:2]
            nonMarkov = strategyHelper[:,2:]
            
            if j!=N-1:
                inputs += [price, informationSet]
            else:
                inputs += [price]
        else:
            
            absoluteChanges = Lambda(lambda x : K.abs(x))(strategy)
            tradingCosts = Dot(axes=1)([absoluteChanges, price])
            tradingVolume = Add()([tradingVolume,tradingCosts])
            tradingCosts = Lambda(lambda x : eps*x)(tradingCosts) 
            
            wealth = Subtract()([wealth, tradingCosts])
            
            mult = Dot(axes=1)([strategy, price])
            wealth = Add()([wealth, mult])
            
            payoff = Input(shape=(1,))
            inputs +=[payoff]
            
            wealth = Subtract()([wealth, payoff])
    
    return Model(inputs=inputs, outputs=[wealth,tradingVolume])
    
    
    
    
    
    








    
    
    
    
    
    








