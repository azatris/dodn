__author__ = 'Azatris'

import layer


class Network(object):
    def __init__(self, architecture, initial_weight_magnitude):
        self.layers = [
            layer.Sigmoid(neurons, inputs_per_neuron, initial_weight_magnitude)
            for neurons, inputs_per_neuron
            in zip(architecture[1:], architecture[:-2])
        ]
        self.layers.append(layer.Softmax(architecture[-1], architecture[-2], initial_weight_magnitude))
        
        for l in xrange(0, len(self.layers)):
            print "Created weight matrtix in layer %d with shape %s"%(l, self.layers[l].weights.shape)
            print "Created biases vector in layer %d with shape %s"%(l, self.layers[l].biases.shape)

    def feed_forward(self, x, return_all=False):
        act=[None]*(len(self.layers)+1) #inputs + layer activations
        act[0]=x
        for l in xrange(0,len(self.layers)):
            act[l+1] = self.layers[l].feed_forward(act[l])
        if return_all is True:
            return act
        return act[-1]
    
    def feed_backward(self, error, a):
        assert len(a)==(len(self.layers)+1) #a should have feats at 0th index
        deltas = [None]*len(self.layers)
        # some notation identieties, z=Wx+b, h=f(z), hence dh/dz depends on non-linearity and z is just linear transform
        # to update params you need dh/dz (deltas) and prev layer activations h(l-1)
        # but you need also pass the signal through linear part, referred as eh below
        #deltas[-1] = error #note, here we make a shortcut, that the error w.r.t cross-entropy and 1ofK targets is actually a correct gradient
        eh = error #this one is supposed to keep dh/dz*W.T, above deltas are only dh/dz
        for l in xrange(len(self.layers)-1,-1,-1):
            deltas[l], eh = self.layers[l].feed_backward(eh, a[l+1])
        return deltas