from neuron import *
import functions.active_fuc as act

if __name__ == "__main__":
    i = [2.7,3.9,3.6,4.2]
    neuron1_1 = neuron(i, act.sigmoid)
    neuron1_2 = neuron(i, act.sigmoid)
    neuron1_3 = neuron(i, act.sigmoid)
    lare2 = [neuron1_1.output, neuron1_2.output, neuron1_3.output]
    neuron2 = neuron(lare2, act.sigmoid)
    print(neuron2.output)
