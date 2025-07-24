import functions.active_fuc as act


class neuron:
    def __init__(self, inputs, fuc):
        self.inputs = inputs
        self.fuc = fuc
        self.output = self.active()

    def active(self):
        return self.fuc(self.inputs)


if __name__ == "__main__":
    i = [1, 2, 3,4,10.1]
    neuron1 = neuron(i, act.sigmoid)
    print(neuron1.output)
