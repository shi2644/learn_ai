import functions.active_fuc as act

class neuron:
    def __init__(self, inputs, fuc):
        if type(inputs) is not list:
            self.inputs = []
            self.inputs.append(inputs)
        else:
            self.inputs = inputs
        self.fuc = fuc
        self.output = self.active()

    def active(self):
        return self.fuc(self.inputs)