from . import units


class Layer:
    def __init__(self,d):
        self.thickness = d # cm

class AbsorbingLayer(Layer):
    def __init__(self,d,mu):
        super().__init__(d)
        self.abscoe = mu # 1/cm

class LayerStack:
    def __init__(self,z0):
        self.position_of_top_layer = z0 # cm
        self.layers = []


    def add_layer(self,layer):
        self.layers.append(layer)


