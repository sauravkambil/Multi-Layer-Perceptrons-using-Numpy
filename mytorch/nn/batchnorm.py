import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N =  self.Z.shape[0] # TODO
        ones = np.ones((np.shape(self.Z)[0],1))
        self.M = 1/self.N * np.sum(self.Z, axis= 0)
        self.M = self.M.reshape(1,len(self.M))
        self.V = 1/self.N * np.sum(np.square(self.Z - self.M), axis= 0)  # TODO
        self.V = self.V.reshape(1,len(self.V))  # TODO

        if eval == False:
            # training mode
            self.NZ = (self.Z - ones@self.M) / (ones @np.sqrt(self.V + self.eps))  # TODO
            self.BZ = np.multiply(self.NZ, ones@self.BW) + ones@self.Bb  # TODO

            self.running_M = self.alpha*self.running_M + (1-self.alpha)*self.M   # TODO
            self.running_V = self.alpha*self.running_V + (1-self.alpha)* self.V    # TODO
        else:
            # inference mode
            self.NZ = (self.Z - self.running_M) / np.sqrt(self.running_V + self.eps)  # TODO
            self.BZ = np.multiply(self.NZ, ones@self.BW) + ones@self.Bb  # TODO

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ)  # TODO
        self.dLdBb = np.sum(dLdBZ)  # TODO

        dLdNZ = dLdBZ*self.BW  # TODO
        dLdV = (-1/2)*np.sum(dLdNZ*(self.Z-self.M)*((self.V + self.eps)**(-3/2)), axis= 0) # TODO
        dLdM = np.sum(dLdNZ*((-(self.V + self.eps)**(-1/2))-((1/2)*(self.Z - self.M))*((self.V + self.eps)**(-3/2))*((-2/self.N)*np.sum(self.Z - self.M, axis= 0))), axis= 0)  # TODO

        dLdZ = dLdNZ*((self.V + self.eps)**(-1/2)) + dLdV*((2/self.N)*(self.Z - self.M)) + (1/self.N)*dLdM  # TODO

        return dLdZ
