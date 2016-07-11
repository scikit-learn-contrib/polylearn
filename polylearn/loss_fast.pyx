from libc.math cimport log, exp

cdef class LossFunction:

     cdef double loss(self, double p, double y):
         raise NotImplementedError()

     cdef double dloss(self, double p, double y):
         raise NotImplementedError()


cdef class Squared(LossFunction):
    """Squared loss: L(p, y) = 0.5 * (y - p)²"""

    def __init__(self):
        self.mu = 1

    cdef double loss(self, double p, double y):
        return 0.5 * (p - y) ** 2

    cdef double dloss(self, double p, double y):
        return p - y


cdef class Logistic(LossFunction):
    """Logistic loss: L(p, y) = log(1 + exp(-yp))"""

    def __init__(self):
        self.mu = 0.25

    cdef double loss(self, double p, double y):
        cdef double z = p * y
        # log(1 + exp(-z))
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z
        return log(1.0 + exp(-z))

    cdef double dloss(self, double p, double y):
        cdef double z = p * y
        #cdef double tau = 1 / (1 + exp(-z))
        #return y * (tau - 1)
        if z > 18.0:
            return -y * exp(-z)
        if z < -18.0:
            return -y
        return -y / (exp(z) + 1.0)


cdef class SquaredHinge(LossFunction):
    """Squared hinge loss: L(p, y) = max(1 - yp, 0)²"""

    def __init__(self):
        self.mu = 2

    cdef double loss(self, double p, double y):
        cdef double z = 1 - p * y
        if z > 0:
            return z * z
        return 0.0

    cdef double dloss(self, double p, double y):
        cdef double z = 1 - p * y
        if z > 0:
            return -2 * y * z
        return 0.0
