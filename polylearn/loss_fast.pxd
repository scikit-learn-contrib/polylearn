cdef class LossFunction:

     cdef double mu
     cdef double loss(self, double p, double y)
     cdef double dloss(self, double p, double y)
