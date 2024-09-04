import powerclasses as pc
from powerclasses import rect
import numpy as np

#Test Values

#This is how to create and populate the specified vector of knowns
#First Create the Quantites that belong in this vector and then add them to the vector
specified = pc.Kvector()
P2 = pc.Qty("P", 2, 0.5)
P3 = pc.Qty("P", 3, -1.5)
Q2 = pc.Qty("Q", 2, 1.0)
specified.push(P2)
specified.push(P3)
specified.push(Q2)


V_matrix = np.array([1.04, 1, 1.04]) 

D_matrix =np.array([0, 0, 0])

#Example use of the rect function to add polar values directly to the admittance matrix
Y_matrix = np.array([[rect(24.23, -75.95), rect(12.13, 104.04), rect(12.13, 104.04)],
                     [rect(12.13, 104.04), rect(24.23, -75.95), rect(12.13, 104.04)],
                     [rect(12.13, 104.04), rect(12.13, 104.04), rect(24.23, -75.95)]],
                    dtype=np.complex64)

#NOTE: TEST VALUES WERE TAKEN FROM THE EXAMPLE 6 QUESTION IN THE SLIDES: UNIT ONE, SLIDE 86

#Test  
pc.MismatchV(specified, D_matrix, V_matrix, Y_matrix)