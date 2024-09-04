from powerclasses import rect
import powerclasses as pc
from NR import NR 
from NR import eval 
import numpy as np

Y = np.array([[rect(60, -90), rect(40, 90), rect(20, 90)], 
              [rect(40, 90), rect(60, -90), rect(20, 90)], 
              [rect(20, 90), rect(20, 90), rect(40, -90)]], 
             dtype=np.complex64)


spec = pc.Kvector()
p2 = pc.Qty('p', 2, 4.0)
p3 = pc.Qty('p', 3, -5.0)
q3 = pc.Qty('q', 3, -4.0)
spec.push(p2)
spec.push(p3)
spec.push(q3)



init2 = pc.Uvector()
d2 = pc.Qty("D", 2, 0.0)
d3 = pc.Qty("D", 3, 0.0)
v3 = pc.Qty("V", 3, 1)
init2.push(d2)
init2.push(d3) 
init2.push(v3)

v = np.array([1.0, 1.05, 1.0])
d = np.array([0.0, 0.0, 0.0])
# v = np.array([1.0, 0.9698996629203881, 1.01])
# d = np.array([0.0, -0.04627935346375924, 0.04029593586468951])

mismatch = pc.MismatchV(spec, d, v, Y)
print(mismatch)
jacob = pc.Jacobian(spec, init2, d, v, Y)
iterationi = eval(init2, jacob, mismatch) 
print(iterationi)


# pc.update(iterationi, d, v) 

# nr5 =   NR(5, spec, init, d, v, Y)

