from powerclasses import rect
import powerclasses as pc
from NR import NR 
from NR import eval 
import numpy as np

Y = np.array([[rect(14, -90), rect(10, 90), rect(4, 90)], 
              [rect(10, 90), rect(15, -90), rect(5, 90)], 
              [rect(4, 90), rect(5, 90), rect(9, -90)]], 
             dtype=np.complex64)


spec = pc.Kvector()
p2 = pc.Qty('p', 2, -0.9)
p3 = pc.Qty('p', 3, 0.6)
q2 = pc.Qty('q', 2, -0.5)
spec.push(p2)
spec.push(p3)
spec.push(q2)

# init = pc.Uvector()
# d2 = pc.Qty("D", 2, 0.0)
# d3 = pc.Qty("D", 3, 0.0)
# v2 = pc.Qty("V", 2, 1.0)
# init.push(d2)
# init.push(d3) 
# init.push(v2)

# v = np.array([1.0, 1.0, 1.01])
# d = np.array([0.0, 0.0, 0.0])

# mismatch = pc.MismatchV(spec, d, v, Y)
# jacob = pc.Jacobian(spec, init, d, v, Y)
# iterationi = eval(init, jacob, mismatch) 
# print(iterationi)


init2 = pc.Uvector()
d2 = pc.Qty("D", 2, -0.04627935346375924)
d3 = pc.Qty("D", 3, 0.04029593586468951)
v2 = pc.Qty("V", 2, 0.9698996629203881)
init2.push(d2)
init2.push(d3) 
init2.push(v2)

v = np.array([1.0, 1.0, 1.01])
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

