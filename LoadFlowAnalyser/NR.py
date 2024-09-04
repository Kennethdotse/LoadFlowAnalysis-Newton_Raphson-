import powerclasses as pc
import numpy as np

def eval(initial, Jacob, mismatch):   
  # converting vector objects into numerical arrays 
  mm = pc.arrify(mismatch)
  init = pc.arrify(initial)
  
  invJacob = np.linalg.inv(Jacob)
  product = np.matmul(invJacob, mm)  
  sum = init + product
  result = pc.vectorfy(sum, initial)

  return result



def NR(iters, spec, init, D, V, Y ):
  for n in range(iters):
    mismatch = pc.MismatchV(spec, D, V, Y)
    jacob = pc.Jacobian(spec, init, D, V, Y)
    iterationi = eval(init, jacob, mismatch) 
    print(f'Iteration:{n+1}{iterationi}')
    pc.update(iterationi, D, V) 
  
  # print(f'{iters}: iterations performed')
  
  return iterationi
  