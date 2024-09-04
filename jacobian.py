import sympy as sp
import powerclasses as pc
from powerclasses import rect
import numpy as np


def Jacobian(Kvector, Uvector, Y, V, D):
  JacobV = np.empty(0)
  for qty in Kvector.data:
    i = qty.bus
    
    #Generating the buses numbers for variables in the equation of each Known Value
    j_indices = [idx + 1 for idx in range(len(Kvector.data)) ]
  
    
    # Dynamically defining variables of corresponding buses
    Vi = sp.symbols(f"V{i}")
    Vj = [sp.symbols(f"V{j}") for j in j_indices]
    Di = sp.symbols(f"D{i}")
    Dj = [sp.symbols(f"D{j}") for j in j_indices]
    
    #Funtion definition for Generating function for Known vector quantity Pi or Qi
    def generate_function(Vi, Vj, Di, Dj):
      summation_term = sum(Vj[k] * np.abs(Y[i-1, k]) * sp.cos(np.angle(Y[i-1, k]) + Dj[k] - Di) for k in range(len(Vj)))
      full = Vi * summation_term
      return full
      
    #Generating the function of the Known Quantity
    f = generate_function(Vi, Vj, Di, Dj)
    
    
    #Differentiating the function with respect to 
    for qty in Uvector.data:       
      var = sp.symbols(f"{qty.type}{qty.bus}")  # determining the independent variables for differentiation for each iteration
      diff = sp.diff(f, var)
      
      # Generating values substitution dictionary
      subs = {}
      
      for item in range(len(V)):      #Values of vector V
        sub = {f'V{item + 1}': V[item]}
        subs.update(sub)
        
      for item in range(len(D)):      #Values of vector Delta
        sub = {f'D{item + 1}': D[item]}
        subs.update(sub)
        
      eval_val = diff.subs(subs)
      JacobV = np.append(JacobV, eval_val)
      n = len(V)
      # JacobM = JacobV.reshape(n, n)
      # print(eval_val)
  return JacobV
      
      

#Test Values
specified = pc.Kvector()
P2 = pc.Qty("P", 2, 0.5)
P3 = pc.Qty("P", 3, -1.5)
Q2 = pc.Qty("Q", 2, 1.0)
specified.push(P2)
specified.push(P3)
specified.push(Q2)

inital = pc.Uvector()
D2 = pc.Qty("D", 2, 0)
D3 = pc.Qty("D", 3, 0)
V2 = pc.Qty("V", 2, 0)
inital.push(D2)
inital.push(D3)
inital.push(V2)

V_matrix = np.array([1.04, 1, 1.04]) 

D_matrix =np.array([0, 0, 0])

Y_matrix = np.array([[rect(24.23, -75.95), rect(12.13, 104.04), rect(12.13, 104.04)],
                     [rect(12.13, 104.04), rect(24.23, -75.95), rect(12.13, 104.04)],
                     [rect(12.13, 104.04), rect(12.13, 104.04), rect(24.23, -75.95)]],
                    dtype=np.complex64)

value = Jacobian(specified, inital, Y_matrix, V_matrix, D_matrix)

print(value)