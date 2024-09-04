import sympy as sp
import powerclasses as pc
from powerclasses import rect
import numpy as np
import math
import cmath



P_spec = np.array([
    [200],
    [300]
])

Q_spec = np.array([
    [50],
    [10]
])

bus_types = ['PQ', 'PV']


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


def power_mismatch(P_spec, Q_spec, V, delta, Y_bus):
    """
    Function for power mismatch calculations
    """
    n = len(V)  # Getting the number of buses
    bus_tracker = n  # To know the number of buses anywhere in the code

    P_calc = np.empty((n, 1))
    Q_calc = np.empty((n, 1))
    dP = np.empty((n, 1))
    dQ = np.empty((n, 1))
    
    max_length = max(len(P_spec), len(Q_spec), len(V), len(delta), len(Y_bus))

    if max_length > len(P_spec):
        P_spec = np.pad(P_spec, (0, max_length - len(P_spec)))
    if max_length > len(Q_spec):
        Q_spec = np.pad(Q_spec, (0, max_length - len(Q_spec)))
    if max_length > len(V):
        V = np.pad(V, (0, max_length - len(V)))
    if max_length > len(delta):
        delta = np.pad(delta, (0, max_length - len(delta)))
    if max_length > len(Y_bus):
        Y_bus = np.pad(Y_bus, ((0, max_length - len(Y_bus)), (0, 0)))

    
    if not (len(P_spec) == len(Q_spec) == len(V) == len(delta) == len(Y_bus)):
        raise ValueError("All input arrays must be the same length")

    assert len(P_spec) == len(Q_spec) == len(V) == len(delta) == len(Y_bus)

    for i in range(max_length):
        if (P_spec[i] == 0).all() and (Q_spec[i] == 0).all() and (V[i] != 0).any():
            # This is a slack bus
            dP[i, 0] = 0
            dQ[i, 0] = 0
        elif (P_spec[i] != 0).all() and (Q_spec[i] == 0).all() and (V[i] != 0).all():
            # This is a PV bus
            dQ[i, 0] = 0
            for j in range(n):
                P_calc[i] += V[j].item() * abs(Y_bus[i, j].item()) * math.cos(cmath.phase(Y_bus[i, j].item()) + delta[j].item() - delta[i].item())
            dP[i, 0] = (P_spec[i].item()) - (V[i].item() * P_calc[i].item())
        elif (P_spec[i] != 0).all() and (Q_spec[i] != 0).all() and (V[i] != 0).all():
            # This is a PQ bus
            for j in range(n):
                P_calc[i] += V[j].item() * abs(Y_bus[i, j].item()) * math.cos(cmath.phase(Y_bus[i, j].item()) + delta[j].item() - delta[i].item())
                Q_calc[i] += -1 * (V[j].item() * abs(Y_bus[i, j].item()) * math.sin(cmath.phase(Y_bus[i, j].item()) + delta[j].item() - delta[i].item()))
            dP[i, 0] = (P_spec[i].item()) - (V[i].item() * P_calc[i].item())
            dQ[i, 0] = Q_spec[i].item() - (V[i].item() * Q_calc[i].item())

    #print("\nKindly check the values inputted. \nYour buses must either be PQ, PV, and one slack bus\n")

    dF = np.vstack((dP, dQ))  # Joins the two matrices to become an n by 1 matrix
    d_del_and_v = np.concatenate((delta, V), axis=0)  # Joining the delta and V to be one matrix
    d_del_and_v = d_del_and_v.reshape(-1, 1)  # Ensuring that the vector is n by 1

    return dF 

mismatch = power_mismatch(P_spec, Q_spec, V_matrix, D_matrix, Y_matrix)
print(mismatch)

def Jacobian(Kvector, Uvector, Y, V, D):
    n = len(V) + len(D)
    JacobV = np.zeros((n,n))   #Initialising the Jacobian matrix

    # Define symbolic variables
    V_symbols = [sp.symbols(f"V{i}") for i in range(1, n+1)]
    D_symbols = [sp.symbols(f"D{i}") for i in range(1, n+1)]
    
    regularization_term = 1e-6  # Choose a suitable regularization term
    np.fill_diagonal(JacobV, JacobV.diagonal() + regularization_term)


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
      JacobV[i] = eval_val
      JacobM = JacobV.reshape(n, n)
    
    return JacobM
      
      
value = Jacobian(specified, inital, Y_matrix, V_matrix, D_matrix)

print(value)


def newton_raphson_iteration(x0, tol=1e-6, max_iter=100):
    
    '''
    F = mismatch array
    J =  Jacobian array
    x0 = initial guess for volatages and angles
    tol = tolerance for convergence 
    max_iter = maximum number of iterations
    '''
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        F_val = power_mismatch(P_spec, Q_spec, x[:len(V_matrix)], x[len(V_matrix):], Y_matrix)
        J_val = Jacobian(specified, inital, Y_matrix, x[:len(V_matrix)], x[len(V_matrix):])

        #print(F_val.shape)  # Should be (n, 1)
        #print(J_val.shape)

        # Check if the Jacobian is singular
        if np.linalg.matrix_rank(J_val) < J_val.shape[0]:
            raise ValueError("Jacobian is singular, cannot proceed with Newton-Raphson iteration.")

        delta_x = np.linalg.solve(J_val, -F_val)

        # Update the solution
        x = x + delta_x

        # Check for convergence
        if np.linalg.norm(delta_x) < tol:
            print(f"Converged in {i+1} iterations.")
            return x

    raise ValueError("Newton-Raphson did not converge within the maximum number of iterations.")

final = newton_raphson_iteration(np.concatenate((V_matrix, D_matrix)), tol=1e-6, max_iter=100)
