import numpy as np

Ybus = np.array([
    [1, 2],
    [4, 5]
])

V = np.array([1, 1]) 

P_spec = np.array([
    [200],
    [300]
])

Q_spec = np.array([
    [50],
    [10]
])

bus_types = ['PQ', 'PV']

def calculate_mismatch(Ybus, V, P_spec, Q_spec, bus_types):
    delta_P = np.zeros(len(V))
    delta_Q = np.zeros(len(V))
    
    for i in range(len(V)):
        P_calc = 0
        Q_calc = 0
        
        for j in range(len(V)):
            P_calc += abs(V[i]) * abs(V[j]) * abs(Ybus[i, j]) * np.cos(np.angle(V[i]) - np.angle(V[j]) - np.angle(Ybus[i, j]))
            Q_calc += abs(V[i]) * abs(V[j]) * abs(Ybus[i, j]) * np.sin(np.angle(V[i]) - np.angle(V[j]) - np.angle(Ybus[i, j]))
        
        print(P_calc)
        print(Q_calc)
        if bus_types[i] == 'PQ' or bus_types[i] == 'PV':
            delta_P[i] = P_spec[i] - P_calc
        
        if bus_types[i] == 'PQ':
            delta_Q[i] = Q_spec[i] - Q_calc[i]
            print(delta_P)
            
    mismatch = np.concatenate((delta_P, delta_Q))
    
    return mismatch

print(calculate_mismatch(Ybus, V, P_spec, Q_spec, bus_types))


