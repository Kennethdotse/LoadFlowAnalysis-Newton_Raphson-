import numpy as np

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

    for i in range(n):
        if P_spec[i] == 0 and Q_spec[i] == 0 and V[i] != 0:
            # This is a slack bus
            dP[i, 0] = 0
            dQ[i, 0] = 0
        elif P_spec[i] != 0 and Q_spec[i] == 0 and V[i] != 0:
            # This is a PV bus
            dQ[i, 0] = 0
            for j in range(n):
                P_calc[i] += V[j].item() * abs(Y_bus[i, j].item()) * math.cos(math.phase(Y_bus[i, j].item()) + delta[j].item() - delta[i].item())
            dP[i, 0] = (P_spec[i].item()) - (V[i].item() * P_calc[i].item())
        elif P_spec[i] != 0 and Q_spec[i] != 0 and V[i] != 0:
            # This is a PQ bus
            for j in range(n):
                P_calc[i] += V[j].item() * abs(Y_bus[i, j].item()) * math.cos(math.phase(Y_bus[i, j].item()) + delta[j].item() - delta[i].item())
                Q_calc[i] += -1 * (V[j].item() * abs(Y_bus[i, j].item()) * math.sin(math.phase(Y_bus[i, j].item()) + delta[j].item() - delta[i].item()))
            dP[i, 0] = (P_spec[i].item()) - (V[i].item() * P_calc[i].item())
            dQ[i, 0] = Q_spec[i].item() - (V[i].item() * Q_calc[i].item())

    print("\nKindly check the values inputted. \nYour buses must either be PQ, PV, and one slack bus\n")

    dF = np.vstack((dP, dQ))  # Joins the two matrices to become an n by 1 matrix
    d_del_and_v = np.concatenate((delta, V), axis=0)  # Joining the delta and V to be one matrix
    d_del_and_v = d_del_and_v.reshape(-1, 1)  # Ensuring that the vector is n by 1

    return dF, d_del_and_v, bus_tracker


