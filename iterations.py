def newton_raphson_iteration(F, J, x0, tol=1e-6, max_iter=100):

    x = np.array(x0, dtype=float)
    
    for i in range(max_iter):
        F_val = F(x)
        J_val = J(x)
        
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