import numpy as np
import copy

class KKTSystemTools:
    # Assumes RBDObj is of type RBDReference
    # Assuems costObj has the methods value, gradient, hessian that take in (xk,uk) for running and (xk,None) for final
    def __init__(self, robot_obj_in, cost_obj_in):
        self.robot = robot_obj_in
        self.cost = cost_obj_in

    def update_cost(self, cost_obj_in):
        self.cost = cost_obj_in

    def formKKTSystem(self, x, u, xs, N):
        N = N - 1 # to have N total states need N - 1 states with controls and then final state so offsetting N here
        nq = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        nu = self.robot.get_num_cntrl()
        nx = nq + nv
        n = nx + nu
        G = np.zeros((n*N + nx, n*N + nx))
        g = np.zeros((n*N + nx, 1))
        C = np.zeros((nx*(N+1), n*N + nx))
        c = np.zeros((nx*(N+1), 1))

        for k in range(N):
            G[k*n:(k+1)*n, k*n:(k+1)*n] = self.cost.hessian(x[:,k], u[:,k], k)
            g[k*n:(k+1)*n, 0] = self.cost.gradient(x[:,k], u[:,k], k)

            Ak, Bk = self.robot.integrator_gradient(x[:,k], u[:,k])
            C[(k+1)*nx:(k+2)*nx, k*n:(k+1)*n+nx] = np.hstack((-Ak, -Bk, np.eye(nx)))
            xkp1 = self.robot.integrator(x[:,k], u[:,k])
            c[(k+1)*nx:(k+2)*nx, 0] = x[:,k+1] - xkp1

        G[N*n:N*n+nx, N*n:N*n+nx] = self.cost.hessian(x[:,N], k = N-1)
        g[N*n:N*n+nx, 0] = self.cost.gradient(x[:,N], k = N-1)
        C[0:nx, 0:nx] = np.eye(nx)
        c[0:nx, 0] = x[:,0] - xs

        return G, g, C, c

    def formSchurSystem(self, x, u, xs, N, rho = 0):
        nq = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        nu = self.robot.get_num_cntrl()
        nx = nq + nv
        
        G, g, C, c = self.formKKTSystem(x, u, xs, N)
        BR = np.zeros((N*nx,N*nx))

        if rho != 0:
            G += rho * np.eye(G.shape[0])

        invG = np.linalg.inv(G)
        S = BR - np.matmul(C, np.matmul(invG, C.transpose()))
        gamma = c - np.matmul(C, np.matmul(invG, g))

        return S, gamma

    def solveKKTSystem(self, x, u, xs, N, rho = 0):
        nq = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        nu = self.robot.get_num_cntrl()
        nx = nq + nv
        n = nx + nu
        
        G, g, C, c = self.formKKTSystem(x, u, xs, N)
        BR = np.zeros((N*nx,N*nx))

        if rho != 0:
            G += rho * np.eye(G.shape[0])

        KKT = np.hstack((np.vstack((G, C)),np.vstack((C.transpose(), BR))))
        kkt = np.vstack((g, c))

        dxul = np.linalg.solve(KKT,kkt)

        return dxul

    def solveKKTSystem_Schur(self, x, u, xs, N, rho = 0):
        S, gamma = self.formSchurSystem(x, u, xs, N, rho)

        l = np.linalg.solve(S, gamma)

        gCl = g - np.matmul(C.transpose(), l)
        dxu = np.matmul(invG, gCl)

        dxul = np.vstack((dxu,l))

        return dxul

    def computePreconditioner(self, A, preconditioner_type = "0"):
        if preconditioner_type == "0": # null aka identity
            return np.identity(A.shape[0])

        if preconditioner_type == "J": # Jacobi aka Diagonal
            return np.linalg.inv(np.diag(np.diag(A)))

        elif preconditioner_type == "BJ": # Block-Jacobi
            nq = self.robot.get_num_pos()
            nv = self.robot.get_num_vel()
            nu = self.robot.get_num_cntrl()
            nx = nq + nv # blocksize

            n_blocks = int(A.shape[0] / nx)
            Pinv = np.zeros(A.shape)

            for k in range(n_blocks):
                rc_k = k*nx
                rc_kp1 = rc_k + nx
                Pinv[rc_k:rc_kp1, rc_k:rc_kp1] = np.linalg.inv(A[rc_k:rc_kp1, rc_k:rc_kp1])

            return Pinv

        elif preconditioner_type == "OB": # Overlapping Block (for blocktridiagonal of blocksize nq+nv)
            nq = self.robot.get_num_pos()
            nv = self.robot.get_num_vel()
            nu = self.robot.get_num_cntrl()
            nx = nq + nv # blocksize

            n_blocks = int(A.shape[0] / nx)

            # first compute the overlapping block inverse
            Pinv = np.zeros(A.shape)
            for k in range(n_blocks):
                rc_k = k*nx
                rc_kp1 = rc_k + nx
                rc_km1 = rc_k - nx
                Pinv[rc_km1:rc_kp1, rc_km1:rc_kp1] += np.linalg.inv(A[rc_km1:rc_kp1, rc_km1:rc_kp1])
            
            # then divide values by 2 or 3
            for k in range(n_blocks):
                rc_k = k*nx
                rc_kp1 = rc_k + nx
                rc_km1 = rc_k - nx
                # for diagonal first and last block left alone
                #     one in from the ends divide by 2 else divide by 3
                # for off diagonal first and last column left along
                #     else divide by 2
                if k > 0 and k < n_blocks - 1:
                    if k > 1 and k < n_blocks - 2:
                        Pinv[rc_k:rc_kp1, rc_k:rc_kp1] /= 3
                    else:
                        Pinv[rc_k:rc_kp1, rc_k:rc_kp1] *= 0.5
                    if k > 1:
                        Pinv[  rc_k:rc_kp1, rc_km1:rc_k  ] *= 0.5
                        Pinv[rc_km1:rc_k,     rc_k:rc_kp1] *= 0.5

            return Pinv

        # Stair (for blocktridiagonal of blocksize nq+nv)
        elif preconditioner_type == "S" or preconditioner_type == "SS" or preconditioner_type == "SA": 
            nq = self.robot.get_num_pos()
            nv = self.robot.get_num_vel()
            nu = self.robot.get_num_cntrl()
            nx = nq + nv # blocksize

            n_blocks = int(A.shape[0] / nx)
            Pinv = np.zeros(A.shape)
            # compute stair inverse
            for k in range(n_blocks):
                # compute the diagonal term
                Pinv[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = np.linalg.inv(A[k*nx:(k+1)*nx, k*nx:(k+1)*nx])
                if np.mod(k, 2): # odd block includes off diag terms
                    # Pinv_left_of_diag_k = -Pinv_diag_k * A_left_of_diag_k * -Pinv_diag_km1
                    Pinv[k*nx:(k+1)*nx, (k-1)*nx:k*nx] = -np.matmul(Pinv[k*nx:(k+1)*nx, k*nx:(k+1)*nx], \
                                                          np.matmul(A[k*nx:(k+1)*nx, (k-1)*nx:k*nx], \
                                                                    Pinv[(k-1)*nx:k*nx, (k-1)*nx:k*nx]))
                elif k > 0: # compute the off diag term for previous odd block (if it exists)
                    # Pinv_right_of_diag_km1 = -Pinv_diag_km1 * A_right_of_diag_km1 * -Pinv_diag_k
                    Pinv[(k-1)*nx:k*nx, k*nx:(k+1)*nx] = -np.matmul(Pinv[(k-1)*nx:k*nx, (k-1)*nx:k*nx], \
                                                          np.matmul(A[(k-1)*nx:k*nx, k*nx:(k+1)*nx], \
                                                                    Pinv[k*nx:(k+1)*nx, k*nx:(k+1)*nx]))
            # add left and right stair
            if preconditioner_type == "SA":
                Pinv = (Pinv.transpose() + Pinv)/2

            # make symmetric
            if preconditioner_type == "SS":
                for k in range(n_blocks):
                    if np.mod(k, 2): # copy from odd blocks
                        # always copy up the left to previous right
                        Pinv[(k-1)*nx:k*nx, k*nx:(k+1)*nx] = Pinv[k*nx:(k+1)*nx, (k-1)*nx:k*nx].transpose()
                        # if not last block copy right to next left
                        if k < n_blocks - 1:
                            Pinv[(k+1)*nx:(k+2)*nx, k*nx:(k+1)*nx] = Pinv[k*nx:(k+1)*nx, (k+1)*nx:(k+2)*nx].transpose()
            return Pinv

           


        else:
            print("Invalid preconditioner options are [J : Jacobi, BJ: Block-Jacobi, AB : Alternating Block, OB : Overlapping Block, S : Stair,  SA: Additive Stair, SS: Symmetric Stair]")
            exit()

    def PCG(self, A, b, guess, Pinv, exit_tolerance = 1e-6, max_iter = 100, DEBUG_MODE = False, RETURN_TRACE = False):
        # initialize
        x = np.reshape(guess, (guess.shape[0],1))
        r = b - np.matmul(A, x)
        
        r_tilde = np.matmul(Pinv, r)
        p = r_tilde
        nu = np.matmul(r.transpose(), r_tilde)
        if DEBUG_MODE:
            print("Initial nu[", nu, "]")
        if RETURN_TRACE:
            trace = nu[0].tolist()
            trace2 = [np.linalg.norm(b - np.matmul(A, x))]
        # loop
        for iteration in range(max_iter):
            Ap = np.matmul(A, p)
            alpha = nu / np.matmul(p.transpose(), Ap)
            r -= alpha * Ap
            x += alpha * p
            
            r_tilde = np.matmul(Pinv, r)
            nu_prime = np.matmul(r.transpose(), r_tilde)
            if RETURN_TRACE:
                trace.append(nu_prime.tolist()[0][0])
                trace2.append(np.linalg.norm(b - np.matmul(A, x)))
            
            if abs(nu_prime) < exit_tolerance:
                if DEBUG_MODE:
                    print("Exiting with err[", abs(nu_prime), "]")
                break
            else:
                if DEBUG_MODE:
                    print("Iter[", iteration, "] with err[", abs(nu_prime), "]")
            
            beta = nu_prime / nu
            p = r_tilde + beta * p
            nu = nu_prime

        if RETURN_TRACE:
            trace = list(map(abs,trace))
            return x, trace, trace2
        else:
            return x

    def solveKKTSystem_Schur_PCG(self, x, u, xs, N, rho = 0, preconditioner_type = "0", exit_tolerance = 1e-6, max_iter = 1000, RETURN_TRACE = False, RETURN_ITERS = False):
        S, gamma = self.formSchurSystem(x, u, xs, N, rho)
        S = -S

        Pinv = self.computePreconditioner(S, preconditioner_type)

        guess = np.zeros((x.shape[0]*N))

        if RETURN_TRACE or RETURN_ITERS:
            l, trace, trace2 = self.PCG(S, gamma, guess, Pinv, exit_tolerance, max_iter, False, True)
            if RETURN_ITERS:
                return len(trace)
        else:
            l = self.PCG(S, gamma, guess, Pinv, exit_tolerance, max_iter, False, RETURN_TRACE)

        gCl = g - np.matmul(C.transpose(), l)
        dxu = np.matmul(invG, gCl)

        dxul = np.vstack((dxu,l))

        if RETURN_TRACE:
            return dxul, trace, trace2
        else:
            return dxul