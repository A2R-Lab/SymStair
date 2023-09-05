import numpy as np
from matplotlib import pyplot
import copy
np.set_printoptions(precision=4, suppress=True, linewidth = 100)

from robotPlant import DoubleIntegratorDynamics, PendulumDynamics, \
					   CartPoleDynamics, URDFDynamics, RobotPlant
from costFunc import QuadraticCost
from kktSystemTools import KKTSystemTools

def computeConditionNumberPrecon(S,Pinv):
	# OB on pend fails chol so need to revert to eigenvalues
	# 	L = np.linalg.cholesky(Pinv)
	# 	cond = np.linalg.cond(np.matmul(np.matmul(L.transpose(),S), L))
	eigs = np.abs(np.linalg.eigvals(np.matmul(Pinv,S)))
	max_eig = np.max(eigs)
	min_eig = np.min(eigs)
	cond = max_eig/min_eig
	return cond

def computeConditionNumbers(kktTools, x, u, xs, N):
	S, _ = kktTools.formSchurSystem(x, u, xs, N)
	S = -S
	cond = np.linalg.cond(S)
	
	Pinv_J = kktTools.computePreconditioner(S, "J")
	cond_J = computeConditionNumberPrecon(S,Pinv_J)

	Pinv_BJ = kktTools.computePreconditioner(S, "BJ")
	cond_BJ = computeConditionNumberPrecon(S,Pinv_BJ)

	Pinv_OB = kktTools.computePreconditioner(S, "OB")
	cond_OB = computeConditionNumberPrecon(S,Pinv_OB)

	Pinv_SA = kktTools.computePreconditioner(S, "SA")
	cond_SA = computeConditionNumberPrecon(S,Pinv_SA)

	Pinv_SS = kktTools.computePreconditioner(S, "SS")
	cond_SS = computeConditionNumberPrecon(S,Pinv_SS)

	return (cond, cond_J, cond_BJ, cond_OB, cond_SA, cond_SS)

def computeEigsPrecon(S,Pinv):
	# OB on pend fails chol so need to revert to the try except BUT
	# since they are similar matricies this shoudl actually have no impact
	# on the eigenvalues -- just on the symmetry of the resulting matrix!
	try:
		L = np.linalg.cholesky(Pinv)
		eigs = np.linalg.eigvals(np.matmul(np.matmul(L.transpose(),S), L))
	except:
		eigs = np.linalg.eigvals(np.matmul(Pinv,S))
	return eigs

def computeEigs(kktTools, x, u, xs, N):
	S, _ = kktTools.formSchurSystem(x, u, xs, N)
	S = -S
	eigs = np.linalg.eigvals(S)
	
	Pinv_J = kktTools.computePreconditioner(S, "J")
	eigs_J = computeEigsPrecon(S,Pinv_J)

	Pinv_BJ = kktTools.computePreconditioner(S, "BJ")
	eigs_BJ = computeEigsPrecon(S,Pinv_BJ)

	Pinv_OB = kktTools.computePreconditioner(S, "OB")
	eigs_OB = computeEigsPrecon(S,Pinv_OB)

	Pinv_SA = kktTools.computePreconditioner(S, "SA")
	eigs_SA = computeEigsPrecon(S,Pinv_SA)

	Pinv_SS = kktTools.computePreconditioner(S, "SS")
	eigs_SS = computeEigsPrecon(S,Pinv_SS)

	return (eigs, eigs_J, eigs_BJ, eigs_OB, eigs_SA, eigs_SS)

def computePCGIters(kktTools, x, u, xs, N):
	iters =    kktTools.solveKKTSystem_Schur_PCG(x, u, xs, N, \
							preconditioner_type = "0", RETURN_ITERS = True)
	iters_J =  kktTools.solveKKTSystem_Schur_PCG(x, u, xs, N, \
							preconditioner_type = "J", RETURN_ITERS = True)
	iters_BJ = kktTools.solveKKTSystem_Schur_PCG(x, u, xs, N, \
							preconditioner_type = "BJ", RETURN_ITERS = True)
	iters_OB = kktTools.solveKKTSystem_Schur_PCG(x, u, xs, N, \
							preconditioner_type = "OB", RETURN_ITERS = True)
	iters_SA = kktTools.solveKKTSystem_Schur_PCG(x, u, xs, N, \
							preconditioner_type = "SA", RETURN_ITERS = True)
	iters_SS = kktTools.solveKKTSystem_Schur_PCG(x, u, xs, N, \
							preconditioner_type = "SS", RETURN_ITERS = True)

	return (iters, iters_J, iters_BJ, iters_OB, iters_SA, iters_SS)

def test_DI():
	plant = RobotPlant(robot_type = "DI")
	nq = plant.get_num_pos()
	nv = plant.get_num_vel()
	nx = nq + nv
	nu = plant.get_num_cntrl()

	N = 10
	x = np.zeros((nx,N))
	x[0,:] = 1
	u = np.zeros((nu,N-1))
	xs = copy.deepcopy(x[:,0])

	Q = np.diag([0.01, 0.01])
	QF = np.diag([10.0, 10.0])
	R = np.diag([0.0001])
	xg = np.array([0,0])
	cost = QuadraticCost(Q,QF,R,xg)

	kkt = KKTSystemTools(plant,cost)

	conds = computeConditionNumbers(kkt, x, u, xs, N)
	iters = computePCGIters(kkt, x, u, xs, N)
	eigs = computeEigs(kkt, x, u, xs, N)
	return conds, iters, eigs

def test_PEND():
	plant = RobotPlant(robot_type = "PEND")
	nq = plant.get_num_pos()
	nv = plant.get_num_vel()
	nx = nq + nv
	nu = plant.get_num_cntrl()

	N = 20
	x = np.zeros((nx,N))
	u = np.zeros((nu,N-1))
	xs = copy.deepcopy(x[:,0])

	Q = np.diag([1.0,1.0])
	QF = np.diag([100.0,100.0])
	R = np.diag([0.1])
	xg = np.array([3.14159,0])
	cost = QuadraticCost(Q,QF,R,xg)

	kkt = KKTSystemTools(plant,cost)

	conds = computeConditionNumbers(kkt, x, u, xs, N)
	iters = computePCGIters(kkt, x, u, xs, N)
	eigs = computeEigs(kkt, x, u, xs, N)
	return conds, iters, eigs

def test_CART():
	N = 40
	dt = 0.5/N
	plant = RobotPlant(robot_type = "CART", options = {"dt" : dt})
	nq = plant.get_num_pos()
	nv = plant.get_num_vel()
	nx = nq + nv
	nu = plant.get_num_cntrl()

	x = np.zeros((nx,N))
	u = np.zeros((nu,N-1))
	xs = copy.deepcopy(x[:,0])

	Q = np.diag([1.0, 1.0, 0.1, 0.1])
	QF = np.diag([1000.0, 1000.0, 1000.0, 1000.0])
	R = np.diag([0.001])
	xg = np.array([0,3.14,0,0])
	cost = QuadraticCost(Q,QF,R,xg)

	kkt = KKTSystemTools(plant,cost)

	conds = computeConditionNumbers(kkt, x, u, xs, N)
	iters = computePCGIters(kkt, x, u, xs, N)
	eigs = computeEigs(kkt, x, u, xs, N)
	return conds, iters, eigs


def test_IIWA():
	N = 20
	dt = 0.5/N
	plant = RobotPlant(robot_type = "URDF", options = {"path_to_urdf" : "iiwa.urdf", "dt" : dt})
	nq = plant.get_num_pos()
	nv = plant.get_num_vel()
	nx = nq + nv
	nu = plant.get_num_cntrl()

	PI = 3.14159
	xs = np.array([-0.5*PI,0.25*PI,0.167*PI,-0.167*PI,0.125*PI,0.167*PI,0.5*PI, 0,0,0,0,0,0,0])
	x = np.zeros((nx,N))
	for k in range(N):
		x[:,k] = copy.deepcopy(xs)
	u = np.zeros((nu,N-1))

	Q = np.eye(nx)
	Q[0:nq,0:nq] *= 1.0
	Q[nq:,nq:] *= 0.1
	QF = 100.0 * np.eye(nx) # 10
	R = 0.0001 * np.eye(nu) # 0.1
	xg = np.array([0,0,-0.25*PI,0,0.25*PI,0.5*PI,0, 0,0,0,0,0,0,0])
	cost = QuadraticCost(Q,QF,R,xg)

	kkt = KKTSystemTools(plant,cost)

	conds = computeConditionNumbers(kkt, x, u, xs, N)
	iters = computePCGIters(kkt, x, u, xs, N)
	eigs = computeEigs(kkt, x, u, xs, N)
	return conds, iters, eigs

def compute_hists(eigs_list):
	eigs, eigs_J, eigs_BJ, eigs_OB, eigs_SA, eigs_SS = eigs_list
	# all_eigs = eigs + eigs_J + eigs_BJ + eigs_OB + eigs_SA + eigs_SS
	all_eigs = eigs_SA + eigs_SS
	my_min = min(all_eigs)
	my_max = max(all_eigs)

	bins = np.linspace(my_min,my_max,1000)
	# hist_J, _, _ = pyplot.hist(eigs_J, bins, alpha=0.5, label='J', density = True, cumulative = True)
	# hist_OB, _, _ = pyplot.hist(eigs_OB, bins, alpha=0.5, label='OB', density = True, cumulative = True)
	# hist_BJ, _, _ = pyplot.hist(eigs_BJ, bins, alpha=0.5, label='BJ', density = True, cumulative = True)
	hist_SS, _, _ = pyplot.hist(eigs_SS, bins, alpha=0.5, label='SS', density = True, cumulative = True)
	hist_SA, _, _ = pyplot.hist(eigs_SA, bins, alpha=0.5, label='SA', density = True, cumulative = True)
	# pyplot.legend(loc='upper right')
	# pyplot.xscale("linear")
	# pyplot.xlim(xmax=1.25)
	# pyplot.xlim(xmin=0.000001)
	# pyplot.show()

	return bins, hist_SA, hist_SS


def test_case_studies():
	conds, iters, eigs = test_DI()
	print("Double Integrator")
	print("Conds: ", conds)
	print("Iters: ", iters)
	print("Eigs: ")
	for vector in eigs:
		print (vector.tolist())
	bins, hist_SA, hist_SS = compute_hists(eigs)
	print("bins: ", bins.tolist())
	print("Hist SA: ", hist_SA.tolist())
	print("Hist SS: ", hist_SS.tolist())

	conds, iters, eigs = test_PEND()
	print("Pendulum")
	print("Conds: ", conds)
	print("Iters: ", iters)
	print("Eigs: ")
	for vector in eigs:
		print (vector.tolist())
	bins, hist_SA, hist_SS = compute_hists(eigs)
	print("bins: ", bins.tolist())
	print("Hist SA: ", hist_SA.tolist())
	print("Hist SS: ", hist_SS.tolist())

	conds, iters, eigs = test_CART()
	print("Cart Pole")
	print("Conds: ", conds)
	print("Iters: ", iters)
	print("Eigs: ")
	for vector in eigs:
		print (vector.tolist())
	bins, hist_SA, hist_SS = compute_hists(eigs)
	print("bins: ", bins.tolist())
	print("Hist SA: ", hist_SA.tolist())
	print("Hist SS: ", hist_SS.tolist())

	conds, iters, eigs = test_IIWA()
	print("IIWA")
	print("Conds: ", conds)
	print("Iters: ", iters)
	print("Eigs: ")
	for vector in eigs:
		print (vector.tolist())
	bins, hist_SA, hist_SS = compute_hists(eigs)
	print("bins: ", bins.tolist())
	print("Hist SA: ", hist_SA.tolist())
	print("Hist SS: ", hist_SS.tolist())

test_case_studies()