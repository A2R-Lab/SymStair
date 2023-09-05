import numpy as np

class QuadraticCost:
	def __init__(self, Q_in, QF_in, R_in, xg_in, QF_start = None):
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in

	def get_currQ(self, u = None, k = None):
		use_QF = isinstance(u,type(None))
		currQ = self.QF if use_QF else self.Q
		return currQ

	def value(self, x, u = None, k = None):
		delta_x = x - self.xg
		currQ = self.get_currQ(u,k)
		cost = 0.5*np.matmul(delta_x.transpose(),np.matmul(currQ,delta_x))
		if not isinstance(u, type(None)):
			cost += 0.5*np.matmul(u.transpose(),np.matmul(self.R,u))
		return cost

	def gradient(self, x, u = None, k = None):
		delta_x = x - self.xg
		currQ = self.get_currQ(u,k)
		top = np.matmul(delta_x.transpose(),currQ)
		if u is None:
			return top
		else:
			bottom = np.matmul(u.transpose(),self.R)
			return np.hstack((top,bottom))

	def hessian(self, x, u = None, k = None):
		nx = self.Q.shape[0]
		nu = self.R.shape[0]
		currQ = self.get_currQ(u,k)
		if u is None:
			return currQ
		else:
			top = np.hstack((currQ,np.zeros((nx,nu))))
			bottom = np.hstack((np.zeros((nu,nx)),self.R))
			return np.vstack((top,bottom))