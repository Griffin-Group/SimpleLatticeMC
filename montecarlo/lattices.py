
import matplotlib.pyplot as plt
import numpy.random as random
random.seed(89) #899
import numpy as np 

####################################################################################
# 3D hexagonal lattice of either up/down/empty sites, N*M total sites.
####################################################################################
class RigidHexagonalLattice3D:

	####################################################################################
	# init functions
	####################################################################################
	def __init__(self, N, M, P, a0, c, occ_config=None):
		
		self.N, self.M, self.P = N, M, P
		self.a0 = a0
		self.a1 = np.array([a0, 0, 0])
		self.a2 = np.array([-a0/2, np.sqrt(3)/2*a0 , 0])
		self.a3 = np.array([0,0,c])
		self.neighbor_table = np.zeros((N*M*P, N*M*P))
		self.set_neighbor_table()
		self.Hsigmai = [None]

		if occ_config is None: 
			self.random_initialize_occ()
		else: self.occ_config = occ_config.flatten()

	def size(self): return self.N * self.M * self.P

	def initialzie_occ(self, occ_config, verbose=False):
		self.occ_config = occ_config
		self.energy = None

	def occ_size(self): return int(np.sum(self.occ_config))

	def setup_occ_energetics(self, coupling_function, image_rep=[1,1,1], chempot=0):
		self.construct_effective_occH(coupling_function, image_rep)
		self.chempot = chempot # no effect if kawasaki
		self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.energy -= self.chempot * np.sum(self.occ_config)

	def random_initialize_occ(self, x=None, verbose=False):
		if x == None: #random itercalation density too
			occ_config = np.zeros((self.N,self.M,self.P))
			for n in range(self.N):
				for m in range(self.M):
					for p in range(self.P):
						occ_config[n,m,p] = random.randint(0, 2)
						if occ_config[n,m,p] == 1:
							if verbose: print('filled site at {},{},{}'.format(site, n, m, p))
			self.occ_config = occ_config.flatten()
		else: #random with specified itercalation density 
			occ_config = np.zeros((self.N,self.M,self.P))
			occ_config = occ_config.flatten()
			for p in range(self.P):
				N_filled_sites_this_layer = int(round(x[p]*self.N*self.M, 0))
				Nfulllayer = 0
				while Nfulllayer < N_filled_sites_this_layer:
					n = random.randint(0, self.N)
					m = random.randint(0, self.M)
					site = self.nm_to_site_index(n,m,p)
					if occ_config[site] == 0:
						occ_config[site] = 1
						Nfulllayer += 1
						if verbose: print('filled site at {},{},{}'.format(n, m, p))
			self.occ_config = occ_config
		self.energy = None
		self.Hsigmai = [None]

	def construct_effective_occH(self, coupling_function, image_rep): # deals with PBC
		# energy is then spin_config.T dot effmagH dot spin_config
		self.effoccH = np.zeros((self.N*self.M*self.P,self.N*self.M*self.P))
		whole_sc_a1 = self.N * self.a1
		whole_sc_a2 = self.M * self.a2
		whole_sc_a3 = self.P * self.a3
		for i in range(self.N*self.M*self.P):
			for j in range(self.N*self.M*self.P):
				rij = self.get_intersite_R(i, j)
				for ii in range(-image_rep[0], image_rep[0]+1):
					for jj in range(-image_rep[1], image_rep[1]+1):
						for kk in range(-image_rep[2], image_rep[2]+1):
							rij_image = rij + ii*whole_sc_a1 + jj*whole_sc_a2 + kk*whole_sc_a3
							self.effoccH[i,j] += coupling_function((rij_image[0]**2 + rij_image[1]**2 + rij_image[2]**2)**0.5)
		return self.effoccH

	def visualize(self, ax=None, path=None):
		if ax == None: 
			f, ax = plt.subplots(1,self.P)

		for layer in range(self.P):

			xy = np.zeros((self.N,self.M,2))
			rgb = np.zeros((self.N,self.M,3))
			
			for n in range(self.N):
				for m in range(self.M):
					vec = n * self.a1 + m * self.a2 
					i = self.nm_to_site_index(n,m,layer)	
					xy[n,m,0] = vec[0]
					xy[n,m,1] = vec[1]
					if self.occ_config[i] == 0: rgb[n,m,:] = [0,0,0]
					elif self.occ_config[i] == 1: rgb[n,m,:] = [1,0,0]

			xy = xy.reshape(self.N*self.M, 2)
			rgb = rgb.reshape(self.N*self.M, 3)

			if self.P > 1:
				ax[layer].scatter(xy[:,0], xy[:,1], color=rgb) 
			else:
				ax.scatter(xy[:,0], xy[:,1], color=rgb) 

		if path == None:
			plt.show()
		else:
			plt.savefig(path, dpi=300)

	####################################################################################
	# composite index handling
	####################################################################################
	def site_index_to_nm(self, i):
		nm = np.unravel_index(i, (self.N, self.M, self.P))
		return nm[0], nm[1], nm[2]
	
	def nm_to_site_index(self,n,m,p): return np.ravel_multi_index((n,m,p), (self.N, self.M, self.P))

	def get_intersite_R(self, site1, site2):
		n1, m1, p1 = self.site_index_to_nm(site1)
		n2, m2, p2 = self.site_index_to_nm(site2)
		delR = (n2-n1) * self.a1 + (m2-m1) * self.a2 + (p2-p1) * self.a3
		return delR

	def set_neighbor_table(self):
		for i in range(self.size()):
			for j in range(self.size()):
				if i > j and self.compute_are_neighbors(i,j):
					self.neighbor_table[i,j] = 1
					self.neighbor_table[j,i] = 1

	def compute_are_neighbors(self, site1, site2):
		rij = self.get_intersite_R(site1, site2)
		whole_sc_a1 = self.N * self.a1
		whole_sc_a2 = self.M * self.a2
		whole_sc_a3 = self.P * self.a3
		for ii in [-1,0,1]:
			for jj in [-1,0,1]:
				for kk in [-1,0,1]:
					rij_image = rij + ii*whole_sc_a1 + jj*whole_sc_a2 + kk*whole_sc_a3
					rij_image_len = (rij_image[0]**2 + rij_image[1]**2 + rij_image[2]**2)**0.5
					if round(rij_image_len,2) == round(self.a0,2): return True
		return False

	def list_nearest_neighbors(self, site): return [i for i in range(self.size()) if (self.neighbor_table[site,i] == 1)] 

	def list_viable_hop_sites(self, site):
		nn = self.list_nearest_neighbors(site)
		site_occ = self.occ_config[site] 
		viable_nn = [i for i in nn if site_occ != self.occ_config[i]]
		return viable_nn

	def propose_hop(self, site1, site2):
		self.active_site1 = site1
		self.active_site2 = site2 
		return True

	def accept_hop(self):
		self.energy = self.proposed_energy
		self.site_swap(self.active_site1, self.active_site2)
		if self.occ_config[self.active_site1] == 1: m, k = self.active_site1, self.active_site2
		else:  k, m = self.active_site1, self.active_site2
		self.proposed_energy = None
		self.active_site1 = None
		self.active_site2 = None
		compute = self.Hsigmai + self.effoccH[m,:] - self.effoccH[k,:]
		#expect  = (self.effoccH @ self.occ_config).flatten();
		#for i in range(len(expect)): assert(np.round(expect[i], 5) == np.round(compute[i], 5))
		self.Hsigmai = compute

	def site_swap(self, s1, s2):
		temp1 = self.occ_config[s1].copy()
		temp2 = self.occ_config[s2].copy()
		self.occ_config[s1] = temp2
		self.occ_config[s2] = temp1

	def reject_hop(self):
		self.proposed_energy = None
		self.active_site1 = None
		self.active_site2 = None

	def get_dE_hop(self):
		#e2 = self.get_dE_hop_rank1upd()
		#e1 = self.get_dE_hop_matrixprod()
		#print(e1, e2); assert(np.round(e1, 5) == np.round(e2, 5))
		return self.get_dE_hop_rank1upd()

	def get_dE_hop_matrixprod(self):
		if self.energy == None: self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.site_swap(self.active_site1, self.active_site2)
		self.proposed_energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.site_swap(self.active_site1, self.active_site2)
		return self.proposed_energy - self.energy 

	def get_dE_hop_rank1upd(self): #
		if self.energy == None: 
			self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config

		if self.Hsigmai[0] == None: 
			self.Hsigmai = (self.effoccH @ self.occ_config).flatten();

		if self.occ_config[self.active_site1] == 1: 
			m, k = self.active_site1, self.active_site2
		else:  
			k, m = self.active_site1, self.active_site2

		dE = self.effoccH[m,m] + self.effoccH[k,k] - 2*self.effoccH[m,k] 
		dE += 2*self.Hsigmai[k] - 2*self.Hsigmai[m]
		self.proposed_energy = self.energy + dE
		return dE

	def interc_density(self): return np.mean(self.occ_config)

	def get_energy(self): return self.energy

	def sublattice_2x2_order_param(self, layer=0):
		assert(self.N%2 == 0 and self.M%2 == 0)
		sl_occ = [0,0,0,0] # 4 different 2x2 sublattices
		for i in range(self.N*self.M*self.P):
			n,m,p = self.site_index_to_nm(i)
			if p == layer:
				if n%2 == 0 and m%2 == 0: sl_occ[0] += self.occ_config[i]
				if n%2 == 0 and m%2 == 1: sl_occ[1] += self.occ_config[i]
				if n%2 == 1 and m%2 == 0: sl_occ[2] += self.occ_config[i]
				if n%2 == 1 and m%2 == 1: sl_occ[3] += self.occ_config[i]
		return (np.max(sl_occ)) - (np.min(sl_occ))
		#return (np.max(sl_occ))/np.sum(sl_occ) # portion in a single sublattice 

	def sublattice_oop_order_param(self):
		total_op = 0
		num = 0
		for n in range(self.N):
			for m in range(self.M):
				columnocc = []
				for p in range(self.P):
					i = self.nm_to_site_index(n,m,p)
					columnocc.append(self.occ_config[i])
				avg_filling = np.mean(columnocc) # = 0.5 if half full etc
				if avg_filling > 0: # ignore totally empty columns
					total_op += avg_filling;
					num += 1;
		# if =1,   all intercalants in a full column
		# if =1/2, then all intercalants in a half full column
		# if =1/3, then all intercalants in a 1/3 full column	
		return total_op/num 

	def sublattice_r3xr3_order_param(self, layer=0):
		assert(self.N%3 == 0 and self.M%3 == 0)
		sl_occ = [0,0,0] # 3 different 2x2 sublattices
		for i in range(self.N*self.M*self.P):
			n,m,p = self.site_index_to_nm(i)
			if p == layer:
				if (n%3 == 0 and m%3 == 0) or (n%3 == 1 and m%3 == 2) or (n%3 == 2 and m%3 == 1): sl_occ[0] += self.occ_config[i]
				if (n%3 == 0 and m%3 == 1) or (n%3 == 1 and m%3 == 0) or (n%3 == 2 and m%3 == 2): sl_occ[1] += self.occ_config[i]
				if (n%3 == 0 and m%3 == 2) or (n%3 == 1 and m%3 == 1) or (n%3 == 2 and m%3 == 0): sl_occ[2] += self.occ_config[i]
		return (np.max(sl_occ)) - (np.min(sl_occ))
		#return (np.max(sl_occ))/np.sum(sl_occ) # portion in a single sublattice -- should be = 1 for perfect r3xr3 

####################################################################################
# 2D hexagonal lattice of either up/down/empty sites, N*M total sites.
####################################################################################
class RigidHexagonalLattice:

	####################################################################################
	# init functions
	####################################################################################
	def __init__(self, N, M, a0=1, c=None, spin_config=None, occ_config=None, superlattice=None):
		
		self.N, self.M = N, M
		self.a0 = a0
		self.a1 = np.array([a0, 0])
		self.a2 = np.array([-a0/2, np.sqrt(3)/2*a0])
		self.neighbor_table = np.zeros((N*M, N*M))
		self.set_neighbor_table()
		self.Hsigmai = [None]

		if occ_config is None: 
			if superlattice is None: 
				self.random_initialize_occ()
			else:
				self.superlattice_init_occ(period=superlattice)
		else: self.occ_config = occ_config.flatten()

		if spin_config is None: self.random_initialize_spin()
		else: self.spin_config = spin_config.flatten()
		self.occ_optimized = False

	def size(self): return self.N * self.M

	def initialzie_occ(self, occ_config, verbose=False):
		self.occ_config = occ_config
		self.energy = None

	def occ_size(self): return int(np.sum(self.occ_config))

	def modify_field(self, newfield): 
		self.field = newfield
		if self.occ_optimized:
			self.energy = np.transpose(self.pruned_spins) @ self.pruned_effmagH @ self.pruned_spins 
			self.energy -= self.field * np.sum(self.pruned_spins)
		else:
			self.energy = np.transpose(self.spin_config) @ self.effmagH @ self.spin_config 
			self.energy -= self.field * np.sum(self.spin_config)

	# gets only portion of effective ham needed so can do smaller matmult at each energy
	# calculation step. Exploits fact that spins only and not occupancies changing.
	# pls dont use when simulating an occupancy hamiltonian
	def optimize_for_occupancies(self):
		n_occ = self.occ_size()
		self.pruned_spins = np.zeros((n_occ,1))
		self.pruned_effmagH = np.zeros((n_occ,n_occ))
		# pluck out only portion of spin_config that matters
		ii = 0
		for i in range(self.N*self.M):
			if self.occ_config[i] == 1:
				self.pruned_spins[ii] = self.spin_config[i]
				ii += 1
		# pluck out only portion of effmagH that matters
		ii = 0
		for i in range(self.N*self.M):
			if self.occ_config[i] == 1: 
				jj = 0
				for j in range(self.N*self.M):
					if self.occ_config[j] == 1:
						self.pruned_effmagH[ii,jj] = self.effmagH[i,j]
						jj += 1
				ii += 1
		self.occ_optimized = True
		self.energy = None

	def setup_occ_energetics(self, coupling_function, image_rep=[1,1], chempot=0):
		self.construct_effective_occH(coupling_function, image_rep)
		self.chempot = chempot # no effect if kawasaki
		self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.energy -= self.chempot * np.sum(self.occ_config)

	def setup_mag_energetics(self, coupling_function, Hfield, image_rep=[10,10]):
		self.construct_effective_magH(coupling_function, image_rep)
		#self.optimize_for_occupancies()
		self.field = Hfield
		if self.occ_optimized:
			self.energy = np.transpose(self.pruned_spins) @ self.pruned_effmagH @ self.pruned_spins 
			self.energy -= self.field * np.sum(self.pruned_spins)
		else:
			self.energy = np.transpose(self.spin_config) @ self.effmagH @ self.spin_config 
			self.energy -= self.field * np.sum(self.spin_config)

	def random_initialize_occ(self, x=None):
		if x == None: #random itercalation density too
			occ_config = np.zeros((self.N,self.M))
			for n in range(self.N):
				for m in range(self.M):
					occ_config[n,m] = random.randint(0, 2)
			self.occ_config = occ_config.flatten()
		else: #random with specified itercalation density 
			N_filled_sites = int(round(x*self.size() , 0))
			occ_config = np.zeros((self.N*self.M,1))
			while np.sum(occ_config) < N_filled_sites:
				site = random.randint(0, self.size())
				occ_config[site] = 1
			self.occ_config = occ_config
			assert(np.sum(self.occ_config) == N_filled_sites)
		self.energy = None
		self.Hsigmai = [None]

	# only works for integers rn so 2x2, 4x4 cant get the r3xr3
	def superlattice_init_occ(self, period=2): 
		occ_config = np.zeros((self.N,self.M))
		for n in range(self.N):
			for m in range(self.M):
				if n%period == 0 and m%period == 0:
					occ_config[n,m] = 1
		self.occ_config = occ_config.flatten()
		self.energy = None

	def random_initialize_spin(self):
		spin_config = np.zeros((self.N,self.M))
		for n in range(self.N):
			for m in range(self.M):
				site = self.nm_to_site_index(n,m)
				if self.occ_config[site] == 1:
					 s = random.randint(0, 2)
					 if s == 0: s = -1
					 spin_config[n,m] = s
		self.spin_config = spin_config.flatten()
		self.energy = None

	####################################################################################
	# image_rep sets how many unit cells (of whole supercell!) go out to 
	# to account for fact that the whole simulation cell has PBC 
	# EXPENSIVE do this only once!
	# for small supercells increase image rep until it reaches its (slow) convergence
	# may need to consider an ewald sum here in future 
	####################################################################################
	def construct_effective_magH(self, coupling_function, image_rep): # deals with PBC
		# energy is then spin_config.T dot effmagH dot spin_config
		self.effmagH = np.zeros((self.N*self.M,self.N*self.M))
		whole_sc_a1 = self.N * self.a1
		whole_sc_a2 = self.M * self.a2
		for i in range(self.N*self.M):
			for j in range(self.N*self.M):
				rij = self.get_intersite_R(i, j)
				# deal with interactions to other periodic images - image_rep increased until convergence 
				for ii in range(-image_rep[0], image_rep[0]+1):
					for jj in range(-image_rep[1], image_rep[1]+1):
						rij_image = rij + ii*whole_sc_a1 + jj*whole_sc_a2
						self.effmagH[i,j] += coupling_function((rij_image[0]**2 + rij_image[1]**2)**0.5)
		return self.effmagH

	def construct_effective_occH(self, coupling_function, image_rep): # deals with PBC
		# energy is then spin_config.T dot effmagH dot spin_config
		self.effoccH = np.zeros((self.N*self.M,self.N*self.M))
		whole_sc_a1 = self.N * self.a1
		whole_sc_a2 = self.M * self.a2
		for i in range(self.N*self.M):
			for j in range(self.N*self.M):
				rij = self.get_intersite_R(i, j)
				for ii in range(-image_rep[0], image_rep[0]+1):
					for jj in range(-image_rep[1], image_rep[1]+1):
						rij_image = rij + ii*whole_sc_a1 + jj*whole_sc_a2
						self.effoccH[i,j] += coupling_function((rij_image[0]**2 + rij_image[1]**2)**0.5)
		return self.effoccH

	# will depend heavily on choice in N, M and appreciably on choice in coupling_function
	def check_convergence_effectiveham(self, coupling_function, range_check = [2, 5, 10, 20]):
		f, ax = plt.subplots(2,2)
		ax = ax.flatten()
		assert(self.N == self.M)
		for i in range(len(range_check)):
			self.visualize_hamiltonian(coupling_function, 0, [range_check[i], range_check[i]], ax[i], show=False)
		plt.show()

	def visualize_hamiltonian(self, coupling_function, Hfield, image_rep, ax=None, show=True):
		if ax == None: f, ax = plt.subplots(1,1)
		self.setup_mag_energetics(coupling_function, Hfield, image_rep)
		if self.occ_optimized: ax.imshow(self.pruned_effmagH)
		else: ax.imshow(self.effmagH)
		if show: plt.show()

	####################################################################################
	# plotting function
	# black : empty
	# red : spin up
	# blue : spin down
	####################################################################################
	def visualize(self, ax=None, debug=True, path=None):
		if ax == None: 
			f, ax = plt.subplots(1,self.P)

		for layer in range(self.P):

			xy = np.zeros((self.N*self.M,2))
			rgb = np.zeros((self.N*self.M,3))
			
			for n in range(self.N):
				for m in range(self.M):
					vec = n * self.a1 + m * self.a2 
					xy[i,0] = vec[0]
					xy[i,1] = vec[1]
					i = self.nm_to_site_index(n,m,layer)	
					if self.occ_config[i] == 0: rgb[i,:] = [0,0,0]
					elif self.occ_config[i] == 1: rgb[i,:] = [1,0,0]
			ax[layer].scatter(xy[:,0], xy[:,1], color=rgb) 

		if path == None:
			plt.show()
		else:
			plt.savefig(path, dpi=300)

	####################################################################################
	# composite index handling
	####################################################################################
	def site_index_to_nm(self, i):
		nm = np.unravel_index(i, (self.N, self.M))
		return nm[0], nm[1]
	def nm_to_site_index(self, n,m):
		return np.ravel_multi_index((n,m), (self.N, self.M))

	####################################################################################
	# without consideration of PBC!
	####################################################################################
	def get_intersite_R(self, site1, site2):
		n1, m1 = self.site_index_to_nm(site1)
		n2, m2 = self.site_index_to_nm(site2)
		delR = (n2-n1) * self.a1 + (m2-m1) * self.a2 
		return delR

	def set_neighbor_table(self):
		for i in range(self.size()):
			for j in range(self.size()):
				if i > j and self.compute_are_neighbors(i,j):
					self.neighbor_table[i,j] = 1
					self.neighbor_table[j,i] = 1

	def compute_are_neighbors(self, site1, site2):
		rij = self.get_intersite_R(site1, site2)
		whole_sc_a1 = self.N * self.a1
		whole_sc_a2 = self.M * self.a2
		for ii in [-1,0,1]:
			for jj in [-1,0,1]:
				rij_image = rij + ii*whole_sc_a1 + jj*whole_sc_a2
				rij_image_len = (rij_image[0]**2 + rij_image[1]**2)**0.5
				if round(rij_image_len,2) == round(self.a0,2): return True
		return False

	def list_nearest_neighbors(self, site):
		return [i for i in range(self.size()) if (self.neighbor_table[site,i] == 1)] 

	def list_viable_hop_sites(self, site):
		nn = self.list_nearest_neighbors(site)
		site_occ = self.occ_config[site] 
		viable_nn = [i for i in nn if site_occ != self.occ_config[i]]
		return viable_nn

	####################################################################################
	# basic MC moves/accept/reject/∆energy functions - spin flip 
	####################################################################################
	def propose_spin_flip(self, site):
		self.active_site = site
		return True

	def accept_spin_flip(self):
		if self.occ_optimized:
			self.pruned_spins[self.active_site] *= -1
		else:
			self.spin_config[self.active_site] *= -1
		self.energy = self.proposed_energy
		self.proposed_energy = None
		self.active_site = None

	def reject_spin_flip(self):
		self.proposed_energy = None
		self.active_site = None

	# expensive... pls only use smaller lattices
	def get_dE_spinflip(self):
		if self.occ_optimized:
			if self.energy == None: self.energy = np.transpose(self.pruned_spins) @ self.pruned_effmagH @ self.pruned_spins
			self.pruned_spins[self.active_site] *= -1
			self.proposed_energy = np.transpose(self.pruned_spins) @ self.pruned_effmagH @ self.pruned_spins 
			self.proposed_energy -= self.field * np.sum(self.pruned_spins)
			self.pruned_spins[self.active_site] *= -1
		else:
			if self.energy == None: self.energy =  np.transpose(self.spin_config) @ self.effmagH @ self.spin_config 
			self.spin_config[self.active_site] *= -1
			self.proposed_energy = np.transpose(self.spin_config) @ self.effmagH @ self.spin_config 
			self.proposed_energy -= self.field * np.sum(self.spin_config)
			self.spin_config[self.active_site] *= -1
		return self.proposed_energy - self.energy 

	####################################################################################
	# basic MC moves/accept/reject/∆energy functions - hop (for kawasaki eqMC)
	####################################################################################
	def propose_hop(self, site1, site2):
		self.active_site1 = site1
		self.active_site2 = site2 
		return True

	def accept_hop(self):
		self.energy = self.proposed_energy
		self.site_swap(self.active_site1, self.active_site2)
		if self.occ_config[self.active_site1][0] == 1: m, k = self.active_site1, self.active_site2
		else:  k, m = self.active_site1, self.active_site2
		self.proposed_energy = None
		self.active_site1 = None
		self.active_site2 = None
		compute = self.Hsigmai + self.effoccH[m,:] - self.effoccH[k,:]
		#expect  = (self.effoccH @ self.occ_config).flatten();
		#for i in range(len(expect)): assert(np.round(expect[i], 5) == np.round(compute[i], 5))
		self.Hsigmai = compute

	def site_swap(self, s1, s2):
		temp1 = self.occ_config[s1].copy()
		temp2 = self.occ_config[s2].copy()
		self.occ_config[s1] = temp2
		self.occ_config[s2] = temp1

	def reject_hop(self):
		self.proposed_energy = None
		self.active_site1 = None
		self.active_site2 = None

	# expensive... pls only use smaller lattices
	# no chemical potential since fixed x when kawasaki
	def get_dE_hop(self):
		#e2 = self.get_dE_hop_rank1upd()
		#e1 = self.get_dE_hop_matrixprod()
		#print(e1, e2); assert(np.round(e1, 5) == np.round(e2, 5))
		return self.get_dE_hop_rank1upd()

	def get_dE_hop_matrixprod(self):
		if self.energy == None: self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.site_swap(self.active_site1, self.active_site2)
		self.proposed_energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.site_swap(self.active_site1, self.active_site2)
		return self.proposed_energy - self.energy 

	def get_dE_hop_rank1upd(self): #
		if self.energy == None: self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config
		if self.Hsigmai[0] == None: self.Hsigmai = (self.effoccH @ self.occ_config).flatten();
		if self.occ_config[self.active_site1][0] == 1: m, k = self.active_site1, self.active_site2
		else:  k, m = self.active_site1, self.active_site2
		dE = self.effoccH[m,m] + self.effoccH[k,k] - 2*self.effoccH[m,k] 
		dE += 2*self.Hsigmai[k] - 2*self.Hsigmai[m]
		self.proposed_energy = self.energy + dE
		return dE

	####################################################################################
	# order params and stats
	####################################################################################
	def magnetization(self):
		if self.occ_optimized:
			return np.sum(self.pruned_spins)/np.sum(np.abs(self.pruned_spins))
		return np.sum(self.spin_config)/np.sum(np.abs(self.spin_config))

	def interc_density(self):
		return np.mean(self.occ_config)

	def get_energy(self):
		return self.energy

	def sublattice_2x2_order_param(self, debug=False):
		assert(self.N%2 == 0 and self.M%2 == 0)
		sl_occ = [0,0,0,0] # 4 different 2x2 sublattices
		for i in range(self.N*self.M):
			n,m = self.site_index_to_nm(i)
			if n%2 == 0 and m%2 == 0: sl_occ[0] += self.occ_config[i]
			if n%2 == 0 and m%2 == 1: sl_occ[1] += self.occ_config[i]
			if n%2 == 1 and m%2 == 0: sl_occ[2] += self.occ_config[i]
			if n%2 == 1 and m%2 == 1: sl_occ[3] += self.occ_config[i]
		if debug:
			f, axes = plt.subplots(1,2)
			ax = axes[1]
			print(sl_occ)
			xy = np.zeros((self.N*self.M,2))
			rgb = np.zeros((self.N*self.M,3))
			for i in range(self.N*self.M):
				n,m = self.site_index_to_nm(i)
				vec = n * self.a1 + m * self.a2 
				xy[i,0] = vec[0]
				xy[i,1] = vec[1]
				if n%2 == 0 and m%2 == 0: rgb[i,:] = [1,0,0]
				if n%2 == 0 and m%2 == 1: rgb[i,:] = [0,1,0]
				if n%2 == 1 and m%2 == 0: rgb[i,:] = [0,0,1]
				if n%2 == 1 and m%2 == 1: rgb[i,:] = [0,0,0]
			ax.scatter(xy[:,0], xy[:,1], color=rgb) 
			self.visualize(axes[0])
		return (np.max(sl_occ))/np.sum(self.occ_config) # portion in a single sublattice 

	def sublattice_r3xr3_order_param(self, debug=False):
		assert(self.N%3 == 0 and self.M%3 == 0)
		sl_occ = [0,0,0] # 3 different 2x2 sublattices
		for i in range(self.N*self.M):
			n,m = self.site_index_to_nm(i)
			if (n%3 == 0 and m%3 == 0) or (n%3 == 1 and m%3 == 2) or (n%3 == 2 and m%3 == 1): sl_occ[0] += self.occ_config[i]
			if (n%3 == 0 and m%3 == 1) or (n%3 == 1 and m%3 == 0) or (n%3 == 2 and m%3 == 2): sl_occ[1] += self.occ_config[i]
			if (n%3 == 0 and m%3 == 2) or (n%3 == 1 and m%3 == 1) or (n%3 == 2 and m%3 == 0): sl_occ[2] += self.occ_config[i]
		if debug:
			f, axes = plt.subplots(1,2)
			ax = axes[1]
			print(sl_occ)
			xy = np.zeros((self.N*self.M,2))
			rgb = np.zeros((self.N*self.M,3))
			for i in range(self.N*self.M):
				n,m = self.site_index_to_nm(i)
				vec = n * self.a1 + m * self.a2 
				xy[i,0] = vec[0]
				xy[i,1] = vec[1]
				if (n%3 == 0 and m%3 == 0): rgb[i,:] = [0,0,1]
				if (n%3 == 1 and m%3 == 2): rgb[i,:] = [0,0,1]
				if (n%3 == 2 and m%3 == 1): rgb[i,:] = [0,0,1]
				if (n%3 == 0 and m%3 == 1): rgb[i,:] = [1,0,0]
				if (n%3 == 1 and m%3 == 0): rgb[i,:] = [1,0,0]
				if (n%3 == 2 and m%3 == 2): rgb[i,:] = [1,0,0]
				if (n%3 == 0 and m%3 == 2): rgb[i,:] = [0,1,0]
				if (n%3 == 1 and m%3 == 1): rgb[i,:] = [0,1,0]
				if (n%3 == 2 and m%3 == 0): rgb[i,:] = [0,1,0]
			ax.scatter(xy[:,0], xy[:,1], color=rgb) 
			self.visualize(axes[0])
		return (np.max(sl_occ))/np.sum(self.occ_config) # portion in a single sublattice -- should be = 1 for perfect r3xr3 

