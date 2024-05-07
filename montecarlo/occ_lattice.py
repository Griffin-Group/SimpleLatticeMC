

import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np 


class OccupationalLattice: # base class to inherit from for a 2d or 3d occupational lattice

	####################################################################################
	# init functions
	####################################################################################
	def __init__(self, dims, latvec, a0, c, occ_config=None, x=None, superlattice=None):
		self.N = dims # set dimensions
		self.d = len(dims)
		if self.d not in [2,3]: 
			print('ERROR: cannot handle dimensionalities other than 2d and bulk 3d sorry!')
			exit()
		self.a = latvec
		self.a0 = a0
		self.c = c
		self.sc_a = [self.N[i] * self.a[i] for i in range(self.d)]
		self.neighbor_table = np.zeros((self.size(), self.size()))
		self.set_neighbor_table()
		if occ_config is None: 
			if superlattice is None: self.random_initialize_occ(x)
			else: self.superlattice_init_occ(period=superlattice)
		else: self.occ_config = occ_config.flatten()

	def initialzie_occ(self, occ_config, verbose=False):
		self.occ_config = occ_config
		self.energy = None
		if verbose:
			for i in range(self.size()):
				if self.occ_config[i]:
					n,m,p = self.site_index_to_nm(i)
					v = n * self.a[0] + m * self.a[1] + p * self.a[2]
					print("placed an interc in :", v)


	# if more than 2 layers, forces same x in each layer!!!
	def random_initialize_occ(self, x=None):
		#print('init occ')
		if x == None: #random itercalation density too
			occ_config = np.zeros((self.size(), 1))
			for n in range(self.size()): occ_config[n] = random.randint(0, 2)
		else: 
			N_filled_sites = int(round(x*self.size() , 0))
			#print('filling {} sites'.format(N_filled_sites))
			occ_config = np.zeros((self.size(), 1))
			
			# random with specified itercalation density 
			if self.d == 2:
				while np.sum(occ_config) < N_filled_sites:
					site = random.randint(0, self.size())
					occ_config[random.randint(0, self.size())] = 1
					#print('filling site', site)
				assert(np.sum(occ_config) == N_filled_sites)

			# if more than 2 layers, forces same x in each layer!!!
			elif self.d == 3:
				Nlayers = self.N[-1]
				#print(Nlayers, " layers")
				N_filled_sites_per_layer = int(N_filled_sites / Nlayers) # will be integer
				#print(N_filled_sites_per_layer, " filled per layer")
				Nfilledsofar = np.zeros((Nlayers, 1))
				while np.sum(occ_config) < N_filled_sites:
					site = random.randint(0, self.size())
					layer = self.layernumber(site)
					if Nfilledsofar[layer] < N_filled_sites_per_layer:
						Nfilledsofar[layer] += 1
						occ_config[site] = 1
						#print('filling a site in layer ', layer)
				assert(np.sum(occ_config) == N_filled_sites)

		#print('done init occ')
		self.occ_config = occ_config
		self.energy = None
		
	def superlattice_init_occ(self, periods=[2,2,0]): # only works for integers rn so 2x2, 4x4 cant get the r3xr3
		occ_config = np.zeros((self.size(), 1))
		for i in range(self.size()):
			n,m,p = self.site_index_to_nm(i)
			if n%periods[0] == 0 and m%periods[1] == 0 and p%periods[2] == 0: occ_config[i] = 1
		self.occ_config = occ_config
		self.energy = None

	######################################################################################
	# hamiltonian functions (general coupling)
	######################################################################################
	def construct_effective_H_general(self, coupling_function, image_rep): # deals with PBC if image rep > 0
		self.effoccH = np.zeros((self.size(), self.size()))
		for i in range(self.size()):
			for j in range(self.size()):
				# get distance between all site pairs
				rij = self.get_intersite_R(i, j) 
				for ii in range(-image_rep[0], image_rep[0]+1):
					for jj in range(-image_rep[1], image_rep[1]+1):
						if self.d == 3:
							for kk in range(-image_rep[2], image_rep[2]+1):
								rij_image = rij + ii*self.sc_a[0] + jj*self.sc_a[1] + kk*self.sc_a[2] 
								self.effoccH[i,j] += coupling_function(np.sqrt(np.sum( [r**2 for r in rij_image])))
						elif self.d == 2:
							rij_image = rij + ii*self.sc_a[0] + jj*self.sc_a[1] 
							self.effoccH[i,j] += coupling_function(np.sqrt(np.sum( [r**2 for r in rij_image])))
		return self.effoccH

	def setup_occ_energetics_general(self, coupling_function, image_rep=[1,1,1], chempot=0, offset=0):
		self.construct_effective_H_general(coupling_function, image_rep)
		self.chempot = chempot # no effect if kawasaki
		self.offset = offset
		self.update_energy()

	def update_energy(self):
		self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.energy += self.chempot * np.sum(self.occ_config) 
		self.energy += self.offset


	####################################################################################
	# order params and stats
	####################################################################################
	def interc_density(self): return np.mean(self.occ_config)
	def get_energy(self): return self.energy
	def size(self): return np.prod(self.N) # number sites
	def occ_size(self): return int(np.sum(self.occ_config)) # number full sites

	####################################################################################
	# basic MC moves (kawaski)
	####################################################################################
	def propose_hop(self, site1, site2):
		self.active_site1, self.active_site2 = site1, site2
		return True
	def accept_hop(self):
		self.energy = self.proposed_energy
		self.site_swap(self.active_site1, self.active_site2)
		self.proposed_energy, self.active_site1, self.active_site2 = None, None, None
	def site_swap(self, s1, s2):
		temp1, temp2 = self.occ_config[s1].copy(), self.occ_config[s2].copy()
		self.occ_config[s1], self.occ_config[s2] = temp2, temp1
	def reject_hop(self): self.proposed_energy, self.active_site1, self.active_site2 = None, None, None
	def list_viable_hop_sites(self, site):
		nn = self.list_nearest_neighbors(site)
		site_occ = self.occ_config[site] 
		viable_nn = [i for i in nn if site_occ != self.occ_config[i]]
		return viable_nn # nearest neigbors (in plane and if applicable out of plane too)
	def get_dE_hop_matrixprod(self):
		if self.energy == None: self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.site_swap(self.active_site1, self.active_site2)
		self.proposed_energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
		self.site_swap(self.active_site1, self.active_site2)
		return self.proposed_energy - self.energy
	def get_dE_hop_rank1upd(self): # TO DO verify with get_dE_hop_matrixprod, know correct but slow
		if self.energy == None: self.update_energy()
		i, j, dE = self.active_site1, self.active_site2, 0
		#for k in range(self.size()):
		#	dE += ( 2 * self.occ_config[k] * (1 - 2 * self.occ_config[j]) * (self.effoccH[k,j] - self.effoccH[k,i]) )
		if self.active_site1 == 1: occsite, vacsite = self.active_site1, self.active_site2
		else: occsite, vacsite = self.active_site1, self.active_site2
		temp = self.effoccH @ self.occ_config 
		dE = 2*temp[vacsite] - 2*temp[occsite] + self.effoccH[occsite,occsite] + self.effoccH[vacsite,vacsite] - 2*self.effoccH[occsite,vacsite]
		self.proposed_energy = self.energy + dE
		return dE
	def get_dE_hop(self): return self.get_dE_hop_rank1upd()

	####################################################################################
	# composite index handling for 2d (d=2) and 3d (d=3)
	####################################################################################
	def site_index_to_nm(self, i):
		if self.d == 2: 
			nm = np.unravel_index(i, (self.N[0], self.N[1]))
			return nm[0], nm[1]
		elif self.d == 3:
			nm = np.unravel_index(i, (self.N[0], self.N[1], self.N[2]))
			return nm[0], nm[1], nm[2]
	
	def nm_to_site_index(self, n, m, p=0): 
		if self.d == 2: return np.ravel_multi_index((n,m), (self.N[0], self.N[1]))
		elif self.d == 3: return np.ravel_multi_index((n,m,p), (self.N[0], self.N[1], self.N[2]))
	
	def layernumber(self, i):
		if self.d == 2: return 0
		elif self.d == 3:
			_,_,p = self.site_index_to_nm(i)
			return p


	def sublattice_r3xr3_order_param(self, debug=False):
		assert(self.N[0]%3 == 0 and self.N[1]%3 == 0)
		sl_occ = [0,0,0] # 3 different 2x2 sublattices
		for i in range(self.size()):
			if self.d == 2: n,m   = self.site_index_to_nm(i)
			if self.d == 3: n,m,p = self.site_index_to_nm(i)

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

	def sublattice_2x2_order_param(self, debug=False):
		assert(self.N[0]%2 == 0 and self.N[1]%2 == 0)
		sl_occ = [0,0,0,0] # 4 different 2x2 sublattices
		for i in range(self.size()):
			if self.d == 2: n,m   = self.site_index_to_nm(i)
			if self.d == 3: n,m,p = self.site_index_to_nm(i)
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

	####################################################################################
	# neighbor handling 
	####################################################################################
	def get_intersite_R(self, site1, site2):
		if self.d == 2:
			n1, m1 = self.site_index_to_nm(site1)
			n2, m2 = self.site_index_to_nm(site2)
			delR = (n2-n1) * self.a[0] + (m2-m1) * self.a[1]
			return delR
		if self.d == 3:
			n1, m1, p1 = self.site_index_to_nm(site1)
			n2, m2, p2 = self.site_index_to_nm(site2)
			delR = (n2-n1) * self.a[0] + (m2-m1) * self.a[1] + (p2-p1) * self.a[2]
			return delR
	def set_neighbor_table(self):
		for i in range(self.size()):
			for j in range(self.size()):
				if i > j and self.compute_are_neighbors(i,j):
					self.neighbor_table[i,j] = 1
					self.neighbor_table[j,i] = 1 # nearest neigbors (in plane and if applicable out of plane too)
	
	def list_nearest_neighbors(self, site): return [i for i in range(self.size()) if (self.neighbor_table[site,i] == 1)] 
	
	def compute_are_neighbors(self, site1, site2): # TEST ME FOR 3D!!!
		if self.d == 2: 
			rij = self.get_intersite_R(site1, site2)
			for ii in [-1,0,1]:
				for jj in [-1,0,1]:
					rij_image = rij + ii*self.sc_a[0] + jj*self.sc_a[1]
					rij_image_len = (rij_image[0]**2 + rij_image[1]**2)**0.5
					if round(rij_image_len,2) == round(self.a0, 2): return True
		elif self.d == 3: 
			rij = self.get_intersite_R(site1, site2)
			for ii in [-1,0,1]:
				for jj in [-1,0,1]:
					for kk in [-1,0,1]:
						rij_image = rij + ii*self.sc_a[0] + jj*self.sc_a[1] + kk*self.sc_a[2]
						rij_image_len = (rij_image[0]**2 + rij_image[1]**2 + rij_image[2]**2)**0.5
						if round(rij_image_len,2) == round(self.a0, 2): return True
		return False

	####################################################################################
	# visulatization functions
	####################################################################################
	def visualize(self, ax=None, path=None):
		if self.d == 2: self.visualize_2d(ax, path)
		if self.d == 3: self.visualize_3d(ax, path)
	def visualize_2d(self, ax=None, path=None):
		if ax == None: f, ax = plt.subplots()
		xy = np.zeros((self.size(),2))
		rgb = np.zeros((self.size(),3))
		for i in range(self.size()):
			n,m,p = self.site_index_to_nm(i)
			vec = n * self.a[0] + m * self.a[1]
			xy[i,0], xy[i,1] = vec[0], vec[1]
			if self.occ_config[i] == 0:   rgb[i,:] = [0,0,0]
			elif self.occ_config[i] == 1: rgb[i,:] = [1,0,0]
		ax.scatter(xy[:,0], xy[:,1], color=rgb) 
		if path == None: plt.show()
		else: plt.savefig(path, dpi=300)
	def visualize_3d(self, ax=None, path=None):
		nlayers = self.N[2]
		inplanesize = self.N[0]*self.N[1]
		if ax == None: f, ax = plt.subplots(1, nlayers)
		xy  = np.zeros((inplanesize,2, nlayers))
		rgb = np.zeros((inplanesize,3, nlayers))
		for i in range(inplanesize):
			n,m,p = self.site_index_to_nm(i)
			layernumber = p
			vec = n * self.a[0] + m * self.a[1]
			xy[i,0,layernumber], xy[i,1,layernumber] = vec[0], vec[1]
			if self.occ_config[i] == 0:   rgb[i,:,layernumber] = [0,0,0]
			elif self.occ_config[i] == 1: rgb[i,:,layernumber] = [1,0,0]
		for l in range(nlayers):
			ax[l].scatter(xy[:,0,l], xy[:,1,l], color=rgb[:,:,l]) 
			ax[l].set_title('Layer #{}'.format(l))
		if path == None: plt.show()
		else: plt.savefig(path, dpi=300)



class HexagonalOccLattice(OccupationalLattice): # for a hexagonal lattice only

	######################################################################################
	# hexagonal-specific hamiltonian functions
	######################################################################################
	def setup_occ_hexIsing(self, J01, J11, J10, Jr30, J20, J02):
		self.construct_H_hexagonal_Ising(J01, J11, J10, Jr30, J20, J02)
		self.energy = np.transpose(self.occ_config) @ self.effoccH @ self.occ_config 
	def construct_H_hexagonal_Ising(self, J01, J11, J10, Jr30, J20, J02):  # deals with PBC
		self.effoccH = np.zeros((self.size(), self.size()))
		for i in range(self.size()):
			for j in range(self.size()):
				n1, m1, p1 = self.site_index_to_nm(site1)
				n2, m2, p2 = self.site_index_to_nm(site2)
				if (p1 == p2):
					if (n1 == n2) and (np.abs(m1 - m2) == 1):   self.effoccH[i,j] = J10 
					if (m1 == m2) and (np.abs(n1 - n2) == 1):   self.effoccH[i,j] = J10
					if ((m1 - m2) == 1) and ((n1 - n2) == -1):  self.effoccH[i,j] = J10
					if ((m1 - m2) == -1) and ((n1 - n2) == 1):  self.effoccH[i,j] = J10
					if (n1 == n2) and (np.abs(m1 - m2) == 2):   self.effoccH[i,j] = J20
					if (m1 == m2) and (np.abs(n1 - n2) == 2):   self.effoccH[i,j] = J20
					if ((m1 - m2) == 2) and ((n1 - n2) == -2):  self.effoccH[i,j] = J20
					if ((m1 - m2) == -2) and ((n1 - n2) == 2):  self.effoccH[i,j] = J20
					if ((m1 - m2) == 1)  and ((n1 - n2) == 1):  self.effoccH[i,j] = Jr30
					if ((m1 - m2) == -1) and ((n1 - n2) == -1): self.effoccH[i,j] = Jr30
					if ((m1 - m2) == 2) and ((n1 - n2) == -1):  self.effoccH[i,j] = Jr30
					if ((m1 - m2) == -2) and ((n1 - n2) == 1):  self.effoccH[i,j] = Jr30
					if ((m1 - m2) == 1) and ((n1 - n2) == -2):  self.effoccH[i,j] = Jr30
					if ((m1 - m2) == -1) and ((n1 - n2) == 2):  self.effoccH[i,j] = Jr30
				if (np.abs(p1 - p2) == 1):
					if (n1 == n2) and (m1 == m2): 				self.effoccH[i,j] = J01 
					if (n1 == n2) and (np.abs(m1 - m2) == 1):   self.effoccH[i,j] = J11 
					if (m1 == m2) and (np.abs(n1 - n2) == 1):   self.effoccH[i,j] = J11
					if ((m1 - m2) == 1) and ((n1 - n2) == -1):  self.effoccH[i,j] = J11
					if ((m1 - m2) == -1) and ((n1 - n2) == 1):  self.effoccH[i,j] = J11
				if (np.abs(p1 - p2) == 2):
					if (n1 == n2) and (m1 == m2): 				self.effoccH[i,j] = J02 
		return self.effoccH	
	
	####################################################################################
	# hexagonal-specific order params 
	####################################################################################
	# fraction of intercalants all within the same in-plane 2x2 sublattice (there are 4)
	# OP=1 when in perfect 2x2 (or 4x4, 6x6, 8x8...) because all in same 2x2 sublattice
	# if exactly half in one sublattice and half in another, OP=1/2 
	# if equally within all 4 sublattices, OP=1/4
	def sublattice_220_order_param(self): 
		assert(self.N[0]%2 == 0 and self.N[1]%2 == 0)
		if self.d == 3: Nlayers = self.N[2]
		else: Nlayers = 1
		sl_occ = np.zeros((4, Nlayers)) # 4 different 2x2 sublattices for each layer
		for i in range(self.size()):
			n,m,p = self.site_index_to_nm(i)
			if n%2 == 0 and m%2 == 0: sl_occ[0,p] += self.occ_config[i]
			if n%2 == 0 and m%2 == 1: sl_occ[1,p] += self.occ_config[i]
			if n%2 == 1 and m%2 == 0: sl_occ[2,p] += self.occ_config[i]
			if n%2 == 1 and m%2 == 1: sl_occ[3,p] += self.occ_config[i]
		if self.d == 2: return (np.max(sl_occ[:,0]))/np.sum(sl_occ[:,0]) # portion of interc in a single 2x2 sublattice 
		o22 = np.zeros((Nlayers, 1))
		for p in range(Nlayers):
			# define separately for each unique layer
			o22[p] = (np.max(sl_occ[:,p]))/np.sum(sl_occ[:,p])
		return o22
	
	# for each column (in-plane site) define two sublattices, OP in that column is fraction of 
	# occupancies in the maximumally full sublattice. So OP=1 when all in a single sublattice,
	# OP=1/2 when equally partitioned between the two (like in-plane OP definitions). 
	# final returned value is the average of all the single-column OPs. 
	# so final OP=1 when all columns in perfect out of plane 2 sublattice.
	# OP=1/2 if half of columns in perfect 2 sublattice. 
	def sublattice_002_order_param(self):
		assert(self.d == 3)
		assert(self.N[2]%2 == 0)
		N_columns = self.N[0]*self.N[1]
		OP = 0
		sl_occ = [0,0] # 2 different sublattices for each column - XOXOXO and OXOXOX.. 
		for n in range(self.N[0]):
			for m in range(self.N[1]):
				for p in range(self.N[2]): sl_occ[p%2] += self.occ_config[self.nm_to_site_index(n,m,p)]
				OP += (np.max(sl_occ[:]))/np.sum(sl_occ[:]) # =1 if all intercalants in same sublattice
		return OP/self.size() # fraction of columns (in-plane sites) that are equivalent every layer

	# same as 002 but with 003
	def sublattice_003_order_param(self):
		assert(self.d == 3)
		assert(self.N[2]%3 == 0)
		N_columns = self.N[0]*self.N[1]
		OP = 0
		sl_occ = [0,0] # 2 different sublattices for each column - XOXOXO and OXOXOX.. 
		for n in range(self.N[0]):
			for m in range(self.N[1]):
				for p in range(self.N[2]): sl_occ[p%3] += self.occ_config[self.nm_to_site_index(n,m,p)]
				OP += (np.max(sl_occ[:]))/np.sum(sl_occ[:]) # =1 if all intercalants in same sublattice
		return OP/self.size() # fraction of columns (in-plane sites) that are equivalent every layer

	# fraction of columns (in-plane sites) that are equivalent (all empty or full) every layer
	# OP=1 when exactly the same between layer 1 and layer 2 
	# OP=0 when no two sites the same between layer 1 and layer 2 
	def sublattice_001_order_param(self): 
		assert(self.d == 3)
		N_columns = self.N[0]*self.N[1]
		OP = 0
		for n in range(self.N[0]):
			for m in range(self.N[1]):
				f_column_same = 0
				for p in range(self.N[2]): f_column_same += self.occ_config[self.nm_to_site_index(n,m,p)]
				f_column_same /= self.N[2] # right now will be 0 if all empty, 1 if all full, 0.5 if mixed. dont want to distinguish between full/mix
				if f_column_same < 0.5: f_column_same = 1-f_column_same
				OP += f_column_same
		return (f_column_same)/N_columns # fraction of columns (in-plane sites) that are equivalent (all empty or full) every layer

	# fraction of intercalants all within the same in-plane r3xr3 sublattice (there are 3)
	# OP=1 when in perfect r3xr3 (or 3x3, r27xr27, 9x9...) because all in same r3xr3 sublattice
	# if exactly half in one sublattice and half in another, OP=1/2 
	# if equally within all 3 sublattices, OP=1/3
	def sublattice_330_order_param(self): 
		assert(self.N[0]%3 == 0 and self.N[1]%3 == 0)
		if self.d == 3: Nlayers = self.N[2]
		else: Nlayers = 1
		sl_occ = np.zeros((3, Nlayers)) # 3 different sublattices for each layer
		for i in range(self.size()):
			n,m,p = self.site_index_to_nm(i)
			if (n%3 == 0 and m%3 == 0) or (n%3 == 1 and m%3 == 2) or (n%3 == 2 and m%3 == 1): sl_occ[0,p] += self.occ_config[i]
			if (n%3 == 0 and m%3 == 1) or (n%3 == 1 and m%3 == 0) or (n%3 == 2 and m%3 == 2): sl_occ[1,p] += self.occ_config[i]
			if (n%3 == 0 and m%3 == 2) or (n%3 == 1 and m%3 == 1) or (n%3 == 2 and m%3 == 0): sl_occ[2,p] += self.occ_config[i]
		if self.d == 2: return (np.max(sl_occ[:,0]))/np.sum(sl_occ[:,0]) # portion in a single sublattice -- should be = 1 for perfect r3xr3 
		o33 = np.zeros((Nlayers, 1))
		for p in range(Nlayers):
			# define separately for each unique layer
			o33[p] = (np.max(sl_occ[:,p]))/np.sum(sl_occ[:,p])
		return o33

