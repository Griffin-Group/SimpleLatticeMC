
import numpy as np  
import numpy.random as random
from lattices import RigidHexagonalLattice
import matplotlib.pyplot as plt
import time

####################################################################################
# metropolis hastings equilibrium monte carlo with a given hamiltonian
# H: external field in : ( E = \sum_ij s_i Ham_ij s_j - H \sum_i s_i )
# beta: 1/kbT
# coupling_function: function for J(r) gives Ham_ij = J(|ri-rj|) summed over all 
# periodic images (can choose to go far out for slowly convergent couplings)
####################################################################################
def spinflip_monte_carlo(lattice, beta, H, verbose=False, n_steps=1e3):

	N = lattice.occ_size()
	op = np.zeros((int(n_steps*N),1))
	e = np.zeros((int(n_steps*N),1))
	lattice.random_initialize_spin()
	lattice.modify_field(H) 
	N_accept = 0
	N_reject = 0

	for step in range(int(n_steps*N)):

		if verbose and (4*step)%(n_steps*N) == 0: 
			print('...{}% done'.format(100*step/(n_steps*N)))

		sucess = False
		while not sucess: # sometimes had some move proposals that can fail (ie hops need specific site choices)
			site = random.randint(0, N)
			sucess = lattice.propose_spin_flip(site)

		dE = lattice.get_dE_spinflip()
		r = random.rand()
		if r <= np.exp(-beta*dE): # do the metropolis hastings 
			lattice.accept_spin_flip()
			N_accept += 1
		else: 
			lattice.reject_spin_flip()
			N_reject += 1

		op[step] = lattice.magnetization() # care about mag rn
		e[step] = lattice.get_energy()

	if verbose: print(N_accept/(N_accept+N_reject) * 100, '% of moves accepted')
	return op, e
	
####################################################################################
# kawasaki dynamics, equilibrium monte carlo with a given hamiltonian at fixed x
# mu: chemical pot in : ( E = \sum_ij s_i Ham_ij s_j - mu \sum_i s_i )
# si are occupancy variables, 0 or 1
# beta: 1/kbT
####################################################################################
def occhop_monte_carlo(lattice, beta, x, verbose=False, n_steps=1e3):

	N = lattice.size()
	op2x2 = np.zeros((int(n_steps*N),1))
	opr3xr3 = np.zeros((int(n_steps*N),1))
	e = np.zeros((int(n_steps*N),1))
	N_accept = 0
	N_reject = 0
	N_propose_fail = 0
	lattice.random_initialize_occ(x)
	#print('GOT PAST RANDOM INIT 3D'); exit()

	if x == 0 or x == 1:
		print('cant hop...lattice completely full or empty?')
		exit()

	for step in range(int(n_steps*N)):

		if verbose and (4*step)%(n_steps*N) == 0: 
			print('...{}% done'.format(100*step/(n_steps*N)))

		sucess = False
		count = 0
		while not sucess: # sometimes had some move proposals that can fail (ie hops need specific site choices)
			site1 = random.randint(0, N)
			nn_list = lattice.list_viable_hop_sites(site1)
			if len(nn_list) == 0: 
				N_propose_fail += 1
				count += 1
			else:
				site2 = nn_list[random.randint(0, len(nn_list))]
				assert(lattice.occ_config[site1] != lattice.occ_config[site2])
				sucess = lattice.propose_hop(site1, site2)
			if count > 150:
				lattice.visualize()
				print('I got stuck...?')
				exit()

		dE = lattice.get_dE_hop()
		r = random.rand()
		if r <= np.exp(-beta*dE): # do the metropolis hastings 
			lattice.accept_hop()
			N_accept += 1
		else: 
			lattice.reject_hop()
			N_reject += 1

		op2x2[step] = lattice.sublattice_2x2_order_param() # care about mag rn
		opr3xr3[step] = lattice.sublattice_r3xr3_order_param() # care about mag rn
		e[step] = lattice.get_energy()

		"""
		if step%50 == 0:
			print('2x2 : ', op2x2[step])
			print('3x3 : ', opr3xr3[step])
			lattice.visualize(debug=False)
		"""

	if verbose:
		f,ax = plt.subplots(1,4)
		ax[0].plot(op2x2)
		ax[0].set_title('2x2')
		ax[1].plot(opr3xr3)
		ax[1].set_title('r3xr3')
		ax[2].plot(e)
		lattice.visualize(ax[3])

	if verbose: 
		print(N_accept/(N_accept+N_reject) * 100, '% of moves accepted')
		print(N_propose_fail, 'proposal failures')
	return op2x2, opr3xr3, e

def occhop_mc_speedtest(layers, lattice, beta, x):

	N = lattice.size()
	# DIFFERENT FROM 2D
	op2x2 = np.zeros((int(n_steps*N),layers))
	opr3xr3 = np.zeros((int(n_steps*N),layers))
	op_oop = np.zeros((int(n_steps*N),1))

	e = np.zeros((int(n_steps*N),1))
	N_accept = 0
	N_reject = 0
	N_propose_fail = 0
	lattice.random_initialize_occ(x, verbose)

	speedtest = True
	
	if speedtest:
		itertest = 1000

		start = time.time()
		for _ in range(itertest):
			sucess = False
			count = 0

			while not sucess: # sometimes had some move proposals that can fail (ie hops need specific site choices)
				site1 = random.randint(0, N)
				nn_list = lattice.list_viable_hop_sites(site1)
				if len(nn_list) == 0: 
					N_propose_fail += 1 # slowing down code a bit! 
					count += 1
				else:
					site2 = nn_list[random.randint(0, len(nn_list))]
					assert(lattice.occ_config[site1] != lattice.occ_config[site2])
					sucess = lattice.propose_hop(site1, site2)
				if count > 150:
					lattice.visualize()
					print('I got stuck...?')
					exit()

		print("move prop " , time.time() - start); 

		start = time.time()
		for _ in range(itertest):
			dE = lattice.get_dE_hop() # optimzied
		print("dE " , time.time() - start);  

		start = time.time()
		for _ in range(itertest):
			for layer in range(layers):
				op2x2[0, layer] = lattice.sublattice_2x2_order_param(layer) # care about mag rn
		print("op 2 " , time.time() - start); 

		start = time.time()
		for _ in range(itertest):
			for layer in range(layers):
				opr3xr3[0, layer] = lattice.sublattice_r3xr3_order_param(layer) # care about mag rn
		print("op 3 " , time.time() - start); 

		start = time.time()
		for _ in range(itertest):
			op_oop[0] = lattice.sublattice_oop_order_param()
		print("op oop " , time.time() - start); 
	
	exit()

def occhop_monte_carlo3d(layers, lattice, beta, x, warmup=0.5, record_interval=100, verbose=False, n_steps=1e3):

	N = lattice.size()
	# DIFFERENT FROM 2D
	Nwarmupsteps = int(n_steps*N*(warmup)) #3600
	Nrecordsteps = int(n_steps*N*(1-warmup)/record_interval)+1 #37

	op2x2 = np.zeros((Nrecordsteps,layers))
	opr3xr3 = np.zeros((Nrecordsteps,layers))
	op_oop = np.zeros((Nrecordsteps,1))

	e = np.zeros((int(n_steps*N*(1-warmup)),1))
	N_accept = 0
	N_reject = 0
	N_propose_fail = 0

	lattice.random_initialize_occ(x, verbose)	

	if np.mean(x) == 0 or np.mean(x) == 1:
		print('cant hop...lattice completely full or empty?')
		exit()

	record_step = 0
	for step in range(int(n_steps*N)):

		if verbose and (4*step)%(n_steps*N) == 0: 
			print('...{}% done'.format(100*step/(n_steps*N)))

		sucess = False
		count = 0

		while not sucess: # sometimes had some move proposals that can fail (ie hops need specific site choices)
			site1 = random.randint(0, N)
			nn_list = lattice.list_viable_hop_sites(site1)
			if len(nn_list) == 0: 
				N_propose_fail += 1
				count += 1
			else:
				site2 = nn_list[random.randint(0, len(nn_list))]
				assert(lattice.occ_config[site1] != lattice.occ_config[site2])
				sucess = lattice.propose_hop(site1, site2)
			if count > 150:
				lattice.visualize()
				print('I got stuck...?')
				exit()

		dE = lattice.get_dE_hop()
		r = random.rand()
		if r <= np.exp(-beta*dE): # do the metropolis hastings 
			lattice.accept_hop()
			N_accept += 1
		else: 
			lattice.reject_hop()
			N_reject += 1

		if step >= Nwarmupsteps and (step%100)==0:
			
			for layer in range(layers):
				op2x2[record_step, layer] = lattice.sublattice_2x2_order_param(layer) # care about mag rn
				opr3xr3[record_step, layer] = lattice.sublattice_r3xr3_order_param(layer) # care about mag rn
			
			op_oop[record_step] = lattice.sublattice_oop_order_param()
			e[record_step] = lattice.get_energy()
			
			record_step += 1
			if record_step == Nrecordsteps:
				return op2x2[:record_step,:], opr3xr3[:record_step,:], op_oop[:record_step], e[:record_step]


	return op2x2[:record_step,:], opr3xr3[:record_step,:], op_oop[:record_step], e[:record_step]



