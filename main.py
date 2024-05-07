
import numpy as np  
import numpy.random as random
import matplotlib.pyplot as plt
from io_utils import plot_and_write
from lattices import RigidHexagonalLattice, RigidHexagonalLattice3D
from occ_lattice import HexagonalOccLattice, OccupationalLattice
from visualize import plot_phase_space, plot_phase_space_multops, plot_pt_look
from drivers import occhop_monte_carlo3d, occhop_monte_carlo
import matplotlib.pyplot as plt
import pickle
import time

def binder_coef(lst):
	avg = np.mean(lst)
	op2 = np.mean(np.array([(op - avg) ** 2 for op in lst]))
	op4 = np.mean(np.array([(op - avg) ** 4 for op in lst]))
	return 1 - op4/(3 * op2**2)

def rolling_binder(lst):
	rolling_b = np.zeros((len(lst), 1))
	for i in range(len(lst)):
		rolling_b[i] = binder_coef(lst[:i])
	return rolling_b

def rolling_average(lst):
	rolling_average = np.zeros((len(lst), 1))
	for i in range(len(lst)):
		rolling_average[i] = np.mean(lst[:i])
	return rolling_average

def looking_for_phase_trans():

	# Zn bilayer
	J1 = 205.8
	J2 = 5.9
	Jr3 = 19.1
	Jc = J1c = Jr3c = 0
	N = [6,6,1]
	a0 = 3.36
	c = 12

	STEPS = 1000
	warmup=0.7
	record_interval=500

	kbts = np.arange(50,100,1)
	x = [0.75]

	def ising_test(r, a0=a0, c=c): 
		if np.round(r, 2) == np.round(a0, 2):            			 return J1
		if np.round(r, 2) == np.round(np.sqrt(3)*a0, 2): 			 return Jr3
		if np.round(r, 2) == np.round(2*a0, 2):          			 return J2
		if np.round(r, 2) == np.round(c, 2):             		     return Jc
		if np.round(r, 2) == np.round(np.sqrt(3*(a0**2) + c**2), 2): return J1c
		if np.round(r, 2) == np.round(np.sqrt(a0**2 + c**2), 2):     return Jr3c
		return 0 #no couple

	layers = N[2]
	lattice = RigidHexagonalLattice3D( N[0], N[1], N[2], a0, c) #redo
	op22_matrix = np.zeros((len(kbts), len(x), layers)) 
	op33_matrix = np.zeros((len(kbts), len(x), layers)) 
	op_oop_matrix = np.zeros((len(kbts), len(x))) 

	start = time.time()
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) #redo
	print("ham init " , time.time() - start, " sec"); 

	Ncombo = len(kbts) * len(x)
	count = 0
	print("will run for {} T,x combos".format(Ncombo))
	
	# construct phase space
	startwholeloop = time.time()
	for i in range(len(kbts)):
		for j in range(len(x)):
			fdone = np.ravel_multi_index((i,j), (len(kbts), len(x)))/(len(kbts) * len(x))
			# print('kbt ', kbts[i], 'x ', x[j])
			if j == 0: print('{}% done'.format(round(100*fdone,1)))
			
			start = time.time()
			op22_trace, op33_trace, op_oop_trace, e_trace = occhop_monte_carlo3d(layers, lattice, 1/kbts[i], [x[j], x[j], x[j], x[j], x[j], x[j]], warmup, record_interval, n_steps=STEPS, verbose=False)
			count += 1
			timeelapse = time.time() - start
			if j == 0: print("mc run " , timeelapse); 
			if j == 0: print("  time left around {} min".format(timeelapse/60 * (Ncombo - count)))

			for layer in range(layers):
				avg_op22 = np.mean(op22_trace[:, layer])
				avg_op33 = np.mean(op33_trace[:, layer])
				op22_matrix[i,j,layer] = np.abs(avg_op22)
				op33_matrix[i,j,layer] = np.abs(avg_op33)
			op_oop_matrix[i,j] = np.abs(np.mean(op_oop_trace))
		
	print("whole loop " , (time.time() - startwholeloop)/60, " min"); 
	pickle.dump( [op22_matrix, op33_matrix, op_oop_matrix, kbts, x], open("ptscan_save{}{}{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(N[0], N[1], N[2], STEPS, J1, Jr3, J2, Jc, J1c, Jr3c), "wb" ))
		
	plot_pt_look(op22_matrix, op33_matrix, op_oop_matrix, kbts)
	
def main_occupancy_3d():

	# Only hop within plane ready!

	# these dont matter much:
	a0 = 3.36
	c = 12

	"""
	# Sc bilayer
	J1  = 281.0;
	J2  = 6.0;
	Jr3 = 19.1;
	Jc = J1c = Jr3c = 0
	N = [6,6,1]
	
	# Sc Bulk
	J1 = 268.0
	J2 = 16.3
	Jr3 = 16.7
	Jc = -55.8
	J1c = 40.0
	Jr3c = -26.8
	N = [6,6,2]
	
	# Zn Bulk
	J1 = 192.8
	J2 = 5.3
	Jr3 = 15.3
	Jc = 38.2
	J1c = 19.2
	Jr3c = 5.3
	N = [6,6,6]
	"""
	
	# Zn bilayer
	J1 = 205.8
	J2 = 5.9
	Jr3 = 19.1
	Jc = J1c = Jr3c = 0
	N = [12,12,1]	

	warmup = 0.5 # percent dont record for
	STEPS = 4 * 1e5 # really sweeps do N*this
	record_interval = N[1]*N[2] # set to N for once per sweep

	#kbts = np.arange(10,300,5)
	#x = [el/(N[0]*N[1]) for el in np.arange(3,N[0]*N[1])] # need to be evenly divisible by number of layers

	x = [0.25]
	kbts = [37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5]

	def ising_test(r, a0=a0, c=c): 
		if np.round(r, 2) == np.round(a0, 2):            			 return J1
		if np.round(r, 2) == np.round(np.sqrt(3)*a0, 2): 			 return Jr3
		if np.round(r, 2) == np.round(2*a0, 2):          			 return J2
		if np.round(r, 2) == np.round(c, 2):             		     return Jc
		if np.round(r, 2) == np.round(np.sqrt(3*(a0**2) + c**2), 2): return J1c
		if np.round(r, 2) == np.round(np.sqrt(a0**2 + c**2), 2):     return Jr3c
		return 0 #no couple

	layers = N[2]
	lattice = RigidHexagonalLattice3D( N[0], N[1], N[2], a0, c) #redo

	op22_matrix = np.zeros((len(kbts), len(x), layers)) 
	op33_matrix = np.zeros((len(kbts), len(x), layers)) 
	bop22_matrix = np.zeros((len(kbts), len(x), layers)) 
	bop33_matrix = np.zeros((len(kbts), len(x), layers)) 
	op_oop_matrix = np.zeros((len(kbts), len(x))) 
	bop_oop_matrix = np.zeros((len(kbts), len(x))) 

	start = time.time()
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) #redo
	print("ham init " , time.time() - start, " sec"); 

	Ncombo = len(kbts) * len(x)
	count = 0
	print("will run for {} T,x combos".format(Ncombo))
	
	# construct phase space
	startwholeloop = time.time()
	for i in range(len(kbts)):
		for j in range(len(x)):
			
			fdone = np.ravel_multi_index((i,j), (len(kbts), len(x)))/(len(kbts) * len(x))
			# print('kbt ', kbts[i], 'x ', x[j])
			if j == 0: print('{}% done'.format(round(100*fdone,1)))
			
			start = time.time()
			op22_trace, op33_trace, op_oop_trace, e_trace = occhop_monte_carlo3d(layers, lattice, 1/kbts[i], [x[j], x[j], x[j], x[j], x[j], x[j]], warmup, record_interval, n_steps=STEPS, verbose=False)
			count += 1
			timeelapse = time.time() - start
			if j == 0: print("mc run " , timeelapse); 
			if j == 0: print("  time left around {} min".format(timeelapse/60 * (Ncombo - count)))

			start = time.time()
			for layer in range(layers):
				avg_op22 = np.mean(op22_trace[:, layer])
				avg_op33 = np.mean(op33_trace[:, layer])
				op22_matrix[i,j,layer] = np.abs(avg_op22)
				op33_matrix[i,j,layer] = np.abs(avg_op33)
	
			print("avg 1 " , time.time() - start, " sec"); 

			start = time.time()
			
			for layer in range(layers):
				bop22_matrix[i,j,layer] = binder_coef(op22_trace[:,layer])
				bop33_matrix[i,j,layer] = binder_coef(op33_trace[:,layer])
				print("binder 2x2 layer {} = {}".format(layer, bop22_matrix[i,j,layer]))
				print("binder 3x3 layer {} = {}".format(layer, bop33_matrix[i,j,layer]))

			if layers > 1:
				op_oop_matrix[i,j] = np.abs(np.mean(op_oop_trace))
				bop_oop_matrix[i,j] = binder_coef(op_oop_trace)

			print("binder comp" , time.time() - start, " sec"); 

			f, ax = plt.subplots(3,1)
			ax = ax.flatten()
			start = time.time()
			ax[0].plot(op22_trace[:,0], c='grey')
			ax[0].plot(rolling_average(op22_trace[:,0]), c='k') 
			ax[0].set_title('2x2 OP (B={})'.format(round(bop22_matrix[i,j,0],4)))
			ax[1].plot(op33_trace[:,0], c='grey')
			ax[1].plot(rolling_average(op33_trace[:,0]), c='k') 
			ax[1].set_title('3x3 OP (B={})'.format(round(bop33_matrix[i,j,0],4)))
			ax[2].plot(e_trace, c='grey')
			ax[2].plot(rolling_average(e_trace[:,0]), c='k') 
			ax[2].set_title('Energy')

			start = time.time()
			titleplt = "kbt_{}_X_{}_STEPS_{}_WU_{}_RI_{}_totN_{}.png".format(kbts[i], x[j], STEPS, warmup, record_interval, N[0]*N[1]*N[2])
			plt.savefig(titleplt)
			print("plot save of {}".format(titleplt) , time.time() - start, " sec"); 

			#lattice.visualize(); exit()
			
	print("whole loop " , (time.time() - startwholeloop)/60, " min"); 
			
	pickle.dump( [op22_matrix, bop22_matrix, op33_matrix, bop33_matrix, op_oop_matrix, bop_oop_matrix, kbts, x], open("save{}{}{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(N[0], N[1], N[2], STEPS, J1, Jr3, J2, Jc, J1c, Jr3c), "wb" ))
	
	for layer in range(layers):
		plot_phase_space_multops([op22_matrix[:,:,layer], op33_matrix[:,:,layer]], kbts, x, 'x')

	plot_phase_space(op_oop_matrix, kbts, x, 'x')

def plot_dft_energies():

	N = [3,3,1] 
	a0, c = 3.36266792, 12 # lattice constants (hexagonal)
	latvec = np.array([ np.array([a0, 0, 0]), np.array([-a0/2, np.sqrt(3)/2*a0, 0]), np.array([0,0,c]) ])

	def ising_test(rvec, a0=a0): 
		r = np.sum([rel**2 for rel in rvec]) ** 0.5
		if np.round(r, 2) == np.round(a0, 2):            return 1.0
		if np.round(r, 2) == np.round(np.sqrt(3)*a0, 2): return 0.0
		return 0 #no couple

	f, ax = plt.subplots(1,1)
	
	#0 Done
	lattice = OccupationalLattice([3,3,1], latvec, a0, c)
	lattice.initialize_occ([1,0,0,0,0,0,0,0,0])
	lattice.setup_occ_energetics_general(ising_test, image_rep=[2,2,1])
	#ax.scatter(lattice.interc_density(), lattice.energy, c='k')
	ax.scatter(lattice.interc_density(), -444.98590046/9, c='r')

	plt.show()

def test_occupancy_energetics_3d():

	mu = -2.63860
	E0 = -49.18652
	a0, c = 3.6, 12 # lattice constants (hexagonal)
	latvec = np.array([ np.array([a0, 0, 0]), np.array([-a0/2, np.sqrt(3)/2*a0, 0]), np.array([0,0,c]) ])

	J1 = 192.778 * 1e-3
	J2 = 5.2711 * 1e-3
	Jr3 = 15.2618 * 1e-3
	Jc = 38.172 * 1e-3
	J1c = 19.204 * 1e-3
	Jr3c = 5.295 * 1e-3

	def ising_test(rvec, a0=a0): 
		#r = np.sum([rel**2 for rel in rvec]) ** 0.5
		r = rvec
		if np.round(r, 2) == np.round(a0, 2):            			 return J1
		if np.round(r, 2) == np.round(np.sqrt(3)*a0, 2): 			 return Jr3
		if np.round(r, 2) == np.round(2*a0, 2):          			 return J2
		if np.round(r, 2) == np.round(c, 2):             		     return Jc
		if np.round(r, 2) == np.round(np.sqrt(3*(a0**2) + c**2), 2):   return J1c
		if np.round(r, 2) == np.round(np.sqrt(a0**2 + c**2), 2):     return Jr3c
		return 0 #no couple

	"""
	lattice = OccupationalLattice([6,6,2], latvec, a0, c)
	lattice.random_initialize_occ(x=0.5)
	lattice.setup_occ_energetics_general(ising_test, image_rep=[1,1,1], chempot=mu, offset=E0*4)
	print('energy is ', lattice.energy)
	lattice.setup_occ_energetics_general(ising_test, image_rep=[1,1,2], chempot=mu, offset=E0*4)
	print('energy is ', lattice.energy)
	lattice.setup_occ_energetics_general(ising_test, image_rep=[2,2,2], chempot=mu, offset=E0*4)
	print('energy is ', lattice.energy)
	exit()
	"""

	lattice = RigidHexagonalLattice3D( 3, 3, 2, a0, c)
	lattice.initialzie_occ([0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0])
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1], chempot=-1*mu)
	print('energy is ', lattice.energy + 9*E0)
	print('expect energy is ', -442.678667641142)

	lattice = RigidHexagonalLattice3D( 3, 3, 2, a0, c)
	lattice.initialzie_occ([1,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0])
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1], chempot=-1*mu)
	print('energy is ', lattice.energy + 9*E0)
	print('expect energy is ', -445.317268181576)

	lattice = RigidHexagonalLattice3D( 3, 3, 2, a0, c)
	lattice.initialzie_occ([1,1,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0])
	print(lattice.site_index_to_nm(0))
	print(lattice.site_index_to_nm(1))
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1], chempot=-1*mu)
	print('energy is ', lattice.energy + 9*E0)
	print('expect energy is ', -447.803176281739)


	"""
	lattice = OccupationalLattice([3,3,2], latvec, a0, c)
	lattice.initialzie_occ([0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0])
	lattice.setup_occ_energetics_general(ising_test, image_rep=[3,3,3], chempot=mu, offset=9*E0)
	print('energy is ', lattice.energy)
	print('expect energy is ', -442.678667641142)

	lattice = OccupationalLattice([3,3,2], latvec, a0, c)
	lattice.initialzie_occ([1,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0])
	lattice.setup_occ_energetics_general(ising_test, image_rep=[3,3,3], chempot=mu, offset=9*E0)
	print('energy is ', lattice.energy)
	print('expect energy is ', -445.317268181576)

	lattice = OccupationalLattice([3,3,2], latvec, a0, c)
	lattice.initialzie_occ([1,1,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0]) # this is one in 0,0,0 and one in 0,0,c
	#print(lattice.site_index_to_nm(0)) -> 000
	#print(lattice.site_index_to_nm(1)) -> 001 -> 0*a1 + 0*a2 + 1*c
	lattice.setup_occ_energetics_general(ising_test, image_rep=[3,3,3], chempot=mu, offset=9*E0)
	print('energy is ', lattice.energy)
	print('expect energy is ', -447.803176281739)
	"""

def test_occupancy_energetics_2d():

	mu = -2.756
	E0 = -49.131
	a0, c = 3.6, 12 # lattice constants (hexagonal)
	latvec = np.array([ np.array([a0, 0, 0]), np.array([-a0/2, np.sqrt(3)/2*a0, 0]), np.array([0,0,c]) ])

	def ising_test(rvec, a0=a0): 
		# r = np.sum([rel**2 for rel in rvec]) ** 0.5
		r = rvec
		if np.round(r, 2) == np.round(a0, 2):            return 0.206
		if np.round(r, 2) == np.round(np.sqrt(3)*a0, 2): return 0.019
		if np.round(r, 2) == np.round(2*a0, 2):          return 0.006
		return 0 #no couple

	#lattice = OccupationalLattice([1,1,1], latvec, a0, c)
	lattice = RigidHexagonalLattice( 1, 1, a0 )
	lattice.initialzie_occ([0])
	lattice.setup_occ_energetics(ising_test, image_rep=[6,6,1], chempot=-1*mu)
	print('energy is ', lattice.energy + E0)
	print('expect energy is ', -49.131)

	#lattice = OccupationalLattice([1,1,1], latvec, a0, c)
	lattice = RigidHexagonalLattice( 1, 1, a0 )
	lattice.initialzie_occ([1])
	lattice.setup_occ_energetics(ising_test, image_rep=[6,6,1], chempot=-1*mu)
	print('energy is ', lattice.energy + E0)
	print('expect energy is ', -50.501)

	lattice = RigidHexagonalLattice(2, 2, a0)
	lattice.initialzie_occ([1,0,0,0])
	lattice.setup_occ_energetics(ising_test, image_rep=[3,3,1], chempot=-1*mu)
	print('energy is ', lattice.energy + E0*4)
	print('expect energy is ', -199.243)
	exit()

	lattice = OccupationalLattice([2,2,1], latvec, a0, c)
	lattice.initialzie_occ([1,1,1,0])
	lattice.setup_occ_energetics_general(ising_test, image_rep=[3,3,1], chempot=mu, offset=E0*4)
	print('energy is ', lattice.energy)
	print('expect energy is ', -202.984)

	# TEST THAT image_rep [1,1] ok for [6,6]
	lattice = OccupationalLattice([6,6,1], latvec, a0, c)
	lattice.random_initialize_occ(x=0.5)

	lattice.setup_occ_energetics_general(ising_test, image_rep=[1,1,1], chempot=mu, offset=E0*4)
	print('energy is ', lattice.energy)

	lattice.setup_occ_energetics_general(ising_test, image_rep=[2,2,1], chempot=mu, offset=E0*4)
	print('energy is ', lattice.energy)

	lattice.setup_occ_energetics_general(ising_test, image_rep=[3,3,1], chempot=mu, offset=E0*4)
	print('energy is ', lattice.energy)

def verify_3d_working():

	STEPS = 5
	c = 60
	N = 12 # need to be mult of 2 and 3 for the order params using...
	
	# Zn
	J1 = 205.8
	J2 = 5.9
	Jr3 = 19.1
	a0 = 3.36 # doesnt really matter

	def ising_test(r, a0=a0): 
		if np.round(r, 2) == np.round(a0, 2): return J1 #15
		if np.round(r, 2) == np.round(np.sqrt(3)*a0, 2): return Jr3 #1
		if np.round(r, 2) == np.round(2*a0, 2): return J2 #0
		return 0 #no couple	

	#### TEST 1

	lattice = RigidHexagonalLattice3D( 3, 3, 2, a0, c)
	lattice.initialzie_occ([1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0])
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) 
	print(lattice.energy)
	sucess = lattice.propose_hop(6, 10)
	n1,m1,p1 = lattice.site_index_to_nm(6)
	n2,m2,p2 = lattice.site_index_to_nm(10)
	print('proposed from {},{},{} to {},{},{}'.format(n1,m1,p1,n2,m2,p2))
	dE = lattice.get_dE_hop()
	print(dE)
	#						0 1 2 3 4 5 6 7 8 9 10
	lattice.initialzie_occ([1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0])
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) 
	print(lattice.energy)

	lattice = RigidHexagonalLattice3D( 3, 3, 1, a0, c)
	lattice.initialzie_occ([1,1,1,1,0,0,0,0,0])
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) 
	print(lattice.energy)
	sucess = lattice.propose_hop(3, 5)
	n1,m1,p1 = lattice.site_index_to_nm(3)
	n2,m2,p2 = lattice.site_index_to_nm(5)
	print('proposed from {},{},{} to {},{},{}'.format(n1,m1,p1,n2,m2,p2))
	dE = lattice.get_dE_hop()
	print(dE)
	lattice.initialzie_occ([1,1,1,0,0,1,0,0,0])
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) 
	print(lattice.energy)

	#### TEST 2

	lattice = RigidHexagonalLattice3D( N, N, 1, a0, c) 
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) 
	op22_trace, op33_trace, _, e_trace = occhop_monte_carlo3d(1, lattice, 1/10, [1/3], n_steps=STEPS, verbose=True)

	lattice = RigidHexagonalLattice3D( N, N, 2, a0, c) #redo
	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) #redo
	op22_trace, op33_trace, _, e_trace = occhop_monte_carlo3d(1, lattice, 1/10, [1/3,0], n_steps=STEPS, verbose=True)

	##### Test 3
	# run main_occupancy_2d with 2 layers and make it [x, 0] so second layer has nothing

def main_occupancy_2d():

	STEPS = 5
	
	c = 12
	N = 6 # need to be mult of 2 and 3 for the order params using...
	
	# Zn
	J1 = 205.8
	J2 = 5.9
	Jr3 = 19.1
	a0 = 3.36 # doesnt really matter

	layers = 1

	# Sc
	#J1  = 281.0;
	#J2  = 6.0;
	#Jr3 = 19.1;
	#a0  = 3.30; # doesnt really matter

	kbts = np.arange(10,200,10)
	x = [el/(N*N) for el in np.arange(3,N*N)]
	
	lattice = RigidHexagonalLattice3D( N, N, layers, a0, c) #redo
	op22_matrix = np.zeros((len(kbts), len(x))) 
	op33_matrix = np.zeros((len(kbts), len(x))) 

	def ising_test(r, a0=a0): 
		if np.round(r, 2) == np.round(a0, 2): return J1 #15
		if np.round(r, 2) == np.round(np.sqrt(3)*a0, 2): return Jr3 #1
		if np.round(r, 2) == np.round(2*a0, 2): return J2 #0
		return 0 #no couple	

	lattice.setup_occ_energetics(ising_test, image_rep=[1,1,1]) #redo
	
	# construct phase space
	for i in range(len(kbts)):
		for j in range(len(x)):
			
			fdone = np.ravel_multi_index((i,j), (len(kbts), len(x)))/(len(kbts) * len(x))
			print('kbt ', kbts[i], 'x ', x[j])
			if j == 0: print('{}% done'.format(round(100*fdone,1)))
			
			op22_trace, op33_trace, _, e_trace = occhop_monte_carlo3d(1, lattice, 1/kbts[i], [x[j]], n_steps=STEPS, verbose=False)
			cutoff = int(STEPS/2) * lattice.size() # ignore equilib in stat collection... check conv plots to make sure ok choice
			avg_op22 = np.mean(op22_trace[cutoff:,0])
			avg_op33 = np.mean(op33_trace[cutoff:,0])
			op22_matrix[i,j] = np.abs(avg_op22) # for testing vis used 2*np.abs((0.5-x[j]))
			op33_matrix[i,j] = np.abs(avg_op33) # for testing vis used x[j]
			
			#op22_matrix[i,j] = 2*np.abs((0.5-x[j]))
			#op33_matrix[i,j] = x[j]

	plot_phase_space_multops([op22_matrix, op33_matrix], kbts, x, 'x')
	pickle.dump( [op22_matrix, op33_matrix, kbts, x], open("save{}_{}_{}_{}_{}_{}.pkl".format(N, STEPS, a0, J1, Jr3, J2), "wb" ))

####################################################################################
# a note on UNITS - 
# all distances in Å
# all energies in meV : ( kb = 25.7/298 meV --> room T is 0.08624 )
# use dft lattice constant for bilayer 2h-TaS2 of 3.3006Å
# use kf sensible from fit to dft data : 
# 	note: 0.535 invÅ from arpes of bulk FexTaS2 [Ko et al PRL]
####################################################################################
def main_magnetic_2d():

	STEPS = 10 #under converged for testing, would need more like 100+

	kbts = np.arange(2,62,2)
	fields = np.arange(0,26,1)
	
	lattice = RigidHexagonalLattice(10, 10, a0=np.sqrt(3)*3.3006, superlattice=1)
	op_matrix = np.zeros((len(kbts), len(fields))) 

	def rkky_coupling(r, normfactor=1): 
		if r == 0: return 0
		kf = 0.5284# 0.6 # inverse angstroms
		a0=3.3006
		Jrkky = (np.sin(kf*2*r)/(r**2) + (-13/(16*kf))*np.cos(kf*2*r)/(r**3)) # 2d rkky
		R = normfactor*a0
		Jrkky_norm_to = (np.sin(kf*2*R)/(R**2) + (-13/(16*kf))*np.cos(kf*2*R)/(R**3))
		return Jrkky/np.abs(Jrkky_norm_to) # in meV

	if True:
		a0=3.3006#*np.sqrt(3)
		J1 = rkky_coupling(a0)
		Jr3 = rkky_coupling(np.sqrt(3)*a0)
		J2 = rkky_coupling(2*a0)
		Jr7 = rkky_coupling(np.sqrt(7)*a0)
		J3 = rkky_coupling(3*a0)
		print(J1, Jr3, J2, Jr7, J3)
		exit()

	# will be way slower ising than expected since dense mat mult, use for debug not practical use
	def ising_test(r, a0=3.3006): 
		if np.round(r, 2) == np.round(a0, 2): return 1 #FM
		if np.round(r, 2) == np.round(np.sqrt(3)*a0, 2): return -1.1 #FM
		if np.round(r, 2) == np.round(2*a0, 2): return 0.44 #FM
		if np.round(r, 2) == np.round(2*a0, 2): return 0.42 #FM
		if np.round(r, 2) == np.round(2*a0, 2): return -0.39 #FM
		return 0 #no couple	

	# increase image_rep until convergence of effective Ham. need more for smaller unitcells.
	# note : must have at least 1 for PBC to be properly dealt with
	# otherwise, image rep of 5 unit cells (each N by M superlattice) would here (N=M=10) yeild
	# a length cutoff of 5 * 10 * a0 = 50a0 = 165Å. Seems like overkill but not, RKKY slowly convergent. 
	# if converged with image_rep of 5 for a 10x10 lattice, would need 10 reps for a 5x5 etc.
	#lattice.check_convergence_effectiveham(rkky_coupling, range_check = [1, 5, 10])
	#exit()

	# once have decent image repitition number
	#lattice.setup_mag_energetics(ising_test, Hfield=0, image_rep=[1,1]) #H=0 is field not hamiltonian
	lattice.setup_mag_energetics(rkky_coupling, Hfield=0, image_rep=[5,5]) #H=0 is field not hamiltonian
	
	# construct phase space
	for i in range(len(kbts)):
		for j in range(len(fields)):

			fdone = np.ravel_multi_index((i,j), (len(kbts), len(fields)))/(len(kbts) * len(fields))
			if j == 0: print('{}% done'.format(round(100*fdone,1)))
			op_trace, e_trace = spinflip_monte_carlo(lattice, 1/kbts[i], fields[j], n_steps=STEPS)
			cutoff = int(STEPS/2) * lattice.size() # ignore equilib in stat collection... check conv plots to make sure ok choice
			avg_op = np.mean(op_trace[cutoff:])
			op_matrix[i,j] = np.abs(avg_op)
			#lattice.visualize()
			#plot_and_write('spinflip_monte_carlo', betas[i], fields[j], lattice, op_trace, e_trace, avg_op, cutoff) 

	plot_phase_space(op_matrix, kbts, fields, 'H')

if __name__ == "__main__": 

	#test_occupancy_energetics_2d()
	#test_occupancy_energetics_3d()
	#verify_3d_working()

	#main_occupancy_2d()
	main_occupancy_3d() # use for 2d now too!
	
	#looking_for_phase_trans()	
	#[op22_matrix, op33_matrix, op_oop_matrix, kbts, x] = pickle.load( open("ptscan_save661_100_205.8_19.1_5.9_0_0_0.pkl", "rb"))
	#plot_pt_look(op22_matrix, op33_matrix, op_oop_matrix, kbts)

	#[op22_matrix, op33_matrix, op_oop_matrix, kbts, x] = pickle.load( open("save661_100_205.8_19.1_5.9_0_0_0.pkl", "rb"))
	#plot_phase_space_multops([op22_matrix[:,:,0], op33_matrix[:,:,0]], kbts, x, 'x')

	#[op22_matrix, op33_matrix, kbts, x] = pickle.load( open("save6_500_3.36_205.8_19.1_5.9.pkl", "rb"))
	#plot_phase_space_multops([op22_matrix, op33_matrix], kbts, x, 'x')

	#plot_dft_energies()


