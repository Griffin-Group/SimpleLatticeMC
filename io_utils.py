
import pickle
from lattices import RigidHexagonalLattice
import matplotlib.pyplot as plt

####################################################################################
# output saving
####################################################################################
def plot_and_write(mctype, beta, H, lattice, op_trace, e_trace, avg_op, cutoff):

	x = lattice.interc_density()

	# save convergence plot
	f, ax = plt.subplots(1,2)
	ax[0].plot(op_trace)
	ax[0].axvline(cutoff, c='r')
	ax[0].set_title('order param convergence')
	ax[1].plot(e_trace)
	ax[1].axvline(cutoff, c='r')
	ax[1].set_title('energy convergence')
	convpng = 'convergence_beta{}_H{}_x{}.png'.format(beta, H, x)
	plt.savefig(convpng)

	# for saving lattice and trace
	picklepath = 'data_m{}_x{}_beta{}_H{}.pkl'.format(lattice.magnetization(), lattice.interc_density(), beta, H)
	with open(picklepath, 'wb') as file:
		pickle.dump([lattice, op_trace, avg_op, e_trace, beta, H, mctype], file)
	
	# save lattice plot
	latticepng = 'lattice_m{}_x{}_beta{}_H{}.png'.format(lattice.magnetization(), lattice.interc_density(), beta, H)
	lattice.visualize(latticepng)

	# book keep
	with open('{}_history.txt'.format(mctype), 'a') as f:
		f.write('\t'.join([str(i) for i in [x, beta, H, avg_op, '\n']]))

	plt.close('all')