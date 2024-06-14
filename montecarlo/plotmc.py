
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib import cm

def examine_local_op(op_matrix, kbts, lagrangemult):

	op_matrix = np.transpose(op_matrix)
	f, ax = plt.subplots(1,2)
	kbts_mg, lagrangemult_mg = np.meshgrid(kbts, lagrangemult)
	min_op = np.zeros((op_matrix.shape[0], op_matrix.shape[1]))

	cut_j = [5, 10, 15, 20, 25]
	colors = ['r', 'orange', 'c', 'b', 'purple']

	axis = ax[0]
	c = axis.pcolor(kbts_mg, lagrangemult_mg, op_matrix, cmap='Greys')
	for j, color in zip(cut_j, colors):
		ax[0].plot([kbts[j] for _ in np.arange(0,0.5,0.01)], np.arange(0,0.5,0.01), c=color)

	#ax[0].plot(np.arange(0,2000), [1/3 for _ in np.arange(0,2000)], 'k')
	axis.set_ylabel('x')
	axis.set_xlabel(r"$T(K)$")
	axis.set_ylim([0.1,0.4])
	axis.set_xlim([400,2000])
	#f.colorbar(c, ax=axis)
	axis.set_title(r'$\langle\bar{\alpha}\rangle$')

	ax[1].plot(lagrangemult, [-x/(1-x) for x in lagrangemult], c='grey')

	#ax[1].plot([1/3 for _ in np.arange(-2, 0.5, 0.01)], np.arange(-2, 0.5, 0.01), 'k')

	ax[1].plot(lagrangemult, [0 for x in lagrangemult], c='grey')
	for j, color in zip(cut_j, colors):
		ax[1].plot(lagrangemult, op_matrix[:,j], c=color)
	ax[1].set_xlabel('x')
	ax[1].set_xlim([0.1,0.4])
	ax[1].set_ylim([-1.1,0.1])
	ax[1].set_ylabel(r'$\langle\bar{\alpha}\rangle$')


def plot_phase_space(op_matrix, kbts, lagrangemult, mult_label='x'):
	f, ax = plt.subplots(1,1)
	kbts_mg, lagrangemult_mg = np.meshgrid(kbts, lagrangemult)
	c = ax.pcolor(kbts_mg, lagrangemult_mg, np.transpose(op_matrix), cmap='RdBu')
	ax.set_ylabel(mult_label)
	ax.set_xlabel(r"$T(K)$")
	f.colorbar(c, ax=ax)
	plt.show()

def plot_phase_space_multops(op_matrix_lst, kbts, lagrangemult, mult_label='x', axes=None):
	kbts_mg, lagrangemult_mg = np.meshgrid(kbts, lagrangemult)
	cmaps = ['Blues','Oranges','Greys']
	cmap_lims = [ [0, 0.25], [0, np.sqrt(3/8)], [-0.5, 0] ]
	plot_phase_space_multop_bivariate(op_matrix_lst, kbts, lagrangemult, mult_label, ax=axes[0])
	i = 0
	for ax, op_matrix, cmap in zip(axes[1:].flatten(), op_matrix_lst, cmaps):
		c = ax.pcolor(kbts_mg, lagrangemult_mg, np.transpose(op_matrix), cmap=cmap, vmin=cmap_lims[i][0], vmax=cmap_lims[i][1])
		ax.set_ylabel(mult_label)
		ax.set_xlabel(r"$T(K)$")
		ax.set(xticks=[500, 1000, 1500, 2000], xticklabels=[500, 1000, 1500, 2000]);
		ax.set(yticks=[0.1, 0.2, 0.3, 0.4, 0.5], yticklabels=[0.1, 0.2, 0.3, 0.4, 0.5]);
		ax.set_ylim([0.1,0.4])
		f.colorbar(c, ax=ax, location='right')
		i += 1
	plt.subplots_adjust(wspace=0.4, top=0.845, bottom=0.34)	

def plot_phase_space_multop_bivariate(op_matrix_lst, kbts, lagrangemult, mult_label='x', ax=None):

	#assert(len(op_matrix_lst) == 2)
	op_matrix_R =  (op_matrix_lst[0])#*lagrangemult
	op_matrix_B =  (op_matrix_lst[1])#*lagrangemult
	colors = np.zeros((len(lagrangemult),len(kbts),3))
	
	op_matrix_R = op_matrix_R/0.66
	op_matrix_B = op_matrix_B/0.66

	colors[:,:,0] = 1 - np.transpose(op_matrix_R)
	colors[:,:,2] = 1 - np.transpose(op_matrix_B)
	colors[:,:,1] = 0.5 * (colors[:,:,0] + colors[:,:,2])
	colors[:,:,:] = colors[::-1,:,:]
	#for i in range(3): colors[:,:,i] = colors[:,:,i]/np.max(colors[:,:,i].flatten())
	ax.imshow(colors)
	xtlabels = [500, 1000, 1500, 2000]
	ytlabels = [0.1, 0.2, 0.3, 0.4]
	xtlocations = [(xtlabel - kbts[0])*(len(kbts)-1)/(kbts[-1]- kbts[0]) for xtlabel in xtlabels]
	ytlocations = [(ytlabel - lagrangemult[-1])*(len(lagrangemult)-1)/(lagrangemult[0]- lagrangemult[-1]) for ytlabel in ytlabels]
	ax.set(xticks=xtlocations, xticklabels=xtlabels);
	ax.set(yticks=ytlocations, yticklabels=ytlabels);
	ax.set_ylim([ytlocations[0],ytlocations[-1]])
	ax.set_ylabel(mult_label)
	ax.set_xlabel(r"$T(K)$")

def main(path, axes, local_only=False):
		#  
	xs = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35,
			   0.36, 0.37, 0.38, 0.39, 0.40]
	kbts = [30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170]

	Nx = len(xs);
	Nkbt = len(kbts);
	op2 = np.zeros((Nx, Nkbt)) 
	op3 = np.zeros((Nx, Nkbt)) 
	srop = np.zeros((Nx, Nkbt)) 

	with open(path) as file:
		nline = 0
		for line in file:
			nline += 1
			if nline == 1: continue
			tokens = line.split('\t')
			tokens = [tok.strip() for tok in tokens]
			tokens = [tok for tok in tokens if len(tok) > 0]
			if len(tokens) < 6: continue
			try:
				i = xs.index( float(tokens[0]) )
			except:
				continue
			j = kbts.index( float(tokens[1]) )
			op2[i,j] = float(tokens[2])
			op3[i,j] = float(tokens[5])
			srop[i,j] = float(tokens[8])

	kbts = [kbt * 298/25.7 for kbt in kbts] # unit change to K

	if local_only:
		examine_local_op(np.transpose(srop), kbts, xs)
		plt.show()
	else:
		plot_phase_space_multops([np.transpose(op2), np.transpose(op3), np.transpose(srop)], kbts, xs, mult_label='x', axes=axes)

if __name__ == '__main__':

	path = "{}/mc_summary.txt".format(sys.argv[1])
	main(path, None, True)
	exit()

	"""
	f, axes = plt.subplots(1, 4, figsize=(12,6))
	path = "{}/mc_summary.txt".format(sys.argv[1])
	main(path, axes)
	plt.show()

	"""
	f, axes = plt.subplots(2,4, figsize=(12,6))

	path = "{}/mc_summary.txt".format(sys.argv[1])
	main(path, axes[0,:])

	path = "{}/mc_summary.txt".format(sys.argv[2])
	main(path, axes[1,:])

	axes[0,0].set_xlabel('')
	axes[0,0].set(xticks=[])
	axes[0,0].set_title(r'$Zn_{x}Ta_2S_4$')

	axes[1,0].set_title(r'$Sc_{x}Ta_2S_4$')

	axes[0,1].set_ylabel('')
	axes[0,1].set_xlabel('')
	axes[0,1].set(xticks=[], yticks=[])
	axes[0,1].set_title(r'$\langle RMS(\gamma_i)\rangle$')
	
	axes[0,2].set_ylabel('')
	axes[0,2].set_xlabel('')
	axes[0,2].set(xticks=[], yticks=[])
	axes[0,2].set_title(r'$\langle RMS(\phi_i)\rangle$')

	axes[0,3].set_ylabel('')
	axes[0,3].set_xlabel('')
	axes[0,3].set(xticks=[], yticks=[])
	axes[0,3].set_title(r'$\langle\bar{\alpha}\rangle$')

	axes[1,1].set_ylabel('')
	axes[1,1].set(yticks=[])
	axes[1,1].set_title(r'$\langle RMS(\gamma_i)\rangle$')

	axes[1,2].set_ylabel('')
	axes[1,2].set(yticks=[])
	axes[1,2].set_title(r'$\langle RMS(\phi_i)\rangle$')

	axes[1,3].set_ylabel('')
	axes[1,3].set(yticks=[])
	axes[1,3].set_title(r'$\langle\bar{\alpha}\rangle$')

	plt.show()


	        


