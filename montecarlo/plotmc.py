
import matplotlib.pyplot as plt
import numpy as np

def plot_phase_space(op_matrix, kbts, lagrangemult, mult_label='x'):
	f, ax = plt.subplots(1,1)
	kbts_mg, lagrangemult_mg = np.meshgrid(kbts, lagrangemult)
	c = ax.pcolor(kbts_mg, lagrangemult_mg, np.transpose(op_matrix), cmap='RdBu')
	ax.set_ylabel(mult_label)
	ax.set_xlabel(r"$T(K)$")
	f.colorbar(c, ax=ax)
	plt.show()

def plot_phase_space_multops(op_matrix_lst, kbts, lagrangemult, mult_label='x'):
	f, axes = plt.subplots(1,len(op_matrix_lst)+1)
	kbts_mg, lagrangemult_mg = np.meshgrid(kbts, lagrangemult)
	cmaps = ['Blues','Oranges']
	for ax, op_matrix, cmap in zip(axes.flatten(), op_matrix_lst, cmaps):
		c = ax.pcolor(kbts_mg, lagrangemult_mg, np.transpose(op_matrix), cmap=cmap)
		ax.set_ylabel(mult_label)
		ax.set_xlabel(r"$T(K)$")
		f.colorbar(c, ax=ax)
	plot_phase_space_multop_bivariate(op_matrix_lst, kbts, lagrangemult, mult_label, ax=axes[-1])
	plt.show()

def plot_phase_space_multop_bivariate(op_matrix_lst, kbts, lagrangemult, mult_label='x', ax=None):
	assert(len(op_matrix_lst) == 2)
	op_matrix_R =  (op_matrix_lst[0])*lagrangemult
	op_matrix_B =  (op_matrix_lst[1])*lagrangemult
	colors = np.zeros((len(lagrangemult),len(kbts),3))
	op_matrix_R = op_matrix_R/np.max(op_matrix_R.flatten())
	op_matrix_B = op_matrix_B/np.max(op_matrix_B.flatten())
	colors[:,:,0] = 1 - np.transpose(op_matrix_R)
	colors[:,:,2] = 1 - np.transpose(op_matrix_B)
	colors[:,:,1] = 0.5 * (colors[:,:,0] + colors[:,:,2])
	colors[:,:,:] = colors[::-1,:,:]
	for i in range(3): colors[:,:,i] = colors[:,:,i]/np.max(colors[:,:,i].flatten())
	ax.imshow(colors)
	xtlabels = [300, 400, 500, 600, 700]
	ytlabels = [0.1, 0.2, 0.3, 0.4, 0.5]
	xtlocations = [(xtlabel - kbts[0])*(len(kbts)-1)/(kbts[-1]- kbts[0]) for xtlabel in xtlabels]
	ytlocations = [(ytlabel - lagrangemult[-1])*(len(lagrangemult)-1)/(lagrangemult[0]- lagrangemult[-1]) for ytlabel in ytlabels]
	ax.set(xticks=xtlocations, xticklabels=xtlabels);
	ax.set(yticks=ytlocations, yticklabels=ytlabels);
	ax.set_ylabel(mult_label)
	ax.set_xlabel(r"$T(K)$")

#path = "results/ZnBL_144.0/mc_summary.txt"
#path = "results/ScBL_144.0/mc_summary.txt"
path = "results/ZnBL_324.0/mc_summary.txt"

#  
xs = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35,
		   0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
kbts = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]

Nx = len(xs);
Nkbt = len(kbts);
op2 = np.zeros((Nx, Nkbt)) 
op3 = np.zeros((Nx, Nkbt)) 

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


kbts = [kbt * 298/25.7 for kbt in kbts]

#plot_phase_space(np.transpose(op2), kbts, xs, mult_label='x')
#plot_phase_space(np.transpose(op3), kbts, xs, mult_label='x')
plot_phase_space_multops([np.transpose(op2), np.transpose(op3)], kbts, xs, mult_label='x')




