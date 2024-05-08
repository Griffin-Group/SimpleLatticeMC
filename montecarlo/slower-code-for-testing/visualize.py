
import numpy as np  
import matplotlib.pyplot as plt

def plot_pt_look(op22_matrix, op33_matrix, op_oop_matrix, kbts, oop=False):
	f, ax = plt.subplots(1,1)
	ax.scatter( kbts.flatten(), op22_matrix.flatten(), c='r' )
	ax.scatter( kbts.flatten(), op33_matrix.flatten(), c='g' )
	if oop: ax.scatter( kbts.flatten(), op_oop_matrix.flatten(), c='b' )
	plt.show()


def plot_phase_space(op_matrix, kbts, lagrangemult, mult_label='H'):
	f, ax = plt.subplots(1,1)
	kbts_mg, lagrangemult_mg = np.meshgrid(kbts, lagrangemult)
	c = ax.pcolor(kbts_mg, lagrangemult_mg, np.transpose(op_matrix), cmap='RdBu')
	ax.set_ylabel(mult_label)
	ax.set_xlabel(r"$k_bT(meV)$")
	f.colorbar(c, ax=ax)
	plt.show()

def plot_phase_space_multop_bivariate(op_matrix_lst, kbts, lagrangemult, mult_label='H', ax=None):
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
	period_tick = 6;
	print(np.arange(len(kbts))[::period_tick])
	print([round(el,2) for el in kbts][::period_tick])
	ax.set(xticks=np.arange(len(kbts))[::period_tick], xticklabels=[round(el,2) for el in kbts][::period_tick]);
	ax.set(yticks=np.arange(len(lagrangemult))[::-1][::period_tick], yticklabels=[round(el,2) for el in lagrangemult][::period_tick]);
	ax.set_ylabel(mult_label)
	ax.set_xlabel(r"$k_bT(meV)$")

def plot_phase_space_multops(op_matrix_lst, kbts, lagrangemult, mult_label='H'):
	f, axes = plt.subplots(1,len(op_matrix_lst)+1)
	kbts_mg, lagrangemult_mg = np.meshgrid(kbts, lagrangemult)
	cmaps = ['Oranges', 'Blues']
	for ax, op_matrix, cmap in zip(axes.flatten(), op_matrix_lst, cmaps):
		c = ax.pcolor(kbts_mg, lagrangemult_mg, np.transpose(op_matrix)*lagrangemult_mg, cmap=cmap)
		ax.set_ylabel(mult_label)
		ax.set_xlabel(r"$k_bT(meV)$")
		f.colorbar(c, ax=ax)
	plot_phase_space_multop_bivariate(op_matrix_lst, kbts, lagrangemult, mult_label, ax=axes[-1])
	plt.show()