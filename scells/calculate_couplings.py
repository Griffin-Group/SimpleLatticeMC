import numpy as np

#                        r3       r7        r12    r13
EXPECTED_VALS = [0, 1, 1.732, 2, 2.646, 3, 3.464, 3.606, 4]
MAXREP = 10#50 # truncate at 4*a0
MAXV = 5#50
CUTOFF = 3 # in terms of a0 

def dist_calc(R1, R2): #default use L2 norm
     return np.sqrt((R1[0] - R2[0])**2 + (R1[1] - R2[1])**2)

def dist_calc3d(R1, R2): #default use L2 norm
     return np.sqrt((R1[0] - R2[0])**2 + (R1[1] - R2[1])**2 + (R1[2] - R2[2])**2)

def min_image_dist(r1, r2, a1, a2, threshold):
     min_equiv_dist = dist_calc(r1, r2)
     dists = []
     for N in np.arange(-MAXV, MAXV+1):
          for M in np.arange(-MAXV, MAXV+1):
               d = dist_calc(r1 + N*np.array(a1) +M*np.array(a2), r2)
               if d < threshold: dists.append(d)
               if d < min_equiv_dist: min_equiv_dist = d
     return min_equiv_dist, dists

def min_image_dist3d(r1, r2, a1, a2, a3, threshold):
     min_equiv_dist = dist_calc3d(r1, r2)
     dists = []
     for N in np.arange(-MAXV, MAXV+1):
          for M in np.arange(-MAXV, MAXV+1):
               for P in np.arange(-MAXV, MAXV+1):
                    d = dist_calc3d(r1 + N*np.array(a1) + M*np.array(a2) + P*np.array(a3), r2)
                    if d < threshold: dists.append(d)
                    if d < min_equiv_dist: min_equiv_dist = d
     return min_equiv_dist, dists

def rkky(r, v1, v0, kf): 
     Jrkky = v1 + v0 * (np.sin(kf*2*r)/(r**2) + (-13/(16*kf))*np.cos(kf*2*r)/(r**3))
     return Jrkky

def turn_agg_breakdown_tostr(lst, a0, calcJ=False):
     simpl_lst = [round(x/a0,2) for x in lst if x < CUTOFF*a0]
     components, counts = np.unique(simpl_lst, return_counts=True)
     if calcJ: 
          J_rkky = np.sum([rkky(v) for v in lst if np.abs(v) > 0.1])
     else:
          J_rkky = None
     readable_aggregateJbreakdown = " + ".join(["{}*J({})".format(c,v) for c, v in zip(counts, components)])
     return counts, components, readable_aggregateJbreakdown, J_rkky

def auto_generate_adjacency3d(a1, a2, a3, basis_vecs, a0=3.30060087, c=12, verbose=False):
     Natom = len(basis_vecs)
     aggBreakdown = dict()
     Jdict = dict()
     adjmat = np.zeros((Natom, Natom))
     sigmaHsigma = dict()
     for i in range(Natom):
          for j in range(Natom):
               
               Ri = basis_vecs[i][0] * np.array(a1) +  basis_vecs[i][1] * np.array(a2) +  basis_vecs[i][2] * np.array(a3) 
               Rj = basis_vecs[j][0] * np.array(a1) +  basis_vecs[j][1] * np.array(a2) +  basis_vecs[j][2] * np.array(a3) 
               d, dists = min_image_dist3d(Ri, Rj, a1, a2, a3, threshold=MAXREP*a0)
               counts, components, key, J = turn_agg_breakdown_tostr(dists, a0)
               if key not in aggBreakdown.keys(): aggBreakdown[key] = dists
               index = np.where([k == key for k in aggBreakdown.keys()])[0]
               adjmat[i,j] = index[0]
               adjmat[j,i] = index[0]
               for component, count in zip(components, counts):
                    if component in sigmaHsigma.keys():
                         sigmaHsigma[component] += count
                    else:
                         sigmaHsigma[component] = count

     if verbose:
          print("printing final adjacency matrix, in units of a0={}ang".format(a0))
          for i in range(Natom): print(adjmat[i,:])
          count = 0
          for k in aggBreakdown.keys(): 
               print("{} means {}".format(count,k))
               count+=1
          if kf != None: J_rkky_vals = []
          count = 0
          for k in aggBreakdown.keys(): 
               dists = aggBreakdown[k]
               if kf != None: 
                    J_rkky = np.sum([rkky(v, v1, v0, kf) for v in dists if np.abs(v) > 0.1])
                    print("{} has Jrkky of {} when kF={} inverse ang".format(count,J_rkky,kf))
                    J_rkky_vals.append(J_rkky)
               count+=1
          if kf != None: return J_rkky_vals
          print('sigma*H*sigma.T for this occupation sigma is the sum of all these')
          print('since this H shown is just the occupied portion of the H (can get if feed all of viable interc vectors)')
          print(sigmaHsigma)
          print("Total terms in sig*H*sig : J(r=?) \t\t count")
     for k in sigmaHsigma.keys(): print("{}\t\t{}".format(k, sigmaHsigma[k]))

def auto_generate_adjacency(a1, a2, basis_vecs, v1=None, v0=None, kf=None, a0=3.30060087,verbose=False):

     Natom = len(basis_vecs)
     aggBreakdown = dict()
     Jdict = dict()
     adjmat = np.zeros((Natom, Natom))

     sigmaHsigma = dict()

     for i in range(Natom):
          for j in range(Natom):
               
               Ri = basis_vecs[i][0] * np.array(a1) +  basis_vecs[i][1] * np.array(a2) 
               Rj = basis_vecs[j][0] * np.array(a1) +  basis_vecs[j][1] * np.array(a2) 
               d, dists = min_image_dist(Ri, Rj, a1, a2, threshold=MAXREP*a0)
               counts, components, key, J = turn_agg_breakdown_tostr(dists, a0)
               
               for component, count in zip(components, counts):
                    if component in sigmaHsigma.keys():
                         sigmaHsigma[component] += count
                    else:
                         sigmaHsigma[component] = count

               if key not in aggBreakdown.keys(): aggBreakdown[key] = dists
               index = np.where([k == key for k in aggBreakdown.keys()])[0]
               adjmat[i,j] = index[0]
               adjmat[j,i] = index[0]

     if verbose:
          print("printing final adjacency matrix, in units of a0={}ang".format(a0))
          for i in range(Natom): print(adjmat[i,:])
          count = 0
          for k in aggBreakdown.keys(): 
               print("{} means {}".format(count,k))
               count+=1
          if kf != None: J_rkky_vals = []
          count = 0
          for k in aggBreakdown.keys(): 
               dists = aggBreakdown[k]
               if kf != None: 
                    J_rkky = np.sum([rkky(v, v1, v0, kf) for v in dists if np.abs(v) > 0.1])
                    print("{} has Jrkky of {} when kF={} inverse ang".format(count,J_rkky,kf))
                    J_rkky_vals.append(J_rkky)
               count+=1
          if kf != None: return J_rkky_vals
          print('sigma*H*sigma.T for this occupation sigma is the sum of all these')
          print('since this H shown is just the occupied portion of the H (can get if feed all of viable interc vectors)')
          print(sigmaHsigma)
          print("Total terms in sig*H*sig : J(r=?) \t\t count")
     for k in sigmaHsigma.keys(): print("{}\t\t{}".format(k, sigmaHsigma[k]))
     
def main():

     print("########################")
     print("1x1 supercell:")
     a0 = 3.3 
     a1 = np.array([a0, 0.0])
     a2 = np.array([-a0/2, np.sqrt(3)*a0/2])
     basis_vecs = [ [0.0, 0.0] ]
     auto_generate_adjacency(a1, a2, basis_vecs, a0)

     print("########################")
     print("2x2 supercell:")
     a0 = 3.3 
     a1 = np.array([a0, 0.0])
     a2 = np.array([-a0/2, np.sqrt(3)*a0/2])
     a1 *= 2
     a2 *= 2
     basis_vecs = [ [0.0, 0.0], [0, 1/2], [1/2, 0], [1/2, 1/2]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0)

     print("########################")
     print("3x3 supercell:")
     a0 = 3.3 
     a1 = np.array([a0, 0.0])
     a2 = np.array([-a0/2, np.sqrt(3)*a0/2])
     a1 *= 3
     a2 *= 3
     basis_vecs = [ [0.0, 0.0], [0, 1/3], [0, 2/3], [1/3, 0], [1/3, 1/3], [1/3, 2/3], [2/3, 0], [2/3, 1/3], [2/3, 2/3]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0)

     print("########################")
     print("complete interc 3x3x2 supercell:")
     a0 = 3.3 
     c = 6
     a1 = np.array([a0, 0.0, 0.0])
     a2 = np.array([-a0/2, np.sqrt(3)*a0/2, 0.0])
     a3 = np.array([0.0, 0.0, c])
     a1 *= 3
     a2 *= 3
     a3 *= 2
     basis_vecs = [ [0, 0, 0],   [0, 1/3, 0],   [0, 2/3, 0],   [1/3, 0, 0],   [1/3, 1/3, 0],   [1/3, 2/3, 0],   [2/3, 0, 0],   [2/3, 1/3, 0], [2/3, 2/3, 0],
                    [0, 0, 0.5], [0, 1/3, 0.5], [0, 2/3, 0.5], [1/3, 0, 0.5], [1/3, 1/3, 0.5], [1/3, 2/3, 0.5], [2/3, 0, 0.5], [2/3, 1/3, 0.5], [2/3, 2/3, 0.5]]
     auto_generate_adjacency3d(a1, a2, a3, basis_vecs, a0, c)

if __name__ == '__main__': main()
