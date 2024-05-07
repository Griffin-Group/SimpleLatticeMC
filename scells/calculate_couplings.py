
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
     simpl_lst = [round(x/a0,3) for x in lst if x < CUTOFF*a0]
     components, counts = np.unique(simpl_lst, return_counts=True)
     if calcJ: 
          J_rkky = np.sum([rkky(v) for v in lst if np.abs(v) > 0.1])
     else:
          J_rkky = None
     readable_aggregateJbreakdown = " + ".join(["{}*J({})".format(c,v) for c, v in zip(counts, components)])
     return readable_aggregateJbreakdown, J_rkky

def auto_generate_adjacency3d(a1, a2, a3, basis_vecs, a0=3.30060087, c=12):
     Natom = len(basis_vecs)
     aggBreakdown = dict()
     Jdict = dict()
     adjmat = np.zeros((Natom, Natom))
     for i in range(Natom):
          for j in range(Natom):
               if i < j: 
                    continue # symmetric
               else:
                    Ri = basis_vecs[i][0] * np.array(a1) +  basis_vecs[i][1] * np.array(a2) +  basis_vecs[i][2] * np.array(a3) 
                    Rj = basis_vecs[j][0] * np.array(a1) +  basis_vecs[j][1] * np.array(a2) +  basis_vecs[j][2] * np.array(a3) 
                    d, dists = min_image_dist3d(Ri, Rj, a1, a2, a3, threshold=MAXREP*a0)
                    key, J = turn_agg_breakdown_tostr(dists, a0)
                    if key not in aggBreakdown.keys():
                         aggBreakdown[key] = dists
                    index = np.where([k == key for k in aggBreakdown.keys()])[0]
                    adjmat[i,j] = index
                    adjmat[j,i] = index

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

def auto_generate_adjacency(a1, a2, basis_vecs, v1=None, v0=None, kf=None, a0=3.30060087):
     Natom = len(basis_vecs)
     aggBreakdown = dict()
     Jdict = dict()
     adjmat = np.zeros((Natom, Natom))
     for i in range(Natom):
          for j in range(Natom):
               if i < j: 
                    continue # symmetric
               else:
                    Ri = basis_vecs[i][0] * np.array(a1) +  basis_vecs[i][1] * np.array(a2) 
                    Rj = basis_vecs[j][0] * np.array(a1) +  basis_vecs[j][1] * np.array(a2) 
                    d, dists = min_image_dist(Ri, Rj, a1, a2, threshold=MAXREP*a0)
                    key, J = turn_agg_breakdown_tostr(dists, a0)
                    if key not in aggBreakdown.keys():
                         aggBreakdown[key] = dists
                    index = np.where([k == key for k in aggBreakdown.keys()])[0]
                    adjmat[i,j] = index
                    adjmat[j,i] = index

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

def main_CASM_files():
     print("########################")
     print("2121 supercell:")
     a1 = [  3.3006010056, 0.0000000000]
     a2 = [  0.0000000000, 5.7168083191]
     basis_vecs = [ [0.0, 0.0], [0.5, 0.5]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("3311 supercell:")
     a1 = [  5.7168083191, 0.0000000000]
     a2 = [ -2.8584041595,4.9509012329 ]
     basis_vecs = [ [0.0, 0.0], [1/3, 2/3], [2/3, 1/3]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("3131 supercell:")
     a1 = [  3.3006010056, 0.0000000000]
     a2 = [ -1.6503003443, 8.5752122765 ]
     basis_vecs = [ [0.0, 0.0], [1/3, 2/3], [2/3, 1/3]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("4221 supercell:")
     a1 = [  6.6012020111, 0.0000000000]
     a2 = [ -3.3006010056, 5.7168086371]
     basis_vecs = [ [0.0, 0.0], [0.0, 0.5], [0.0, 0.5], [0.5, 0.5]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("4141 supercell:")
     a1 = [ 3.3006010056, 0.0000000000]
     a2 = [ 0.0000000000, 11.4336166382]
     basis_vecs = [ [0.0, 0.0], [0.5, 0.25], [0.0, 0.5], [0.5, 0.75]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("4411 supercell:")
     a1 = [ 5.7168083191, 0.0000000000]
     a2 = [ 0.0000000000, 6.6012020111]
     basis_vecs = [ [0.0, 0.0], [0.5, 0.25], [0.0, 0.5], [0.5, 0.75]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("5151 supercell:")
     a1 = [ 3.3006010056, 0.0000000000]
     a2 = [ -1.6503003443, 14.2920211045 ]
     basis_vecs = [ [0.0, 0.0], [0.6, 0.2], [0.2, 0.4], [0.8, 0.6], [0.4, 0.8]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("5511 supercell:")
     a1 = [ 5.7168083191,  0.0000000000]
     a2 = [ -2.8584041162, 8.2515018464 ]
     basis_vecs = [ [0.0, 0.0], [0.6, 0.2], [0.2, 0.4], [0.8, 0.6], [0.4, 0.8]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("6161 supercell:")
     a1 = [ 3.3006010056, 0.0000000000]
     a2 = [ 0.0000000000, 17.1504249573]
     basis_vecs = [ [0.0, 0.0], [0.5, 1/6], [0, 1/3], [0.5, 0.5], [0, 2/3], [0.5, 5/6]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

     print("########################")
     print("9911 supercell:")
     a1 = [8.7325687408,         0.0000000000]
     a2 = [-1.8712648492,         9.7233775431]  
     basis_vecs = [[0.000000000,         0.000000000],     
     [0.666666687,         0.111111097],    
     [0.333333343,         0.222222209],    
     [0.000000000,         0.333333343],    
     [0.666666687,         0.444444418],    
     [0.333333313,         0.555555522],    
     [0.000000000,         0.666666687],    
     [0.666666687,         0.777777731],    
     [0.333333313,         0.888888896]]
     auto_generate_adjacency(a1, a2, basis_vecs, a0=3.30060087)

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

     """
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
     """

main()
