import numpy as np
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import h5py


#import pdb; pdb.set_trace()


def InitialConditions(x, rho, v, Pressure, args, p):

	N = args.n
	dx = np.gradient(x)                     # Cell Sizes (UNIFORM GRID ONLY)
	M = (rho*dx).sum()                      # Total Mass (UNIFORM GRID ONLY)

	# Particle Parameters
	mp = M/N                         # Particle Mass, Total Mass divided by total particles
	p['mp'] = mp
	# Final number of particles per cell, and total
	Np = np.floor((rho*dx)/mp + np.random.uniform(size=rho.size)).astype(int)
	NP = Np.sum()
	p['NP'] = NP
	# Particle Diameter
	Dmax = (dx/Np).min()
	D = 1e-4*Dmax
	D = 0
	p['D'] = D

	#Initialize Velocities
	s = np.sqrt(Pressure/rho)


	# bin and effbin sizes
	# velocities
	effbins = np.array([])
	bins    = np.array([])
	effbins = []
	bins    = []

	vels = []
	for j in range(len(dx)):

	    effbins.append( np.ones(Np[j])*(dx[j]/Np[j]-D))
	    bins.append( np.ones(Np[j])*dx[j]/Np[j]   )
	    
	    vels.append(np.random.randn(Np[j])*s[j] + v[j])
	    
	#effbins = np.array(effbins)
	#bins = np.array(bins)
	#pos_in_bin = effbins*np.random.uniform(size=effbins.size)
	effbins = np.concatenate(effbins)
	bins = np.concatenate(bins)
	vels = np.concatenate(vels)
	uniforms = np.random.random(Np.sum())

	# Computing particle dx's
	if args.b == 0: #Periodic Boundary Conditions
		Pdx = np.roll(effbins*uniforms, -1) - effbins*uniforms + bins
		v0 = vels[0]
		P0 = uniforms[0]*effbins[0] + D/2
		vels = np.roll(vels, -1) - vels
		P = np.concatenate(([P0], P0 + np.cumsum(Pdx[:-1])))
	elif args.b == 1: #Reflective Boundary Conditions #NEED TO FIX VELOCITY
		Pdx = effbins[1:]*uniforms[1:] - effbins[:-1]*uniforms[:-1] + bins[:-1]
		vels = np.roll(vels, -1) - vels
		P0 = uniforms[0]*effbins[0] + D/2
		Pdx = np.append(np.array([P0]),Pdx)
		P = np.cumsum(Pdx)
	#MP = mp*np.arange(1,Np.sum())
	#rhoP = mp/Pdx

	return P, Pdx, vels, P0, p


# Computes Densities 
# Smooths gradients over Ns particles
def DumpDensity(Pdx, x, p, P, Ns=10**3, fname='Output/Density000.png'):

	plt.figure(0)
	plt.clf()
	j = int(np.floor(p['NP']/Ns))
	#Pdx_smooth = Ns*p['mp']/np.array([Pdx[i*Ns:(i+1)*Ns].sum() for i in range(j)])
	Pdx_smooth = 1*p['mp']/np.array([Pdx[i*Ns:(i+1)*Ns].mean() for i in range(j)])
	P_smooth = np.array([P[i*Ns] for i in range(j)])
	
	#plt.plot(P[::Ns], mp/(Pdx[::Ns][1:]+Pdx[::Ns][:-1]), label=r'$\rho_P$')
	plt.plot(x, p['rho'], label=r'$\rho$')
	plt.plot(P_smooth,Pdx_smooth, lw=1, label = r'$\rho_P$')
	plt.xlabel("Distance")
	plt.ylabel("Density")
	plt.grid(alpha=0.3)
	plt.ylim(0.9,1.1)
	plt.xlim(0,1)
	#plt.yticks([0,1/2,1,3/2,2])
	plt.xticks([0,1/4,2/4,3/4,1])
	plt.legend()
	sns.despine()
	plt.tight_layout()	
	plt.savefig(fname)

# Preview Phase Space
def DumpPhaseSpace(P, vels, Nc = 128, fname='Output/Phase000.png'):
	plt.figure(1)
	plt.clf()
	plt.hist2d(P,vels,bins=(Nc,Nc), range = [[0,1],[-4,4]])
	plt.xlabel("Distance")
	plt.ylabel("Velocity")
	plt.title("Phase Space")
	plt.tight_layout()
	plt.savefig(fname)


