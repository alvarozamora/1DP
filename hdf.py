import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import h5py
sns.set_context("talk") #darkgrid, whitegrid, dark, white, ticks
parser = argparse.ArgumentParser(description='HDF5 Initial Condition Pipeline')
parser.add_argument('--file', type=str, default='particle/particle', help='Output: hdf base file path/name')
parser.add_argument('-n', type=int, default=10**6, help='Particle Number')
parser.add_argument('-c', type=int, default=2, help='Number of cores/files')
parser.add_argument('-b', type=int, default=0, help='Periodic (0), Reflective (1)')
args = parser.parse_args()
#import pdb; pdb.set_trace()


# Sound Wave
f = 1          # Degrees of Freedom
g = (f+2)/f    # Adiabatic Index
c2 = 1.0       # Squared Sound Speed
delta = 0.05   # Sound Wave - Perturbation Amplitude

# Setting up Grid
Nc = 512                                # Number of Cells
x = np.linspace(0+1/Nc/2,1-1/Nc/2,Nc)   # Grid Centers on (0,1)
v = delta*np.sin(4*np.pi*x)             # Bulk Velocity
rho = 1 - v                             # Density  
P = rho*c2                              # Pressure
#ied = c2/(g-1)                          # Thermal Energy
dx = np.gradient(x)                     # Cell Sizes (UNIFORM GRID ONLY)
M = (rho*dx).sum()                      # Total Mass (UNIFORM GRID ONLY)


# Particle Parameters
N = 10**6                         # Approximate number of particles
mp = M/N                         # Particle Mass
# Final number of particles per cell, total
Np = np.floor((rho*dx)/mp + np.random.uniform(size=rho.size)).astype(int)
NP = Np.sum()
# Particle Diameter
Dmax = (dx/Np).min()
D = 1e-4*Dmax

#Initialize Velocities
s = np.sqrt(2*P/rho)


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
	import pdb; pdb.set_trace()
	vels = np.roll(vels, -1) - vels
elif args.b == 1: #Reflective Boundary Conditions #NEED TO FIX VELOCITY
	Pdx = effbins[1:]*uniforms[1:] - effbins[:-1]*uniforms[:-1] + bins[:-1]
	vels = np.roll(vels, -1) - vels
	P0 = uniforms[0]*effbins[0] + D/2
	Pdx = np.append(np.array([P0]),Pdx)
P = np.cumsum(Pdx)
MP = mp*np.arange(1,Np.sum())
rhoP = mp/Pdx


# In[9]:


Np.sum(), Pdx.shape


# In[10]:


# Bin Check
print(P0, bins.sum(), effbins.sum(), 1-Np.sum()*D)


# In[11]:


# Computes Densities 
# Smooths gradients over Ns particles
sns.set_style("white")
Ns = 10**3
j = int(np.ceil(Np.sum()/Ns))
Pdx_smooth = Ns*mp/np.array([Pdx[i*Ns:(i+1)*Ns].sum() for i in range(j)])
P_smooth = np.array([P[i*Ns] for i in range(j)])

#plt.plot(P[::Ns], mp/(Pdx[::Ns][1:]+Pdx[::Ns][:-1]), label=r'$\rho_P$')
#plt.plot(P_smooth,Pdx_smooth, label = r'$\rho_p')
plt.plot(x,rho, label=r'$\rho$')
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
plt.savefig("Positions.png")


# Preview Phase Space
plt.figure()
plt.hist2d(P,vels,bins=(Nc,128), range = [[0,1],[-4,4]])
plt.xlabel("Distance")
plt.ylabel("Velocity")
plt.title("Phase Space")
plt.tight_layout()
plt.savefig("PhaseSpace.png")

#import pdb; pdb.set_trace()
Pdx = np.array_split(Pdx, args.c)
vels = np.array_split(vels, args.c)
parts = np.array([len(q) for q in Pdx]).astype(int)
print(f'Particle 1 : ({P0:.3e}. {v0:.3e})')
with h5py.File(args.file, 'w') as hdf:
	hdf.create_dataset('n', data=parts)
	hdf.create_dataset('x', data=np.array([P0]))
	hdf.create_dataset('v', data=np.array([v0]))
	hdf.create_dataset('D', data=np.array([D]))
for p,data in enumerate(Pdx):
	with h5py.File(args.file+f'{p:03d}', 'w') as hdf:
		hdf.create_dataset("dx", data=data)
		hdf.create_dataset("v", data=vels[p])
