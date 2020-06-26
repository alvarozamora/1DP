import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import h5py
import pdb
sns.set_context("talk") #darkgrid, whitegrid, dark, white, ticks
parser = argparse.ArgumentParser(description='HDF5 Initial Condition Pipeline')
parser.add_argument('--file', type=str, default='particle/particle', help='Output: hdf base file path/name')
parser.add_argument('-n', type=int, default=10**9, help='Particle Number')
parser.add_argument('-c', type=int, default=10**3, help='Number of cores/files')
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
xL = np.linspace(0,1-1/Nc,Nc)		# Grid Left
v = delta*np.sin(4*np.pi*x)             # Bulk Velocity
rho = 1 - v                             # Density  
Pressure = rho*c2                              # Pressure
#ied = c2/(g-1)                          # Thermal Energy
dx = np.gradient(x)                     # Cell Sizes (UNIFORM GRID ONLY)
M = (rho*dx).sum()                      # Total Mass (UNIFORM GRID ONLY)


# Particle Parameters
N = args.n                         # Approximate number of particles
mp = M/N                         # Particle Mass
# Final number of particles per cell, total
Np = np.floor((rho*dx)/mp + np.random.uniform(size=rho.size)).astype(int)
NP = Np.sum()
# Particle Diameter
Dmax = (dx/Np).min()
D = 1e-4*Dmax
D = 0

#Initialize Velocities
s = np.sqrt(2*Pressure/rho)

print(len(Np), len(rho), len(v), len(Pressure))
# bin and effbin sizes
# velocities
with h5py.File(args.file, 'w') as hdf:
	hdf.create_dataset('D', data=np.array([D]))
	hdf.create_dataset('m', data=np.array([mp]))
	hdf.create_dataset('Nc', data=np.array([Nc]))
	hdf.create_dataset('N', data=np.array([NP]))
	hdf.create_dataset('Np', data=Np)
	hdf.create_dataset('xgrid', data=x)
	hdf.create_dataset('rhogrid', data=rho)
	hdf.create_dataset('vgrid', data=v)
	hdf.create_dataset('Pgrid', data=Pressure)
print("Saved Metadata")
print(Nc)
print(NP)
print(mp)

print(f"Making Initial Conditions for {args.n:.0e} particles across {Nc} cells")
for j in range(len(dx)):
	Left = xL[j]

	effbin = dx[j]/Np[j]-D
	bin = dx[j]/Np[j]

	vels = np.random.randn(Np[j])*s[j] + v[j]

	uniforms = np.random.random(Np[j])

	P = Left + np.arange(Np[j])*bin + uniforms*effbin + D/2

	#pdb.set_trace()
	print(f"Finished cell {j} with minmax = ({P.min():.2e}, {P.max():.2e})")
	file = format
	with h5py.File(args.file+f"{j:03d}", "w") as hdf:
		hdf.create_dataset("x", data=P)
		hdf.create_dataset("v", data=vels)
		#hdf.create_dataset("xnew", data=P[p]) #keep in case we want to output positions

