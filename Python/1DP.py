from utils import *
import numpy as np
import matplotlib
import pdb
#matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import h5py
import time

sns.set_context("talk") #darkgrid, whitegrid, dark, white, ticks
sns.set_style("white")

parser = argparse.ArgumentParser(description='HDF5 Initial Condition Pipeline')
parser.add_argument('--file', type=str, default='particle/particle', help='Output: hdf base file path/name')
parser.add_argument('-n', type=int, default=10**9, help='Particle Number')
parser.add_argument('-c', type=int, default=10**3, help='Number of cores/files')
parser.add_argument('-b', type=int, default=0, help='Periodic (0), Reflective (1)')
args = parser.parse_args()


# Sound Wave
f = 1          # Degrees of Freedom
g = (f+2)/f    # Adiabatic Index
c2 = 1.0       # Squared Sound Speed
delta = 0.05   # Sound Wave - Perturbation Amplitude
# Setting up Grid
Nc = 512                                # Number of Cells
x = np.linspace(0+1/Nc/2,1-1/Nc/2,Nc)   # Grid Centers on (0,1)
v = delta*np.sin(4*np.pi*x)             # Bulk Velocity
rho = 1 - v/np.sqrt(c2)                  # Density  
Pressure = rho*c2                       # Pressure
#ied = c2/(g-1)                         # Thermal Energy
p = {'rho': rho, 'Pressure': Pressure, 'v': v, 'x': x}

P, Pdx, vels, P0, p = InitialConditions(x, rho, v, Pressure, args, p)
#pdb.set_trace()
DumpDensity(Pdx, x, p, P)
DumpPhaseSpace(P, vels, Nc)


ntimes = 100
Tf = 1.0
times = np.linspace(Tf/ntimes, Tf, ntimes-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P = torch.from_numpy(P).to(device)
vels = torch.from_numpy(vels).to(device) 
for i in range(ntimes-1):
	t = times[i]

	start = time.time()
	Pnew = torch.sort((P + vels*t)%1)[0]
	#pdb.set_trace()
	Pdxnew = Pnew.roll(-1)-Pnew
	end = time.time()
	print(f"Advected to time {t:.3e} in {end-start:.3e} seconds")
	DumpDensity(Pdxnew.cpu().numpy(), x, p, Pnew.cpu().numpy(), Ns=10**6, fname=f'Output/Density{i+1:03d}.png')













# P = np.array_split(P, args.c)
# Pdx = np.array_split(Pdx, args.c)
# vels = np.array_split(vels, args.c)
# parts = np.array([len(q) for q in Pdx]).astype(int)
# print(f'Particle 1 : ({P0:.3e}. {v0:.3e})')
# with h5py.File(args.file, 'w') as hdf:
# 	hdf.create_dataset('n', data=parts)
# 	hdf.create_dataset('x', data=np.array([P0]))
# 	hdf.create_dataset('v', data=np.array([v0]))
# 	hdf.create_dataset('D', data=np.array([D]))
# 	hdf.create_dataset('Nc', data=np.array([Nc]))
# 	hdf.create_dataset('xgrid', data=x)
# 	hdf.create_dataset('rhogrid', data=rho)
# 	hdf.create_dataset('vgrid', data=v)
# 	hdf.create_dataset('Pgrid', data=Pressure)
# for p,data in enumerate(Pdx):
# 	with h5py.File(args.file+f'{p:03d}', 'w') as hdf:
# 		hdf.create_dataset("dx", data=data)
# 		hdf.create_dataset("v", data=vels[p])
# 		hdf.create_dataset("x", data=P[p])
# 		hdf.create_dataset("xnew", data=P[p])
