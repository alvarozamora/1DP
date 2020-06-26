import imageio as imo
import os
import glob
import re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from torchvision.utils import save_image
import io
import struct

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

print("Making Gif")
particle_names = glob.glob("Data/rho*")
sort_nicely(particle_names)
times = len(particle_names)

# Gather Euler Data
grid_names = []
for i in range(401):
	#grid_names.append(f'/Euler10/density_{i}.npy')
	grid_names.append(f'Euler/density_{i}.npy')

#print(particle_names)
#making animation

dur = 1/24.
#dur = 1.
t = 'd'
size = 8
with imo.get_writer('ParticleSoundWave.gif', duration=dur) as writer:
	for i in range(len(particle_names)):

		g_data = np.load(grid_names[i])
		Nc = len(g_data)

		#import pdb; pdb.set_trace()
		f = open(particle_names[i], 'rb')
		X = f.read(Nc*size)
		X = np.array(struct.unpack(t*Nc, X))

		x = np.linspace(1/2/Nc, 1 - 1/2/Nc, Nc)


		plt.figure(0,figsize=(6,4))
		plt.clf()
		plt.plot(x, g_data, label = r"$\rho$")
		plt.plot(x, X, lw = 1, label = r"$\rho_p$")
		plt.xlabel("Distance")
		plt.ylabel("Density")
		plt.grid(alpha=0.2)
		plt.xticks([0,0.25,0.5,0.75,1.0])
		d = 0.05
		plt.yticks([1-2*d, 1-d, 1, 1+d, 1+2*d])
		plt.ylim(1-2*d,1+2*d)
		plt.xlim(0,1)
		buf = io.BytesIO()
		plt.savefig(buf, format="png", dpi=230)
		buf.seek(0)
		image = imo.imread(buf)
		writer.append_data(image)
	writer.close()
'''
with imo.get_writer('ParticleSoundWave.mp4', fps=60) as writer:
	for i in range(len(particle_names)):

		g_data = np.load(grid_names[i])
		Nc = len(g_data)

		#import pdb; pdb.set_trace()
		f = open(particle_names[i], 'rb')
		X = f.read(Nc*size)
		X = np.array(struct.unpack(t*Nc, X))

		x = np.linspace(1/2/Nc, 1 - 1/2/Nc, Nc)


		plt.figure(0,figsize=(6,4))
		plt.clf()
		plt.plot(x, g_data, label = r'$\rho$')
		plt.plot(x, X, lw = 1, label = r'$\rho_p$')
		plt.xlabel('Distance')
		plt.ylabel('Density')
		plt.grid(alpha=0.2)
		plt.xticks([0,0.25,0.5,0.75,1.0])
		d = 0.05
		plt.yticks([1-2*d, 1-d, 1, 1+d, 1+2*d])
		plt.ylim(1-2*d,1+2*d)
		plt.xlim(0,1)
		buf = io.BytesIO()
		plt.savefig(buf, format='png', dpi=230)
		buf.seek(0)
		image = imo.imread(buf)
		writer.append_data(image)
	writer.close()
'''
