import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import struct
import sod


reals = range(1)
t = range(10)
boxes = 200
L = 10**6
w = 30875045135703220*50
w = 3087504513.570322036743164*50
w = 3087504513.570322036743164*boxes/2
grid = 0
h = np.zeros((len(t),boxes))
for i in reals:
	for k in t:
		grid += 1
		file = 'Grid'+str(grid)
		size = 8
		num  = L*4
		type = 'd' #d is double, f is float, i is integer

		f = open(file, 'rb')

		X = f.read(num*size)
		X = np.array(struct.unpack(type*num, X))

		z = X[:L]/w
		vx = X[L:2*L]/w
		vy = X[2*L:3*L]/w
		vz = X[3*L:4*L]/w

		h[k] += np.histogram(z, bins = boxes, range = (0,1))[0]
		print(grid)
h = h/len(reals)/L*(9/16)*boxes

gamma = 5/3
dustFrac = 0.0
npts = 500
time = 0.2
left_state = (1,1,0)
right_state = (0.1, 0.125, 0.)

# left_state and right_state set pressure, density and u (velocity)
# geometry sets left boundary on 0., right boundary on 1 and initial
# position of the shock xi on 0.5
# t is the time evolution for which positions and states in tube should be
# calculated
# gamma denotes specific heat
# note that gamma and npts are default parameters (1.4 and 500) in solve
# function

plt.figure()
for T in t:
	time = 0.006266570686578*(T+1)
	plt.bar(np.arange(boxes)/boxes+0.5/boxes, h[T] , width = 1/boxes)
	positions, regions, values = sod.solve(left_state=left_state, right_state=right_state, geometry=(0., 1., 0.5), t=time, gamma=gamma, npts=npts, dustFrac=dustFrac)
	#x = [0,0.5,0.5,1]
	y = np.array([1,1,1/8,1/8])
	#print(y,np.sum(y))
	#plt.plot(x,y,'k')
	plt.plot(values['x'], values['rho'], 'k')
	print(np.sum(h[T])/boxes, np.sum(y)/4) 

	while len(str(T)) < 4:
		T = '0' + str(T)
	plt.savefig('hist'+str(T)+'.png')
	plt.clf()
