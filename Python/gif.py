import imageio as io
import os
import glob
import re
import numpy as np
import torch
import torch.nn.functional as F
#from torchvision.utils import save_image

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
file_names = glob.glob("Output/Density*.png")
sort_nicely(file_names)
#print(file_names)
#making animation

dur = 1/30.
with io.get_writer('ParticleSoundWave.gif', duration=dur) as writer:
    for filename in file_names:
        image = io.imread(filename)
        writer.append_data(image)
writer.close()

