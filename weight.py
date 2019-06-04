import tensorflow
from model import *
import numpy as np

np.set_printoptions(edgeitems = 100000, suppress = True)
num_mcep = 24
num_speakers = 10

model = CCGAN(num_features = num_mcep, num_speakers = num_speakers)
model.load(filepath = os.path.join('./model', 'CCGAN.ckpt'))
model.print_weight()