import random
import math
import numpy as np
import json

points = []

for i in range(0, 10000):
	x = float(random.randrange(1, 400))/100
	fx = (math.sin(math.pi * x)) / (math.pi*x)
	
	points.append( { "x" : round(x,1), "fx" : round(fx, 1)} )
	#points.append( { "x" : x, "fx" : fx} )
	
with open('data_fx.json', 'w') as outfile:
    json.dump(points, outfile)







