import random
import math
import numpy as np
import json

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

class Circle:
    def __init__(self, origin, radius):
        self.origin = origin
        self.radius = radius

origin = Point(0, 0)
radius = 1
circle = Circle(origin, radius)

points = []

for i in range(0, 10000):
    p = random.random() * 2 * math.pi
    r = circle.radius * math.sqrt(random.random())
    x = math.cos(p) * r
    y = math.sin(p) * r

    points.append( { "x" : round(x,1), "y" : round(y, 1)} )



positions = { "c1" : [], "c2" : [], "c3" : [], "c4" : [], "c5" : [], "c6" : [], "c7" : [], "c8" : [] }


for point in points:

    if point["x"] > 0 and point["y"] > 0: # 1 quadrante

        if abs(point["x"]) + abs(point["y"]) < 1:
            positions["c1"].append(point)
        else:
            positions["c5"].append(point)


    elif point["x"] < 0 and point["y"] > 0: # 2 quadrante

        if abs(point["x"]) + abs(point["y"]) < 1:
            positions["c2"].append(point)
        else:
            positions["c6"].append(point)


    elif point["x"] < 0 and point["y"] < 0: # 3 quadrante
        
        if abs(point["x"]) + abs(point["y"]) < 1:
            positions["c3"].append(point)
        else:
            positions["c7"].append(point)

    elif point["x"] > 0 and point["y"] < 0: # 4 quadrante

        if abs(point["x"]) + abs(point["y"]) < 1:
            positions["c4"].append(point)
        else:
            positions["c8"].append(point)


with open('data.json', 'w') as outfile:
    json.dump(positions, outfile)







