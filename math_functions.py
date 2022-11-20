import numpy as np
import math

vec2 = lambda x, y: np.array([x, y], dtype=np.float64)
vec3 = lambda x, y, z: np.array([x, y, z], dtype=np.float64)

def rotate_vec2(vec, angle):
	angle = angle/180*math.pi
	return vec2(vec[0]*math.cos(angle)-vec[1]*math.sin(angle), vec[0]*math.sin(angle)+vec[1]*math.cos(angle))

clamp = lambda value, _min, _max: max(_min, min(_max, value))