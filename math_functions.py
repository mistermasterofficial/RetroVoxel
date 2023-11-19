import numpy as np
import math

vec2 = lambda x, y: np.array([x, y], dtype=np.float64)
vec3 = lambda x, y, z: np.array([x, y, z], dtype=np.float64)

lenght2 = lambda v: math.sqrt(v[0]*v[0]+v[1]*v[1])
norm2 = lambda v: v/lenght2(v)
dot2 = lambda v1, v2: v1[0]*v2[0]+v1[1]*v2[1]

lenght3 = lambda v: math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
norm3 = lambda v: v/lenght3(v)
dot3 = lambda v1, v2: v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

def rotate_vec2(vec, angle):
	angle = angle/180*math.pi
	return vec2(vec[0]*math.cos(angle)-vec[1]*math.sin(angle), vec[0]*math.sin(angle)+vec[1]*math.cos(angle))

clamp = lambda value, _min, _max: max(_min, min(_max, value))

def vec2_to_angle(vec1, vec2):
    return math.atan2(vec1[1] - vec2[1], vec1[0] - vec2[0])

def rotate_vec3_x(vec, angle):
	angle = angle/180*math.pi
	return vec3(
		vec[0],
		math.cos(angle)*vec[1]+math.sin(angle)*vec[2]
		-math.sin(angle)*vec[1]+math.cos(angle)*vec[2]
	)

def rotate_vec3_y(vec, angle):
	angle = angle/180*math.pi
	return vec3(
		math.cos(angle)*vec[0]-math.sin(angle)*vec[2],
		vec[1],
		math.sin(angle)*vec[0]+math.cos(angle)*vec[2]
	)

def rotate_vec3_z(vec, angle):
	angle = angle/180*math.pi
	return vec3(
		math.cos(angle)*vec[0]+math.sin(angle)*vec[1],
		-math.sin(angle)*vec[0]+math.cos(angle)*vec[1],
		vec[2]
	)