import pygame as pg
import math
import numpy as np
from numba import njit, prange

from math_functions import *

def get_model(fp, resolution):
	source_model = pg.image.load(fp); source_model = pg.surfarray.array3d(source_model)
	model = np.full((resolution,resolution,resolution,3), 0, dtype=np.uint8)
	for i in range(resolution):
		for j in range(resolution):
			for h in range(resolution):
				model[j][h][i] = source_model[h][i*resolution+j]
	return model

@njit(fastmath=True, parallel=True)
def ray_cast(camera_pos, vecs, distance, field, assets, resolution):
	screens = np.full((distance, len(vecs), resolution, 3), 0, dtype=np.uint8)

	for v in prange(len(vecs)):
		for z in prange(np.shape(screens)[0]):
			vec = vecs[v]*(z+1)/resolution
			if vec[0]+camera_pos[0]>=0 and vec[0]+camera_pos[0]<np.shape(field)[0] and vec[1]+camera_pos[1]>=0 and vec[1]+camera_pos[1]<np.shape(field)[1]:
				screens[z, v, :] = assets[field[int(vec[1]+camera_pos[1])][int(vec[0]+camera_pos[0])]][:][int((vec[1]+camera_pos[1])*resolution%resolution)][int((vec[0]+camera_pos[0])*resolution%resolution)]*(1-z/np.shape(screens)[0])
			# else:
			# 	break

	return screens

@njit(fastmath=True, parallel=True)
def multi_ray_cast(camera_pos, vecs, distance, field, assets, resolution):
	result_screens = np.full((distance, len(vecs), resolution*np.shape(field)[0], 3), 0, dtype=np.uint8)

	for i in prange(np.shape(field)[0]):
		result_screens[:, :, i*resolution:i*resolution+resolution] = ray_cast(camera_pos, vecs, distance, field[i], assets, resolution)

	return result_screens

class CameraRayCasting:
	def __init__(self, pos, distance, fov, ray_num, resolution):
		self.pos = pos
		self.distance = distance

		self.fov = fov

		self.resolution = resolution

		self.angle_y = 0
		
		try:
			self.vecs = [vec2(1, math.tan((-fov/2+fov/(ray_num-1)*(i))/180*math.pi)) for i in range(ray_num)]
		except ZeroDivisionError:
			self.vecs = [vec2(1, 0)]
	
	def rotate(self, angle_x, angle_y):
		self.vecs = [rotate_vec2(v, angle_x) for v in self.vecs]
		self.angle_y -= angle_y; self.angle_y = clamp(self.angle_y, -self.perspective_scale(1, self.resolution, len(self.vecs))*2, self.perspective_scale(1, self.resolution, len(self.vecs))*2)
	
	@staticmethod
	def perspective_scale(x, resolution, vecs_num):
		return (resolution*vecs_num)/(x*25)
	
	def render(self, field, assets):
		screen = pg.Surface((len(self.vecs), self.resolution*2))
		
		surfaces = multi_ray_cast(self.pos[:2], np.array(self.vecs), self.distance, field, assets, self.resolution)
		surfaces = [pg.surfarray.make_surface(surfaces[np.shape(surfaces)[0]-1-i]) for i in range(np.shape(surfaces)[0])]

		[surfaces[i].set_colorkey((0,0,0)) for i in range(len(surfaces))]

		surfaces = [pg.transform.scale(surfaces[i], (len(self.vecs), self.resolution*np.shape(field)[0]*self.perspective_scale(len(surfaces)-i, self.resolution, len(self.vecs)))) for i in range(len(surfaces))]

		[screen.blit(surfaces[i], surfaces[i].get_rect(centerx=len(self.vecs)*0.5, centery=screen.get_rect().centery+self.angle_y+(self.pos[2]/np.shape(field)[0]-0.5)*screen.get_height()*np.shape(field)[0]*self.perspective_scale(len(surfaces)-i, self.resolution, len(self.vecs))/2)) for i in range(len(surfaces))]

		return screen
