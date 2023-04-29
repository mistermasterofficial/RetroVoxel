import pygame as pg
import numpy as np
from numba import njit, cuda
from .math_functions import *

@njit(cache=True)
def multi_ray_cast_cpu(camera_pos, angle, fov, ray_num, distance, field, assets, resolution):
	screens = np.full((distance*resolution, ray_num, resolution*np.shape(field)[0], 3), 0)

	for v in range(ray_num):
		static_vec_x, static_vec_y = math.cos((angle-fov/2+fov/(ray_num-1)*(v))/180*math.pi)/math.cos((-fov/2+fov/(ray_num-1)*(v))/180*math.pi),math.sin((angle-fov/2+fov/(ray_num-1)*(v))/180*math.pi)/math.cos((-fov/2+fov/(ray_num-1)*(v))/180*math.pi)
		for z in range(np.shape(screens)[0]):
			vec_x = static_vec_x*(z+1)/resolution + camera_pos[0]
			vec_y = static_vec_y*(z+1)/resolution + camera_pos[1]
			if vec_x>=0 and vec_x<np.shape(field)[2] and vec_y>=0 and vec_y<np.shape(field)[1] and (screens[z-1, v, :, :]==0).any():
				for i in range(np.shape(field)[0]):
					screens[z, v, i*resolution:i*resolution+resolution] = assets[field[i][int(vec_y)][int(vec_x)]][:][int((vec_y)*resolution%resolution)][int((vec_x)*resolution%resolution)]
					#*(1-z/np.shape(screens)[0])
			else:
				break

	return screens
#(argtypes=[f8[:], f8, f8, i8, i8, i8[:,:,:], i8[:,:,:,:], i8])

@cuda.jit
def multi_ray_cast_cuda(camera_pos, angle, fov, ray_num, distance, field, assets, resolution, screens):
	#cuda.blockDim.x
	v = cuda.blockIdx.x
	i = cuda.blockIdx.y
	z = cuda.threadIdx.x
	vec_x = (math.cos((angle-fov/2+fov/(ray_num-1)*(v))/180*math.pi)/math.cos((-fov/2+fov/(ray_num-1)*(v))/180*math.pi))*(z+1)/resolution + camera_pos[0]
	vec_y = (math.sin((angle-fov/2+fov/(ray_num-1)*(v))/180*math.pi)/math.cos((-fov/2+fov/(ray_num-1)*(v))/180*math.pi))*(z+1)/resolution + camera_pos[1]
	if vec_x>=0 and vec_x<field.shape[2] and vec_y>=0 and vec_y<field.shape[1]:
		for j in range(resolution):
			screens[z, v, i*resolution+j, 0] = assets[field[i][int(vec_y)][int(vec_x)]][int((vec_y)*resolution%resolution)][int((vec_x)*resolution%resolution)][j][0]
			screens[z, v, i*resolution+j, 1] = assets[field[i][int(vec_y)][int(vec_x)]][int((vec_y)*resolution%resolution)][int((vec_x)*resolution%resolution)][j][1]
			screens[z, v, i*resolution+j, 2] = assets[field[i][int(vec_y)][int(vec_x)]][int((vec_y)*resolution%resolution)][int((vec_x)*resolution%resolution)][j][2]

def multi_ray_cast_gpu(camera_pos, angle, fov, ray_num, distance, field, assets, resolution):
	# stream = cuda.stream()
	screens = np.zeros((distance*resolution, ray_num, resolution*np.shape(field)[0], 3))
	# print(screens.flags)
	multi_ray_cast_cuda[(ray_num, field.shape[0]),(distance*resolution, 1)](np.ascontiguousarray(camera_pos), angle, fov, ray_num, distance, np.ascontiguousarray(field), np.ascontiguousarray(assets), resolution, np.ascontiguousarray(screens))
	# res = screens.copy_to_host(stream)
	# res = multi_ray_cast(camera_pos, angle, fov, ray_num, distance, field, assets, resolution)
	return np.asarray(screens)

def multi_ray_cast(camera_pos, angle, fov, ray_num, distance, field, assets, resolution):
	if cuda.is_available():
		return multi_ray_cast_gpu(camera_pos, angle, fov, ray_num, distance, field, assets, resolution)
	return multi_ray_cast_cpu(camera_pos, angle, fov, ray_num, distance, field, assets, resolution)


class CameraRayCasting:
	def __init__(self, pos, distance, fov, ray_num, resolution):
		self.pos = pos
		self.distance = distance

		self.anglex = 0
		self.angley = 0
		self.vec = vec2(1,0)

		self.ray_num = ray_num

		self.fov = fov

		self.resolution = resolution
	
	def rotate(self, anglex, angley):
		self.vec = rotate_vec2(self.vec, anglex)
		self.anglex += anglex; self.anglex %= 360
		self.angley += angley; self.angley = clamp(self.angley, -45, 45)
	
	def perspective_scale(self, x):
		return (self.resolution*3.6)/(x*self.fov)*self.ray_num
	
	def render(self, field, assets, entities, aspect):
		screen = pg.Surface((self.ray_num, self.ray_num//aspect))

		entities = sorted(entities, key=lambda e: lenght2(e[2][:2]-self.pos[:2]), reverse=False)
		entities_queue = []
		for e in entities:
			if not (dot2(norm2(e[2][:2]-self.pos[:2]), self.vec)<=0 or lenght2(e[2][:2]-self.pos[:2])>self.distance):
				entities_queue.append(e)
		perpendicular = norm2(rotate_vec2(self.vec, 90))

		surfaces = multi_ray_cast_gpu(self.pos[:2], self.anglex, self.fov, self.ray_num, self.distance, field, assets, self.resolution)

		for i in range(np.shape(surfaces)[0]):
			surface = pg.transform.scale(pg.surfarray.make_surface(surfaces[np.shape(surfaces)[0]-1-i]), (self.ray_num, self.resolution*np.shape(field)[0]*self.perspective_scale(np.shape(surfaces)[0]-i)))
			surface.set_colorkey((0,0,0))
			screen.blit(surface, surface.get_rect(centerx=self.ray_num*0.5, bottom=screen.get_rect().centery+math.tan(self.angley/180*math.pi)*screen.get_height()+(self.pos[2])*self.resolution*self.perspective_scale(np.shape(surfaces)[0]-i)))

			delete_entities = []

			for e in entities_queue:
				alpha = 1-math.acos(dot2(norm2(e[2][:2]-self.pos[:2]), perpendicular))/math.pi*2
				if math.isnan(alpha): alpha = 0

				if int(lenght2(e[2][:2]-self.pos[:2])*math.cos(alpha*math.pi/2)*self.resolution) == (np.shape(surfaces)[0]-1-i):
					
					y = int(e[2][2]*self.resolution*self.perspective_scale(np.shape(surfaces)[0]-i))
					# x = math.tan(alpha*math.pi/2)*(45/self.fov)*(self.ray_num+e[0].get_width()/2*self.perspective_scale(np.shape(surfaces)[0]-i))
					x = math.tan(alpha*math.pi/2)*(45/self.fov)*(self.ray_num+self.resolution/2*self.perspective_scale(np.shape(surfaces)[0]-i))

					entity_surface = pg.transform.scale(e[0], (self.resolution*e[1][0]*self.perspective_scale(np.shape(surfaces)[0]-i),self.resolution*e[1][1]*self.perspective_scale(np.shape(surfaces)[0]-i)))

					screen.blit(entity_surface, entity_surface.get_rect(centerx=screen.get_rect().centerx+x, bottom=screen.get_rect().centery-y+math.tan(self.angley/180*math.pi)*screen.get_height()+(self.pos[2])*self.resolution*self.perspective_scale(np.shape(surfaces)[0]-i)))

					delete_entities.append(e)

			for e in delete_entities:
				entities_queue.remove(e)

		return screen
