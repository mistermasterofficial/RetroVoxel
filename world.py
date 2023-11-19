from .controller import *
from .entity import *

import json
from copy import deepcopy

class CollisionCells:
	def __init__(self):
		self.relief_cells = []
	
	def collide(self, point_a, point_b, field):
		x = point_a[0]
		y = point_a[1]
		z = point_a[2]

		if point_b[0] >= 0 and point_b[1] >= 0 and point_b[2] >= 0 and point_b[0] < np.shape(field)[2] and point_b[1] < np.shape(field)[1] and point_b[2] < np.shape(field)[0]:
			if field[int(point_b[2])][int(point_b[1])][int(point_b[0])]==1:
				x = point_b[0]
				if field[int(z)][int(y)][int(x)] == 1:
					x = point_a[0]

				y = point_b[1]
				if field[int(z)][int(y)][int(x)] == 1:
					y = point_a[1]

				z = point_b[2]
				if field[int(z)][int(y)][int(x)] == 1:
					z = point_a[2]

			elif field[int(point_b[2])][int(point_b[1])][int(point_b[0])]>=1:
				x = point_b[0]
				y = point_b[1]
				z = max(point_b[2], int(point_b[2])+self.relief_cells[field[int(point_b[2])][int(point_b[1])][int(point_b[0])]-2][int((point_b[1]%1)*len(self.relief_cells[field[int(point_b[2])][int(point_b[1])][int(point_b[0])]-2]))][int((point_b[0]%1)*len(self.relief_cells[field[int(point_b[2])][int(point_b[1])][int(point_b[0])]-2][0]))])
			elif field[int(point_b[2])][int(point_b[1])][int(point_b[0])]==0:
				x = point_b[0]
				y = point_b[1]
				z = point_b[2]

		result = vec3(x, y, z)

		return result

	@staticmethod
	def border_collision(point, field, resolution):
		return vec3(clamp(point[0], 0, np.shape(field)[2]-1/resolution),clamp(point[1], 0, np.shape(field)[1]-1/resolution),clamp(point[2], 0, np.shape(field)[0]-1/resolution))

class Scene:
	def __init__(self, window_size, resolution, camera_distance, camera_fov, camera_ray_num, fog_color=None, fog_distance=0.5, skybox=None):
		self.window_size = window_size

		self.global_resolution = resolution

		self.assets = []

		self.old_props = {}
		self.props = {}
		self.field = None
		self.collision_field = None

		self.onload_script = lambda s: print("ONLOAD SCRIPT IS NOT FOUND!")
		self.onrun_script = lambda s: print("ONRUN SCRIPT IS NOT FOUND!")
		self.onrender_script = lambda s, scr: print("ONRENDER SCRIPT IS NOT FOUND!")

		self.camera = CameraController(vec3(0,0,0), camera_distance, camera_fov, camera_ray_num, self.global_resolution)
		self.fog_color = fog_color
		self.fog_distance = fog_distance
		self.skybox = skybox

		self.collision = CollisionCells()

		self.entities = []

		self.is_stop = False
		self.is_stop_camera = False

		self.events = []
		self.global_vars = globals()

		self.delta_time = 0

	@staticmethod
	def get_model(fp, resolution):
		source_model = pg.image.load(fp); source_model = pg.surfarray.array3d(source_model)
		model = np.full((resolution,resolution,resolution,3), 0, dtype=np.uint8)
		for i in range(resolution):
			for j in range(resolution):
				for h in range(resolution):
					model[j][h][i] = source_model[h][i*resolution+j]
		return model

	def load_model(self, filename):
		self.assets.append(self.get_model(f"assets/{filename}", self.global_resolution))

	def load_script(self, filename, global_vars):
		with open(f"scripts/{filename}.py", "r", encoding='utf-8') as f:
			info = f.read()

			self.field = None
			self.props = {}
			self.entities.clear()
			self.collision_field = None
			self.is_stop = False
			self.is_stop_camera = False
			self.camera.rotate(-self.camera.anglex, -self.camera.angley)

			self.global_vars = global_vars
			scope = {}; scope.update(self.global_vars)

			exec(info, scope)

			self.onload_script = scope.get("on_load", lambda s, scr: print("ONLOAD SCRIPT IS NOT FOUND!"))
			self.onrun_script = scope.get("on_run", lambda s, scr: print("ONRUN SCRIPT IS NOT FOUND!"))
			self.onrender_script = scope.get("on_render", lambda s, scr: print("ONRENDER SCRIPT IS NOT FOUND!"))
		self.onload_script(self)

	def get_rendered_image(self):
		entities = [e.get_tuple() for e in self.entities]
		return self.camera.render(self.field, np.array(self.assets), entities, self.window_size[0]/self.window_size[1], fog_color=self.fog_color, fog_distance=self.fog_distance, skybox=self.skybox)

	def render(self):
		return self.onrender_script(self, self.get_rendered_image())
	
	def panorama(self):
		entities = [e.get_tuple() for e in self.entities]
		return self.camera.panorama(self.field, np.array(self.assets), entities, self.window_size[0]/self.window_size[1], fog_color=self.fog_color, fog_distance=self.fog_distance, skybox=self.skybox)

	def teleport_camera_to(self, index):
		for i in range(np.shape(self.field)[0]):
			for j in range(np.shape(self.field)[1]):
				for k in range(np.shape(self.field)[2]):
					if self.field[i][j][k]==index:
						self.camera.pos = vec3(k, j, i)
						return

	def check_camera_place(self):
		return self.field[int(self.camera.pos[2])][int(self.camera.pos[1])][int(self.camera.pos[0])]

	def spawn_entity(self, entity):
		self.entities.append(entity)

	def kill_entity(self, name, like_name = False):
		for e in self.entities:
			if e.name == name or (like_name and name in e.name):
				self.entities.remove(e)
				return True
		return False

	def kill_entities(self, name, like_name = False):
		is_killed = False
		for e in self.entities:
			if e.name == name or (like_name and name in e.name):
				self.entities.remove(e)
				is_killed = True
		return is_killed

	def get_entity(self, name, like_name = False):
		for e in self.entities:
			if e.name == name or (like_name and name in e.name):
				return e
		return False

	def get_all_entity(self, name, like_name = False):
		all_entities = []
		for e in self.entities:
			if e.name == name or (like_name and name in e.name):
				all_entities.append(e)
		return all_entities

	def update(self, events, screen, global_vars):
		self.events = events
		self.global_vars = global_vars
		if not self.is_stop:
			if not self.is_stop_camera:
				point_a = deepcopy(self.camera.pos)
				self.camera.update(self, screen)
				point_b = self.camera.pos
				self.camera.pos = self.collision.collide(point_a, point_b, self.collision_field)

			[e.update(self) for e in self.entities]
			self.onrun_script(self)