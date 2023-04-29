from .controller import *
from .entity import *

import json
from copy import deepcopy

class CollisionCells:
	def __init__(self):
		self.cells = []
	
	def move(self, point_a, point_b, field):
		x = point_a[0]
		y = point_a[1]
		z = point_a[2]

		if point_b[0] >= 0 and point_b[1] >= 0 and point_b[2] >= 0 and point_b[0] < np.shape(field)[2] and point_b[1] < np.shape(field)[1] and point_b[2] < np.shape(field)[0]:
			x = point_b[0]
			try:
				if field[int(z)][int(y)][int(x)] in self.cells:
					x = point_a[0]
			except IndexError:
				x = point_b[0]
			y = point_b[1]
			try:
				if field[int(z)][int(y)][int(x)] in self.cells:
					y = point_a[1]
			except IndexError:
				y = point_b[1]
			z = point_b[2]
			try:
				if field[int(z)][int(y)][int(x)] in self.cells:
					z = point_a[2]
			except IndexError:
				z = point_b[2]
		else:
			x = point_b[0]
			y = point_b[1]
			z = point_b[2]

		result = vec3(x, y, z)

		return result

	@staticmethod
	def border_collision(point, field):
		return vec3(clamp(point[0], 0, np.shape(field)[2]),clamp(point[1], 0, np.shape(field)[1]),clamp(point[2], 0, np.shape(field)[0]))

class Scene:
	def __init__(self, resolution, camera_distance, camera_fov, camera_ray_num):
		self.global_resolution = resolution

		self.assets = [np.full((self.global_resolution,self.global_resolution,self.global_resolution,3), 0)]

		self.field_properties = None
		self.field = None

		self.camera = CameraController(vec3(0,0,0), camera_distance, camera_fov, camera_ray_num, self.global_resolution)

		self.collision = CollisionCells()

		self.entities = []

		self.is_stop = False

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

	def load_field(self, filename):
		with open(f"maps/{filename}.json", "r") as f:
			info = json.load(f)
			self.field = np.array(list(info["field"]))
			self.field_properties = info["properties"]

	def render(self, aspect):
		entities = [e.get_tuple() for e in self.entities]
		return self.camera.render(np.flip(self.field, 0), np.array(self.assets), entities, aspect)

	def teleport_camera_to(self, index):
		for i in range(np.shape(self.field)[0]):
			for j in range(np.shape(self.field)[1]):
				for k in range(np.shape(self.field)[2]):
					if self.field[i][j][k]==index:
						self.camera.pos = vec3(k+0.5, j+0.5, i+0.5)
						return

	def check_camera_place(self):
		return self.field[int(self.camera.pos[2])][int(self.camera.pos[1])][int(self.camera.pos[0])]

	def spawn_entity(self, entity):
		self.entities.append(entity)

	def kill_entity(self, name):
		for e in entities:
			if e.name == name:
				self.entities.remove(e)
				return True
		return False

	def kill_entities(self, name):
		is_killed = False
		for e in entities:
			if e.name == name:
				self.entities.remove(e)
				is_killed = True
		return is_killed

	def update(self, events, screen):
		if not self.is_stop:
			point_a = deepcopy(self.camera.pos)
			self.camera.update(events, screen.get_rect().center)
			point_b = self.camera.pos
			self.camera.pos = self.collision.move(point_a, point_b, self.field)
			self.camera.pos = self.collision.border_collision(self.camera.pos, self.field)

			[e.update() for e in self.entities]