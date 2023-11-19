from .camera import *

class CameraController(CameraRayCasting):
	def __init__(self, pos, distance, fov, ray_num, resolution):
		self.move_vecs = [vec2(math.cos(i/2*math.pi), -math.sin(i/2*math.pi)) for i in range(4)]
		self.main_vec = vec3(1,0,0)
		super().__init__(pos, distance, fov, ray_num, resolution)
	
	def rotate(self, anglex, angley):
		self.move_vecs = [rotate_vec2(v, anglex) for v in self.move_vecs]
		super().rotate(anglex, angley)
		self.main_vec = vec3(math.cos(self.anglex*math.pi/180)*math.cos(self.angley*math.pi/180),math.sin(self.anglex*math.pi/180)*math.cos(self.angley*math.pi/180),math.sin(self.angley*math.pi/180))

	def update(self, scene, screen):
		center = screen.get_rect().center
		for e in scene.events:
			if e.type == pg.MOUSEMOTION:
				if abs(e.rel[0])>1:
					self.rotate(e.rel[0], 0)
					pg.mouse.set_pos(center)
				if abs(e.rel[1])>1:
					self.rotate(0, -e.rel[1])
					pg.mouse.set_pos(center)
		keys = pg.key.get_pressed()
		if keys[pg.K_w]:
			self.pos[:2] += self.move_vecs[0]/16
		if keys[pg.K_a]:
			self.pos[:2] += self.move_vecs[1]/16
		if keys[pg.K_s]:
			self.pos[:2] += self.move_vecs[2]/16
		if keys[pg.K_d]:
			self.pos[:2] += self.move_vecs[3]/16
		if keys[pg.K_SPACE]:
			self.pos[2] += 1/self.resolution
		if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]:
			self.pos[2] -= 1/self.resolution