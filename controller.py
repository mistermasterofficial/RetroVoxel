from .camera import *

class CameraController(CameraRayCasting):
	def __init__(self, pos, distance, fov, ray_num, resolution):
		self.move_vecs = [vec2(math.cos(i/2*math.pi), -math.sin(i/2*math.pi)) for i in range(4)]
		super().__init__(pos, distance, fov, ray_num, resolution)
	
	def rotate(self, anglex, angley):
		self.move_vecs = [rotate_vec2(v, anglex) for v in self.move_vecs]
		super().rotate(anglex, angley)

	def update(self, event, center):
		for e in event:
			if e.type == pg.MOUSEMOTION:
				if abs(e.rel[0])>1:
					self.rotate(e.rel[0], 0)
					pg.mouse.set_pos(center)
				if abs(e.rel[1])>1:
					self.rotate(0, -e.rel[1])
					pg.mouse.set_pos(center)
		keys = pg.key.get_pressed()
		if keys[pg.K_w]:
			self.pos[:2] += self.move_vecs[0]/self.resolution
		if keys[pg.K_a]:
			self.pos[:2] += self.move_vecs[1]/self.resolution
		if keys[pg.K_s]:
			self.pos[:2] += self.move_vecs[2]/self.resolution
		if keys[pg.K_d]:
			self.pos[:2] += self.move_vecs[3]/self.resolution
		if keys[pg.K_SPACE]:
			self.pos[2] += 1/self.resolution
		if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]:
			self.pos[2] -= 1/self.resolution