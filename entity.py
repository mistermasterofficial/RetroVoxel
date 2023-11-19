import pygame as pg
from .math_functions import *

class Entity:
	def __init__(self, name, assets, start_texture_name, size, pos, properties={}):
		self.assets = assets
		self.texture_name = start_texture_name

		self.size = size

		self.pos = pos

		self.name = name
		self.properties = properties

	def get_tuple(self):
		return (self.assets[self.texture_name], self.size, self.pos)

	def update(self, scene):
		pass

class SoundPoint:
	def __init__(self, sound, pos, distance, is_loop=False):
		self.sound = sound
		self.channel = None
		self.pos = pos
		self.is_loop = is_loop
		self.distance = distance
	
	def play(self):
		if isinstance(self.channel, type(None)) or not self.channel.get_busy():
			self.channel = self.sound.play()

	def update(self, scene):
		try:
			if self.channel.get_busy():
				self.channel.set_source_location(-(1-math.acos(dot2(norm2(self.pos[:2]-scene.camera.pos[:2]), rotate_vec2(scene.camera.vec, -90)))/math.pi*2)*90, int(clamp((lenght2(scene.camera.pos[:2]-self.pos[:2]))/self.distance,0,1)*255))
		except:
			pass
		if self.is_loop:
			self.play()