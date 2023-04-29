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

	def update(self):
		pass