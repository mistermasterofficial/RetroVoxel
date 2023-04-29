import pygame as pg
from copy import deepcopy

class Theme:
	def __init__(self):
		self.font = pg.font.Font(pg.font.get_default_font(), 36)
		# self.font = pg.font.Font(r"E:\GitHub\Mino\assets\font.ttf", 72)

		self.default_fg = (255,255,255)
		self.default_bg = (1,1,1)

		self.hover_bg = (255,255,255)
		self.hover_fg = (0,0,0)

		self.pressed_fg = (255,255,255)
		self.pressed_bg = (0,127,0)

default_theme = Theme()

class Container:
	def __init__(self, screen_size, grid_size, theme=default_theme):
		self.screen_size = screen_size
		self.grid_size = grid_size

		self.theme = theme

		self.tile_size = (screen_size[0]//grid_size[0], screen_size[1]//grid_size[1])

		self.elements = []

	def update(self, events):
		screen = pg.Surface(self.screen_size)
		screen.set_colorkey((0,0,0))

		for e in self.elements:
			screen.blit(e.update(events), (self.tile_size[0]*e.coords[0], self.tile_size[1]*e.coords[1]))

		return screen

class ContainerGroup:
	def __init__(self, containers, current_name, theme=default_theme):
		self.containers = containers
		self.start_num = current_name
		self.name = current_name

	def update(self, events):
		return self.containers[self.name].update(events)

	def change_container(self, new_name):
		if new_name in self.containers:
			self.name = new_name

class Element:
	def __init__(self, root, coords, size, theme=default_theme):
		self.root = root
		self.root.elements.append(self)

		self.coords = coords
		self.size = size

		self.theme = theme

	def update(self, events, surface=pg.Surface((0,0))):
		return pg.transform.scale(surface, (self.size[0]*self.root.tile_size[0],self.size[1]*self.root.tile_size[1]))

class Label(Element):
	def __init__(self, root, coords, size, text, theme=default_theme):
		super().__init__(root, coords, size, theme=theme)

		self.text = text

	def update(self, events):
		return super().update(events, surface=self.theme.font.render(self.text, False, self.theme.default_fg, self.theme.default_bg))

class Button(Element):
	def __init__(self, root, coords, size, text, func, theme=default_theme):
		super().__init__(root, coords, size, theme=theme)

		self.func = func

		self.text = text

	def update(self, events):
		fg = self.theme.default_fg
		bg = self.theme.default_bg

		mouse_pos = pg.mouse.get_pos()
		if mouse_pos[0] >= self.coords[0] * self.root.tile_size[0] and mouse_pos[0] < (self.coords[0]+self.size[0]) * self.root.tile_size[0] and mouse_pos[1] >= self.coords[1] * self.root.tile_size[1] and mouse_pos[1] < (self.coords[1]+self.size[1]) * self.root.tile_size[1]:
			fg, bg = self.theme.hover_fg, self.theme.hover_bg
			mouse_pressed = pg.mouse.get_pressed()
			if mouse_pressed[0]:
				fg, bg = self.theme.pressed_fg, self.theme.pressed_bg

			for e in events:
				if e.type == pg.MOUSEBUTTONDOWN:
					self.func()

		return super().update(events, surface=self.theme.font.render(self.text, False, fg, bg))

class ImageLabel(Element):
	def __init__(self, root, coords, size, image):
		super().__init__(root, coords, size)

		self.image = image

	def update(self, events):
		return super().update(events, surface=self.image)

class ImageButton(Element):
	def __init__(self, root, coords, size, images, func):
		super().__init__(root, coords, size)

		self.images = images

		self.func = func

	def update(self, events):
		image_name = "default"

		mouse_pos = pg.mouse.get_pos()
		if mouse_pos[0] >= self.coords[0] * self.root.tile_size[0] and mouse_pos[0] < (self.coords[0]+self.size[0]) * self.root.tile_size[0] and mouse_pos[1] >= self.coords[1] * self.root.tile_size[1] and mouse_pos[1] < (self.coords[1]+self.size[1]) * self.root.tile_size[1]:
			image_name = "hover"
			mouse_pressed = pg.mouse.get_pressed()
			if mouse_pressed[0]:
				image_name = "pressed"

			for e in events:
				if e.type == pg.MOUSEBUTTONDOWN:
					self.func()

		return super().update(events, surface=self.images[image_name])