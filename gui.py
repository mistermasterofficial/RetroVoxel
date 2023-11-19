import pygame as pg
from copy import deepcopy
import sys

class BasicTheme:
	def __init__(self):
		self.font = pg.font.Font(pg.font.get_default_font(), 36)

		self.default_fg = (255,255,255)
		self.default_bg = (1,1,1)

		self.hover_bg = (255,255,255)
		self.hover_fg = (0,0,0)

		self.pressed_fg = (255,255,255)
		self.pressed_bg = (0,127,0)

class Container:
	def __init__(self, screen_size, grid_size, theme):
		self.screen_size = screen_size
		self.grid_size = grid_size

		self.theme = theme

		self.tile_size = (screen_size[0]//grid_size[0], screen_size[1]//grid_size[1])

		self.elements = []

	def update(self, events):
		screen = pg.Surface(self.screen_size, pg.SRCALPHA, 32)
		# screen.set_colorkey((0,0,0))

		for e in self.elements:
			surf = e.update(events)
			screen.blit(surf, (self.tile_size[0]*e.coords[0]+(self.tile_size[0]*e.size[0]-surf.get_width())/2, self.tile_size[1]*e.coords[1]+(self.tile_size[1]*e.size[1]-surf.get_height())/2))

		return screen

class ContainerGroup:
	def __init__(self, containers, current_name):
		self.containers = containers
		self.start_num = current_name
		self.name = current_name

	def update(self, events):
		return self.containers[self.name].update(events)

	def change_container(self, new_name):
		if new_name in self.containers:
			self.name = new_name

class Element:
	def __init__(self, root, coords, size, proportions=True):
		self.root = root
		self.root.elements.append(self)

		self.coords = coords
		self.size = size

		self.proportions = proportions

	def update(self, events, surface=pg.Surface((0,0)), is_transparent=True):
		if is_transparent:
			surface = surface.convert()
			surface.set_colorkey((0,0,0))
		if self.proportions:
			scale = min(self.size[0]*self.root.tile_size[0]/surface.get_width(), self.size[1]*self.root.tile_size[1]/surface.get_height())
			return pg.transform.scale(surface, (surface.get_width()*scale,surface.get_height()*scale))
		return pg.transform.scale(surface, (self.size[0]*self.root.tile_size[0],self.size[1]*self.root.tile_size[1]))	

class Label(Element):
	def __init__(self, root, coords, size, text):
		super().__init__(root, coords, size)

		self.text = text

	def update(self, events):
		return super().update(events, surface=self.root.theme.font.render(self.text, False, self.root.theme.default_fg, self.root.theme.default_bg))

class Button(Element):
	def __init__(self, root, coords, size, text, func):
		super().__init__(root, coords, size)

		self.func = func

		self.is_hover = False

		self.text = text

	def update(self, events):
		fg = self.root.theme.default_fg
		bg = self.root.theme.default_bg

		mouse_pos = pg.mouse.get_pos()
		if mouse_pos[0] >= self.coords[0] * self.root.tile_size[0] and mouse_pos[0] < (self.coords[0]+self.size[0]) * self.root.tile_size[0] and mouse_pos[1] >= self.coords[1] * self.root.tile_size[1] and mouse_pos[1] < (self.coords[1]+self.size[1]) * self.root.tile_size[1]:
			if not self.is_hover:
				try:
					self.root.theme.hover_sound.play()
				except:
					pass
				self.is_hover = True
			fg, bg = self.root.theme.hover_fg, self.root.theme.hover_bg
			mouse_pressed = pg.mouse.get_pressed()
			if mouse_pressed[0]:
				fg, bg = self.root.theme.pressed_fg, self.root.theme.pressed_bg

			for e in events:
				if e.type == pg.MOUSEBUTTONUP:
					try:
						self.root.theme.press_sound.play()
					except:
						pass
					self.func()
		else:
			self.is_hover = False

		return super().update(events, surface=self.root.theme.font.render(self.text, False, fg, bg))

class ImageLabel(Element):
	def __init__(self, root, coords, size, image):
		super().__init__(root, coords, size)

		self.image = image

	def update(self, events):
		return super().update(events, surface=self.image, is_transparent=False)

class ImageButton(Element):
	def __init__(self, root, coords, size, images, func, hover_func=lambda: print("hover_func")):
		super().__init__(root, coords, size)

		self.images = images

		self.func = func
		self.hover_func = hover_func

		self.is_hover = False

	def update(self, events):
		image_name = "default"

		mouse_pos = pg.mouse.get_pos()
		if mouse_pos[0] >= self.coords[0] * self.root.tile_size[0] and mouse_pos[0] < (self.coords[0]+self.size[0]) * self.root.tile_size[0] and mouse_pos[1] >= self.coords[1] * self.root.tile_size[1] and mouse_pos[1] < (self.coords[1]+self.size[1]) * self.root.tile_size[1]:
			if not self.is_hover:
				try:
					self.root.theme.hover_sound.play()
				except:
					pass
				self.is_hover = True
			image_name = "hover"
			mouse_pressed = pg.mouse.get_pressed()
			if mouse_pressed[0]:
				image_name = "pressed"

			for e in events:
				if e.type == pg.MOUSEBUTTONUP:
					try:
						self.root.theme.press_sound.play()
					except:
						pass
					self.func()
		else:
			self.is_hover = False

		return super().update(events, surface=self.images[image_name], is_transparent=False)

class Text(Element):
	def __init__(self, root, coords, size, text, text_size, align):
		super().__init__(root, coords, size)

		self.text = text
		self.text_size = text_size
		self.text_align = align

	def update(self, events):
		font_size = self.root.theme.font.render("0", False, self.root.theme.default_fg, self.root.theme.default_bg).get_size()
		surf = pg.Surface((font_size[0]*self.text_size[0], font_size[1]*self.text_size[1]))
		surf.fill(self.root.theme.default_bg)
		txt = ""
		row_num = 0
		for i in range(len(self.text)):
			if row_num>=self.text_size[1]:
				break
			txt += self.text[i]
			if len(txt)==self.text_size[0] or self.text[i]=="\n":
				if self.text[i]=="\n":
					txt = txt[:-1]
					if txt=="":
						continue
				row_surf = self.root.theme.font.render(txt, False, self.root.theme.default_fg, self.root.theme.default_bg)
				if self.text_align=="LEFT":
					surf.blit(row_surf, row_surf.get_rect(x=0, y=row_num*font_size[1]))
				elif self.text_align=="CENTER":
					surf.blit(row_surf, row_surf.get_rect(centerx=surf.get_width()/2, y=row_num*font_size[1]))
				else:
					surf.blit(row_surf, row_surf.get_rect(right=surf.get_width(), y=row_num*font_size[1]))
				row_num += 1
				txt = ""

		return super().update(events, surface=surf)