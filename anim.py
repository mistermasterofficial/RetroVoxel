import time

class Anim():
	def __init__(self, func, duration, is_loop):
		self.func = func

		self.start_time, self.pause_time, self.end_time = 0, 0, 0

		self.duration = duration

		self.is_start = False
		self.is_pause = False
		self.is_loop = is_loop
	
	def start(self):
		self.is_start = True
		self.start_time = time.time()
		self.end_time = self.start_time + self.duration

	def __call__(self):
		if self.is_start and not self.is_pause:
			if not self.is_loop:
				return self.func(min((time.time()-self.start_time)/self.duration, 1))
			return self.func((time.time()-self.start_time)%self.duration/self.duration)
		elif self.is_pause:
			if not self.is_loop:
				return self.func(min((self.pause_time-self.start_time)/self.duration, 1))
			return self.func((self.pause_time-self.start_time)%self.duration/self.duration)
		return self.func(0)
	
	def pause(self):
		self.is_pause = True
		self.pause_time = time.time()

	def unpause(self):
		self.is_pause = False
		self.start_time, self.end_time = (time.time() - self.pause_time + self.start_time, 
			time.time() - self.pause_time + self.end_time)

	def stop(self):
		self.is_start = False
	
	def is_stop(self):
		return self.end_time <= time.time() and not self.is_loop or not self.is_start
