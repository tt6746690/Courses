
class MyRectangle:

	def __init__(self,x, y, width,height):			# def = definding a method
		''' Initialize this MyRectangle.

		@type self: MyRectangle
		@type x: int
						the x coordinate of top-left corner of this rectangle
		@type y: int
						the y coordinate of top-left corner of this rectangle
		@type width: int
						the width of this rectangle
		@type heigt: int
						the height of this rectangle
		'''
		self.x = x
		self.y = y
		self.width = width
		self.height = height
	def moveToRight(self, num):
		''' move the rectangle to the right by a number of pixels

		@type self: MyRectangle
		@type num: intergers
		@type: NoneType				change some property of class but does not return value
		'''
		self.x = self.x + num
