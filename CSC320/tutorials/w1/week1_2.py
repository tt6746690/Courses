import numpy as np
import cv2, cv

def imshow(img):
    cv2.imshow("windows name", img)
    cv2.waitKey(0)

# Read an image using imread
img = cv2.imread("pie.png")
# Show an image
imshow(img)
print(img.shape)
# we can get some subarea of the image
imshow(img[100:200, 100:200, :])
# we can also change the intensity of some area on the image
img[100:150, 50:100, :] = np.zeros([50, 50, 3])
imshow(img)
# or make the area white
img[100:150, 50:100, :] = 255
imshow(img)
# or red
img[100:150, 50:100, 0] = 0 # the 0 before "]" stands for blue channel
img[100:150, 50:100, 1] = 0 # the 1 before "]" stands for green channel, and 2 stands for red channel
# Note that the color channels are in order blue, green and red, although we generally use RGB (red, green, blue)
imshow(img)

# Read an image in greyscale using imread
img = cv2.imread("pie.png", cv2.IMREAD_GRAYSCALE)
print(img.dtype) # uint8, which means "unsigned integer 8 bits", so each value must between 0 and 255
imshow(img)
print(img.shape) # 2 dimentional since color dimension is removed

# imread function will read an image in to an array, so we can do array operation on the image
# flip the image upside-down
img_flipped = img[::-1, :]
imshow(img_flipped)
# make the image brighter
img_brighter = img + 100  # will overflow since the max possible intensity for a pixel is 255
imshow(img_brighter)
img_brighter_2 = img + (255 - img) / 2  # will make the image brighter, but with lower contrast
imshow(img_brighter_2)

img_darker = img.astype(np.uint16) # changing the value type will make the image much darker
imshow(img_darker)

img = cv2.imread("pie.png") # Note that this is a colored image
# We can draw a rectangle on the image
img_rectangle = img.copy()
# (50, 50) is the top left corner, (100, 200) is the lower right corner,
# (255, 0, 0) is the color (and brightness), 5 is the thickness of the sides
cv2.rectangle(img_rectangle, (50, 50), (100, 200), (255, 0, 0), 5)
imshow(img_rectangle)

# Or a line
img_line = img.copy()
# (50, 50) and (100, 200) are the two end points of the line segment, (255, 0, 0) is the color, 5 is the thickness
cv2.line(img_line, (50, 50), (100, 200), (255, 0, 0), 5)
imshow(img_line)

# Or a circle
img_circle = np.zeros([300, 300], dtype=np.uint8) # create a black image of size 300*300, note that we don't have color channel, so this is a grayscale image
imshow(img_circle)
# (150, 120) is the center of the circle, 50 is the radius, 150 is the brightness, and -1 stands for making the circle a solid one
cv2.circle(img_circle, (150, 120), 50, 150, -1)
imshow(img_circle)
# similar as before, but the last parameter 10 is the thickness
cv2.circle(img_circle, (150, 120), 100, 255, 10)
imshow(img_circle)

# create a new image file on the computer, file name is circle, type is jpg, and the image content is from matrix circle
cv2.imwrite("circle.jpg", img_circle)
