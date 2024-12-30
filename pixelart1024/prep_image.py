
from math import sqrt
from PIL import Image

import numpy as np

def round_img_to_nearest(img : Image.Image, n : int) -> Image.Image:
	# Convert image to numpy array
	img_array = np.array(img, dtype=np.float32)  # Use float32 for intermediate calculations
	# Round each pixel to the nearest nth value
	rounded_array = np.clip(np.round(img_array / n) * n, 0, 255).astype(np.uint8)
	# Convert back to PIL image
	rounded_image = Image.fromarray(rounded_array, mode=img.mode)
	return rounded_image

def split_image_into_rectangles(image: Image.Image, rect_width=128, rect_height=128, grid_size_squared=4) -> list[np.ndarray]:
	# Calculate the desired dimensions
	desired_width = rect_width * grid_size_squared
	desired_height = rect_height * grid_size_squared

	# Crop the image to fit the desired dimensions
	image.thumbnail((desired_width, desired_height))
	image = image.crop((0, 0, desired_width, desired_height))

	# Convert the cropped image to a NumPy array
	image_array = np.array(image)
	img_height, img_width, channels = image_array.shape

	# Calculate the number of rectangles in each direction
	rows = img_height // rect_height
	cols = img_width // rect_width

	# Create an array to store the rectangles
	rectangles = []

	# Split the image into rectangles
	for row in range(rows):
		for col in range(cols):
			rect = image_array[
				row * rect_height: (row + 1) * rect_height,
				col * rect_width: (col + 1) * rect_width,
				:
			]
			rectangles.append(rect)

	# Fill remaining rectangles with black if total rectangles aren't enough
	blank_rectangle = np.zeros((rect_height, rect_width, channels), dtype=np.uint8)
	while len(rectangles) < grid_size_squared * grid_size_squared:
		rectangles.append(blank_rectangle)

	return np.array(rectangles)

# prepare the image
img = Image.open('pixelart1024/test.png')
img = img.convert('RGB')
img.thumbnail((1024,1024), Image.Resampling.BICUBIC)
img = round_img_to_nearest(img, 2)
img.save('pixelart1024/test-out.jpg')

# get each section
sections : list[Image.Image] = split_image_into_rectangles(img, rect_width=256, rect_height=256, grid_size_squared=4)
for index, item in enumerate(sections):
	imgcrop = Image.fromarray(item).convert('RGB')
	imgcrop.save(f"pixelart1024/crop_{index}.jpg")

# convert to data
class HugeMemory:

	@staticmethod
	def encode(num):
		chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/"
		num1 = chars[(num & 0b1111000000000000) >> 12]
		num2 = chars[(num & 0b0000111111000000) >> 6]
		num3 = chars[num & 0b0000000000111111]
		return num1 + num2 + num3

	@staticmethod
	def encode_to_memory(bin_array: np.ndarray) -> str:
		'''Convert binary array to memory.'''
		# Convert binary to integers in a vectorized manner
		numbers = np.array([int(binarystr, 2) for binarystr in bin_array])
		# Get massive memory encoding from the array and return it
		massive_memory = [HugeMemory.encode(n) for n in numbers]
		return "".join(massive_memory)

def fast_binary_conversion(pixel_array : np.ndarray) -> np.ndarray:
	# Ensure the array is of integer type
	pixel_array = pixel_array.astype(np.uint8)
	# Convert the array to binary with a fixed width of 5 bits
	bin_array = np.vectorize(lambda x: f"{x:05b}")(pixel_array)
	# Combine each row into a single 15-bit binary string
	bin_array = np.apply_along_axis(lambda row: "".join(row), axis=1, arr=bin_array)
	return bin_array

memories : list[np.ndarray] = []
for index, section in enumerate(sections):
	print(f'Section_{index}')
	mem_section = []
	# get all pixels
	pixel_array = np.array(Image.fromarray(section).convert('RGB').getdata())
	print(f'Got a total pixel count of {len(pixel_array)}.')
	# divide all pixels by 32 to round to a 2^6 number
	pixel_array = np.clip(np.round(pixel_array / 32), 0, 32).astype(np.uint8)
	# shift bits right one so you store the bit 2,4,8,16,32 excluding 1s
	pixel_array = np.right_shift(pixel_array, 1)
	# now get the bin of the number and pad up to 5 digits total
	# and combine the numbers to form a array of binary of 15 bits length
	bin_array = fast_binary_conversion(pixel_array)
	print(bin_array[0])
	# now encode to memory and store it
	memories.append(HugeMemory.encode_to_memory(bin_array))

for index, mem_value in enumerate(memories):
	with open(f"pixelart1024/memory_{index}.txt", "w") as file:
		file.write(mem_value)
