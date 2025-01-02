
from math import sqrt
from PIL import Image

import numpy as np
import base64
import zlib

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
	def numbers_to_hugememory(numbers : list[str]) -> str:
		MAX_LENGTH = 2**16
		MAX_VALUE = (2**16) - 1

		modified_numbers = []
		for num in numbers:
			if num > MAX_VALUE:
				print("a number too big so im gonna set it to 65535 k?")
				print(num)
				modified_numbers.append(MAX_VALUE)
			else:
				modified_numbers.append(num)

		if len(modified_numbers) > MAX_LENGTH:
			print("list too long so im gonna truncate it k?")
			modified_numbers = modified_numbers[:MAX_LENGTH]

		if len(modified_numbers) < MAX_LENGTH:
			modified_numbers.extend([0] * (MAX_LENGTH - len(modified_numbers)))

		data_to_encode = b''.join(num.to_bytes(2, 'little') for num in modified_numbers)
		compressed_data = zlib.compress(data_to_encode, level=9)[2:-4]  # strip zlib header/footer
		encoded_chunk = base64.b64encode(compressed_data).decode()
		return encoded_chunk

def fast_binary_conversion(pixel_array : np.ndarray) -> np.ndarray:
	# Ensure the array is of integer type
	pixel_array = pixel_array.astype(np.uint8)
	# Convert the array to binary with a fixed width of 5 bits
	bin_array = np.vectorize(lambda x: f"{x:06b}")(pixel_array)
	# Combine each row into a single 15-bit binary string
	bin_array = np.apply_along_axis(lambda row: "0" + "".join(row), axis=1, arr=bin_array)
	return bin_array

# TODO fix below

raw_bits : list[list[str]] = []
memories : list[str] = []
for index, section in enumerate(sections):
	print(f'Section_{index}')
	mem_section = []
	# get all pixels
	pixel_array = np.array(Image.fromarray(section).convert('RGB').getdata())
	print(f'Got a total pixel count of {len(pixel_array)}.')
	# divide all pixels by 32 to round to a 2^6 number
	pixel_array = np.clip(np.round(pixel_array / 32), 0, 31).astype(np.uint8) # Use 31 for 5-bit max value
	# now get the bin of the number and pad up to 5 digits total
	# and combine the numbers to form a array of binary of 15 bits length
	bin_array = fast_binary_conversion(pixel_array)
	raw_bits.append(bin_array)
	print(bin_array[0])
	int_array = [int(item, base=2) for item in bin_array]
	# now encode to memory and store it
	memories.append(HugeMemory.numbers_to_hugememory(int_array))

for index, mem_value in enumerate(memories):
	with open(f"pixelart1024/memory_{index}.txt", "w") as file:
		file.write(mem_value)

for index, array_of_bits in enumerate(raw_bits):
	print(f'Reconstructing {index}.')
	pixels = []
	for binary in array_of_bits:
		r = int(binary[:5], base=2) # First 5 bits for red
		g = int(binary[5:10], base=2) # Next 5 bits for green
		b = int(binary[10:], base=2) # Last 5 bits for blue
		pixels.append([r, g, b])
	pixels = np.array(pixels, dtype=np.uint8) * 32
	pixels = np.left_shift(pixels, 1)
	pixels = np.clip(pixels, 0, 255)
	section = pixels.reshape(256, 256, 3)
	reconstructed_image = Image.fromarray(section)
	reconstructed_image.save(f"pixelart1024/{index}_reconstructed_section.png")
