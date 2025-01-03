
import json
import os
from PIL import Image

import numpy as np
import base64
import zlib

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

def round_img_to_nearest(img : Image.Image, n : int) -> Image.Image:
	# Convert image to numpy array
	img_array = np.array(img, dtype=np.float32)  # Use float32 for intermediate calculations
	# Round each pixel to the nearest nth value
	rounded_array = np.clip(np.round(img_array / n) * n, 0, 255).astype(np.uint8)
	# Convert back to PIL image
	rounded_image = Image.fromarray(rounded_array, mode=img.mode)
	return rounded_image

def split_image_into_crops(image: Image.Image, rect_width : int = 128, rect_height : int = 128, grid_size_squared=4) -> list[np.ndarray]:
	# Calculate the desired dimensions
	desired_width : int = int(rect_width * grid_size_squared)
	desired_height : int = int(rect_height * grid_size_squared)
	# Crop the image to fit the desired dimensions
	image.thumbnail((desired_width, desired_height))
	image = image.crop((0, 0, desired_width, desired_height))
	# Convert the cropped image to a NumPy array
	image_array : np.ndarray = np.array(image)
	img_height, img_width, channels = image_array.shape
	# Calculate the number of rectangles in each direction
	rows : int = int(img_height // rect_height)
	cols : int = int(img_width // rect_width)
	# Create an array to store the rectangles
	rectangles : list[Image.Image] = []
	# Split the image into rectangles
	for row in range(rows):
		for col in range(cols):
			rectangles.append(image_array[
				row * rect_height: (row + 1) * rect_height,
				col * rect_width: (col + 1) * rect_width,
				:
			])
	# fill remaining rectangles with black if total rectangles aren't enough
	blank_rectangle = np.zeros((rect_height, rect_width, channels), dtype=np.uint8)
	while len(rectangles) < grid_size_squared * grid_size_squared:
		rectangles.append(blank_rectangle)
	return np.array(rectangles)

def image_to_huge_memory(index : int, image : Image.Image) -> tuple[str, list]:
	# pixel data
	pixel_array = np.array(Image.fromarray(image).convert('RGB').getdata()).astype(np.uint8)
	print(f'Got a total pixel count of {len(pixel_array)}.')
	# round down to 6 bits
	pixel_array = np.clip(np.round(pixel_array / 31), 0, 31).astype(np.uint8)
	# convert to binary
	bin_array : np.ndarray[str] = np.vectorize(lambda x: f"{x:05b}")(pixel_array)
	bin_array : np.ndarray[str] = np.apply_along_axis(lambda row: "0" + "".join(row), axis=1, arr=bin_array)
	print(bin_array[:5])
	# with open(f"pixelart1024/temp/raw_{index}.txt", "w") as file:
	# 	file.write(json.dumps(bin_array.tolist(), indent=4))
	int_array : list[int] = [int(item, base=2) for item in bin_array]
	print(int_array[:5])
	mem : str = HugeMemory.numbers_to_hugememory(int_array)
	print(mem[:5])
	return mem, bin_array

def reconstruct_from_memory(index : int, bits : str, im_size_sqr : int) -> Image.Image:
	print(f'Reconstructing {index}.')
	pixels = []
	for index, binary in enumerate(bits):
		if index == 0:
			print(binary)
		# 16 bits
		r = int(binary[1:6], base=2) # blue
		g = int(binary[6:11], base=2) # green
		b = int(binary[11:], base=2) # red
		if index == 0:
			print(binary[1:6], binary[6:11], binary[11:])
			print(r, g, b)
			print(r * 31, g * 31, b * 31)
		pixels.append([r, g, b])
	pixels = np.array(pixels, dtype=np.uint8) * 31
	pixels = np.clip(pixels, 0, 255)
	section = pixels.reshape(im_size_sqr, im_size_sqr, 3)
	return Image.fromarray(section)

def recreate_image_from_crops(rectangles: list[np.ndarray], rect_width: int = 128, rect_height: int = 128, grid_size_squared: int = 4) -> Image.Image:
	# Determine the total number of rows and columns
	rows = cols = int(grid_size_squared)

	# Ensure the number of rectangles matches the expected grid size
	assert len(rectangles) == rows * cols, "Number of rectangles does not match the grid size."

	# Create a blank array to hold the full image
	full_image_height = rows * rect_height
	full_image_width = cols * rect_width
	channels = rectangles[0].shape[2]  # Assuming all rectangles have the same number of channels

	full_image_array = np.zeros((full_image_height, full_image_width, channels), dtype=np.uint8)

	# Stitch rectangles back into the full image array
	for i, rect in enumerate(rectangles):
		row = i // cols
		col = i % cols
		full_image_array[
			row * rect_height: (row + 1) * rect_height,
			col * rect_width: (col + 1) * rect_width,
			:
		] = rect

	# Convert the NumPy array back to a PIL Image
	reconstructed_image = Image.fromarray(full_image_array)
	return reconstructed_image

def main() -> None:
	os.makedirs("pixelart1024/temp", exist_ok=True)
	img : Image.Image = Image.open('pixelart1024/test.jpg')
	img = img.convert('RGB')
	img.thumbnail((256,256), Image.Resampling.BICUBIC)
	img = round_img_to_nearest(img, 2)
	img.save('pixelart1024/test-out-rounded.jpg')

	# crop the image into 256x256 sections
	crops : list[Image.Image] = split_image_into_crops(img, rect_width=128, rect_height=128, grid_size_squared=2)
	for index, item in enumerate(crops):
		crop = Image.fromarray(item).convert('RGB')
		crop.save(f"pixelart1024/temp/crop_{index}.jpg")

	# convert crops to pixel binary data
	int_values : list[list] = []
	for index, crop in enumerate(crops):
		print(f'Convert {index} to bin data.')
		mem_str, int_array = image_to_huge_memory(index, crop)
		int_values.append(int_array)
		with open(f"pixelart1024/temp/memory_{index}.txt", "w") as file:
			file.write(mem_str)

	# reconstruct the image back
	sections : np.ndarray = []
	for index, bits in enumerate(int_values):
		print(f'Convert {index} to bin data.')
		img = reconstruct_from_memory(index, bits, 128)
		img.save(f"pixelart1024/temp/{index}_reconstructed_section.jpg")
		sections.append(np.array(img))

	rejoined = recreate_image_from_crops(sections, rect_width=128, rect_height=128, grid_size_squared=2)
	rejoined.save("pixelart1024/test-out-reconstructed.jpg")

def main2() -> None:
	image = Image.open("pixelart1024/")
	image_to_huge_memory(0, image)

	pass

if __name__ == '__main__':
	main()
