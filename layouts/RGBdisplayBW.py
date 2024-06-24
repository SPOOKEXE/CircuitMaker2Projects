
# massive_memory_output has binary 0b0000000000000000 (16-bit)

from PIL import Image

import numpy as np

class MassiveMemory:
	'''16-bit memory.'''

	BASE64 : str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
	MAX_CHARACTERS_PER_MEMORY : int = 200000

	@staticmethod
	def number_to_massivememory( number : int ) -> str:
		'''Encode the given number to the 16bit custom base64.'''
		d3 : int = (number & 0b111111)
		d2 : int = (number & 0b111111_000000) >> 6
		d1 : int = (number & 0b111111_000000_000000) >> 12
		return MassiveMemory.BASE64[d3] + MassiveMemory.BASE64[d2] + MassiveMemory.BASE64[d1]

	@staticmethod
	def fill_padding( memory : str ) -> str:
		length : int = MassiveMemory.MAX_CHARACTERS_PER_MEMORY - len(memory)
		return memory + ( MassiveMemory.BASE64[0] * length )

def clamp(v : int | float, minn : int | float, maxx : int | float) -> int | float:
	return min(max(v, minn), maxx)

def binl( value : int, length : int = 5 ) -> str:
	item : str = '{0:0' + str(length) + 'b}'
	return item.format(value)

def encoded_image( image : Image.Image ) -> list[str]:
	'''List of rows, each row having a pixel coordinate + B/W value'''
	image.thumbnail((127, 127))
	image = image.convert('L')
	image.save('layouts/test.PNG')
	rows : list[list[str]] = []
	for index, row in enumerate(np.array(image)):
		columns : list[str] = []
		for column_n, c in enumerate(row):
			on_off = (c > 127) and '1' or '0'
			columns.append( binl( column_n, length=7 ) + binl( index, length=7 ) + on_off + '0' )
		rows.append( columns )
	with open('layouts/bw_chunks.txt', 'w') as file:
		file.write('\n'.join([str(v) for v in rows]))
	rows = [ MassiveMemory.number_to_massivememory(int(v, base=2)) for item in rows for v in item ]
	with open('layouts/video_memory.txt', 'w') as file:
		file.write('\n'.join(rows))
	return rows

def encode_image_rgbBWDisplay_massivememory( image : Image.Image ) -> str:
	memory : str = ''.join(encoded_image( image))
	padding : str = 'A' * ( MassiveMemory.MAX_CHARACTERS_PER_MEMORY - len(memory) - memory.count('\n') )
	return memory + padding

# def rgb_to_binary_str( rgb : tuple[int, int, int] ) -> str:
# 	'''Every 5 bits represents a number from 0 -> 63.'''
# 	# binary
# 	bmask = 0b11111
# 	# rgb bits
# 	red_bits = rgb[0] & bmask
# 	green_bits = rgb[1] & bmask
# 	blue_bits = rgb[2] & bmask
# 	print(binl(red_bits, length=5), binl(green_bits, length=5), binl(blue_bits, length=5))
# 	# combine
# 	return binl(red_bits) + binl(green_bits) + binl(blue_bits)

# def to_pixelized_encoded_image( image : Image.Image ) -> list[str]:
# 	image.thumbnail((64, 64))
# 	image.save('layouts/test.PNG')
# 	# encode image
# 	chunks = []
# 	for row in np.array(image):
# 		for rgb in row:
# 			# print(rgb)
# 			rgb_number = int(rgb_to_binary_str(rgb))
# 			chunks.append(rgb_number)
# 	with open('layouts/rgb_chunks.txt', 'w') as file:
# 		file.write('\n'.join([str(item) for item in chunks]))
# 	chunks = [ MassiveMemory.number_to_massivememory( item ) for item in chunks ]
# 	# encode memory
# 	memory : str = ''.join( chunks )
# 	padding : str = 'A' * ( MassiveMemory.MAX_CHARACTERS_PER_MEMORY - len(memory) - memory.count('\n') )
# 	return memory + padding

encoded_video = encode_image_rgbBWDisplay_massivememory( Image.open('tests\Capture.PNG') )
with open( 'layouts/video_memory.txt', 'w') as file:
	file.write( encoded_video )
