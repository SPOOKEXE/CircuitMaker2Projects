
import numpy as np
import cv2

from PIL import Image

BS4 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

# 16-bit output
def number_to_massivememory( number : int ) -> str:
	d3 = number & 0b111111
	d2 = (number & 0b111111_000000) >> 6
	d1 = (number & 0b111111_000000_000000) >> 12
	return BS4[d3] + BS4[d2] + BS4[d1]

def split_to_chunks( array : list, length : int, padd : bool = True ) -> list:
	rem : int = len(array) % length
	if rem != 0 and padd == True:
		array.extend(['0']*rem)
	return [ array[i:i+length] for i in range(0, len(array), length) ]

def convert_to_grayscale_grid( image : Image.Image, grid : int = 16, bgc : int = 0, centered : bool = True ) -> Image.Image:
	image = image.convert('L')
	image.thumbnail( (grid, grid), Image.BILINEAR )
	img : Image.Image = Image.new('L', (grid, grid), bgc)
	img.paste( image, ( 8 - round( image.size[0] / 2 ), ( 8 - round( image.size[1] / 2 ) ) ) )
	return img

def grayscale_image_to_encoded_massivememory( image : Image.Image, output_bits : int = 16 ) -> str:
	'''Returns a list of strings for each row of pixels in the image.
	#### Assumes image is GRAYSCALE.'''
	print('ENCODING NEW IMAGE')
	encoded_image : list[str] = [ ]
	for index, row in enumerate(np.array( image )):
		print( index+1, row )
		# pixels : list[int] = [ v > 127 and WHITE_COLOR or BLACK_COLOR for v in row ]
		row = [ v > 127 and '1' or '0' for v in row ]
		print( row )
		row = int(''.join(row), 2)
		print( row )
		row = number_to_massivememory( row )
		print( row )
		encoded_image.append( row )
	print('IMAGE HAS BEEN ENCODED')
	return ''.join(encoded_image)

def extract_frames_from_video( filepath : str, max_frames : int = -1 ) -> list:
	frames : list[Image.Image] = []
	capture = cv2.VideoCapture( filepath )
	while True:
		success, vframe = capture.read()
		if success == False: break
		frame : Image.Image = Image.fromarray( cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) )
		frames.append(frame)
		if max_frames != -1 and len(frames) >= max_frames: break
	capture.release()
	return frames

def encode_image_to_massivememory(
	source_image : Image.Image,
	GRID_SIZE : int = 16,
	MAX_CHARACTERS_PER_MEMORY : int = 200000
) -> str:
	pixelated_image : Image.Image = convert_to_grayscale_grid( source_image, grid=GRID_SIZE, bgc=0, centered=True )
	pixelated_image.save('video-encoder/pixelated.jpg')

	memory : str = grayscale_image_to_encoded_massivememory( pixelated_image )
	padding : str = 'A' * ( MAX_CHARACTERS_PER_MEMORY - len(memory) - memory.count('\n') )
	return memory + padding

def encode_video_to_massivememory(
	video_filepath : str,
	GRID_SIZE : int = 16,
	MAX_FRAMES : int = 1500,
	NTH_FRAME : int = 3,
	MAX_CHARACTERS_PER_MEMORY : int = 200000
) -> str:
	DEFAULT_SUFFIX = 'th'
	CUSTOM_SUFFIX = { '2' : 'nd', '3' : 'rd' }

	print('Extracting Frames:', video_filepath)
	frames : list[Image.Image] = extract_frames_from_video( video_filepath, max_frames=MAX_FRAMES )

	print(f'Gathering every { NTH_FRAME }{ CUSTOM_SUFFIX.get(str(NTH_FRAME)) or DEFAULT_SUFFIX } frame.')
	frames : list[Image.Image] = frames[ 0::NTH_FRAME ]

	print(f'Grayscaling and downscaling frames.')
	frames : list[Image.Image] = [ convert_to_grayscale_grid( img, grid=GRID_SIZE, bgc=0, centered=True ) for img in frames ]

	print(f'Encoding frames.')
	encoded_frames : list[str] = [ grayscale_image_to_encoded_massivememory(img) for img in frames ]

	print(f'Concatenating frames.')
	memory : str = ''.join( encoded_frames )
	padding : str = 'A' * ( MAX_CHARACTERS_PER_MEMORY - len(memory) - memory.count('\n') )

	print(f'Completed process.')
	return memory + padding

if __name__ == '__main__':

	# video_filepath : str = 'video-encoder/flashing_1.mp4'
	video_filepath : str = 'video-encoder/badapple_compact.mp4'
	encoded_video : str = encode_video_to_massivememory( video_filepath, NTH_FRAME=10 )
	with open('video-encoder/0_memory.txt', 'w') as file:
		file.write( encoded_video )
