
import os
import sys

from PIL import Image

FILE_DIRECTORY : str = os.path.dirname( os.path.realpath(__file__) )
sys.path.append( os.path.join( FILE_DIRECTORY, '..' ) )

from cm2 import (
	# utility
	ImageEditor, VideoEditor, number_to_nth_str, split_into_chunks,
	# components
	MassiveMemory, MassMemory,
	# systems (save files)
	LED16x16, LED32x32,
)

sys.path.pop()

if __name__ == '__main__':

	### VIDEO ENCODER ###
	VIDEO_FILENAME : str = 'badapple_compact.mp4'
	# VIDEO_FILENAME : str = 'flashing_1.mp4'
	video_filepath : str = os.path.join( FILE_DIRECTORY, VIDEO_FILENAME )
	encoded_video : str = LED32x32.encode_video( video_filepath, NTH_FRAME=9, MAX_FRAMES=1000, debug=True )
	with open( FILE_DIRECTORY + '/video_memory.txt', 'w') as file:
		file.write( encoded_video )

	### IMAGE ENCODER ###
	# IMAGE_FILENAME : str = 'Capture.PNG'
	# image_filepath : str = os.path.join( FILE_DIRECTORY, IMAGE_FILENAME )
	# encoded_image : str = LED32x32.encode_image( Image.open(image_filepath) )
	# with open( FILE_DIRECTORY + '/image_memory.txt', 'w') as file:
	# 	file.write( encoded_image )

	### DEBUG ###
	# IMAGE_FILENAME : str = 'Capture.PNG'
	# image_filepath : str = os.path.join( FILE_DIRECTORY, IMAGE_FILENAME )

	# image : Image.Image = Image.open( image_filepath )
	# for grid_size in [ 16, 32, 48, 64 ]:
	# 	image : Image.Image = ImageEditor.to_grayscale_pixelated(image, grid=grid_size, bgc=0, centered=True)
	# 	image.save(FILE_DIRECTORY + f'/test_{grid_size}.jpg')
