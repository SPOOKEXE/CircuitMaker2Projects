
import numpy as np
import os

from PIL import Image
from .components import ( MassiveMemory, MassMemory )
from .utility import ( ImageEditor, VideoEditor, number_to_nth_str )

class LEDEditor:

	@staticmethod
	def image_to_massivememory( image : Image.Image, threshold : int = 127, debug : bool = False ) -> str:
		'''Returns a list of strings for each row of pixels in the image.
		#### Assumes image is GRAYSCALE.'''
		#if debug: print('TO GRAYSCALE')
		#pixelated_image : Image.Image = ImageEditor.to_grayscale_pixelated( image, grid=LED16x16.GRID_SIZE, bgc=0, centered=True )
		if debug: print('ENCODING IMAGE')
		encoded_image : list[str] = [ ]
		for index, row in enumerate( np.array(image) ):
			if debug: print( index+1, row )
			row = [ v > threshold and '1' or '0' for v in row ]
			if debug: print( row )
			if debug: print( ''.join(row) )
			row = int(''.join(row), 2)
			row = MassiveMemory.number_to_massivememory( row )
			if debug: print( row )
			encoded_image.append( row )
		if debug: print('IMAGE HAS BEEN ENCODED')
		return ''.join(encoded_image)


class LED16x16:

	GRID_SIZE : int = 16

	@staticmethod
	def encode_image( source_image : Image.Image, threshold : int = 127, debug : bool = True ) -> str:
		'''Encode a single image to the 16x16 grid memory data. Same as a single frame from a video.'''
		memory : str = LEDEditor.image_to_massivememory( source_image, threshold=threshold, debug=debug )
		return MassiveMemory.fill_padding( memory )

	@staticmethod
	def encode_video( video_filepath : str, threshold : int = 127, MAX_FRAMES : int = 900, NTH_FRAME : int = 3, debug : bool = True ) -> str:
		'''Encode the video to the 16x16 grid memory data.'''
		video_frames : list[Image.Image] = VideoEditor.extract_frames_from_video( video_filepath, MAX_FRAMES=MAX_FRAMES, NTH_FRAME=NTH_FRAME, debug=debug )

		if debug: print(f'Grayscaling and downscaling frames.')
		frames : list[Image.Image] = [ ImageEditor.to_grayscale_pixelated( image, LED16x16.GRID_SIZE, bgc=0, centered=True ) for image in video_frames ]

		if debug:
			print(f'Encoding a total of { len(frames) } frames to debug video.')
			DEBUG_SIZE : int = 64
			head, _ = os.path.split( video_filepath )
			VideoEditor.output_debug_video( video_frames, head + f'/debug/{DEBUG_SIZE}.avi', size=DEBUG_SIZE, fps=10, threshold=threshold, debug=False )

		if debug: print(f'Encoding frames.')
		encoded_frames : list[str] = [ LEDEditor.image_to_massivememory( img, threshold=threshold, debug=False ) for img in frames ]

		if debug: print(f'Concatenating frames.')
		memory : str = ''.join( encoded_frames )
		padding : str = 'A' * ( MassiveMemory.MAX_CHARACTERS_PER_MEMORY - len(memory) - memory.count('\n') )

		if debug: print(f'Completed process.')
		return memory + padding

class LED32x32:

	GRID_SIZE = 32

	@staticmethod
	def encode_image( image : Image.Image, threshold : int = 127, debug : bool = True ) -> str:
		image : Image.Image = ImageEditor.to_grayscale_pixelated( image, LED32x32.GRID_SIZE, bgc=0, centered=True )
		chunks : list[Image.Image] = ImageEditor.split_image_to_quad_chunks( image, chunk_size=16 )

		if debug:
			for index, c in enumerate( chunks ):
				c.save(f'{index}_chunk.jpg')

		if debug: print(f'Encoding frames.')
		chunks : list[str] = [ LEDEditor.image_to_massivememory( img, threshold=threshold, debug=False ) for img in chunks ]

		if debug: print(f'Concatenating frames.')
		memory : str = ''.join( chunks )
		padding : str = 'A' * ( MassiveMemory.MAX_CHARACTERS_PER_MEMORY - len(memory) - memory.count('\n') )

		if debug: print(f'Completed process.')
		return memory + padding

	@staticmethod
	def encode_video( video_filepath : str, threshold : int = 127, MAX_FRAMES : int = 900, NTH_FRAME : int = 3, debug : bool = True ) -> str:
		'''Encode the video to the 32x32 grid memory data.'''
		video_frames : list[Image.Image] = VideoEditor.extract_frames_from_video( video_filepath, MAX_FRAMES=MAX_FRAMES, NTH_FRAME=NTH_FRAME )

		if debug: print(f'Grayscaling and downscaling frames.')
		frames : list[Image.Image] = [ ImageEditor.to_grayscale_pixelated( image, LED32x32.GRID_SIZE, bgc=0, centered=True ) for image in video_frames ]

		if debug:
				print(f'Encoding a total of { len(frames) } frames to debug video.')
				DEBUG_SIZE : int = 64
				head, _ = os.path.split( video_filepath )
				VideoEditor.output_debug_video( frames, head + f'/debug/{DEBUG_SIZE}.avi', size=DEBUG_SIZE, fps=10, threshold=threshold, debug=False )

		# split the frames into 4 quads, top-left, top-right, bottom-left, bottom-right and put in memory in that order
		# the counter circuit will automatically pick the correct quads and the order.
		frames : list[Image.Image] = []

		chunks_array : list[Image.Image] = [ ImageEditor.split_image_to_quad_chunks( frame, chunk_size=16 ) for frame in frames ]
		for image_chunks in chunks_array:
			frames.extend( image_chunks )

		if debug: print(f'Encoding frames.')
		encoded_frames : list[str] = [ LEDEditor.image_to_massivememory( img, threshold=threshold, debug=False ) for img in frames ]

		if debug: print(f'Concatenating frames.')
		memory : str = ''.join( encoded_frames )
		padding : str = 'A' * ( MassiveMemory.MAX_CHARACTERS_PER_MEMORY - len(memory) - memory.count('\n') )

		if debug: print(f'Completed process.')
		return memory + padding
