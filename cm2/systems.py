
import numpy as np
import os

from PIL import Image
from .components import ( MassiveMemory, MassMemory )
from .utility import ( ImageEditor, VideoEditor, number_to_nth_str )

class LED16x16:

	GRID_SIZE : int = 16

	@staticmethod
	def image_to_massivememory( image : Image.Image, threshold : int = 127, debug : bool = False ) -> str:
		'''Returns a list of strings for each row of pixels in the image.
		#### Assumes image is GRAYSCALE.'''
		if debug: print('TO GRAYSCALE')
		pixelated_image : Image.Image = ImageEditor.to_grayscale_pixelated( image, grid=LED16x16.GRID_SIZE, bgc=0, centered=True )
		if debug: print('ENCODING IMAGE')
		encoded_image : list[str] = [ ]
		for index, row in enumerate(np.array( pixelated_image )):
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

	@staticmethod
	def encode_image( source_image : Image.Image, threshold : int = 127, debug : bool = True ) -> str:
		'''Encode a single image to the 16x16 grid memory data. Same as a single frame from a video.'''
		memory : str = LED16x16.image_to_massivememory( source_image, threshold=threshold, debug=debug )
		return MassiveMemory.fill_padding( memory )

	@staticmethod
	def output_debug_video( frames : list[Image.Image], filepath : str, size : int = 64, fps : int = 10, threshold : int = 127, debug : bool = False ) -> None:
		debug_frames =  [ ImageEditor.to_grayscale_pixelated( image, size, bgc=0, centered=True ) for image in frames ]
		debug_frames : list[Image.Image] = [ f.resize((size, size), Image.NEAREST) for f in debug_frames ]
		debug_frames : list[Image.Image] = [ f.point( lambda p: 255 if p > threshold else 0 ) for f in debug_frames ]
		VideoEditor.frames_to_video( debug_frames, (size, size), filepath, fps=fps, debug=debug )

	@staticmethod
	def encode_video( video_filepath : str, threshold : int = 127, MAX_FRAMES : int = 900, NTH_FRAME : int = 3, debug : bool = True ) -> str:
		'''Encode the video to the 16x16 grid memory data.'''
		assert os.path.exists( video_filepath ), f'Video at "{video_filepath}" does not exist.'

		if debug: print('Extracting Frames:', video_filepath)
		video_frames : list[Image.Image] = VideoEditor.extract_frames_from_video( video_filepath, MAX_FRAMES=MAX_FRAMES )
		if debug: print(f'A total of {len(video_frames)} frames have been extracted from the video.')

		if NTH_FRAME != 1:
			if debug: print(f'Gathering every { number_to_nth_str( NTH_FRAME ) } frame.')
			video_frames : list[Image.Image] = video_frames[ 0::NTH_FRAME ]
			if debug: print(f'A total of {len(video_frames)} frames are remaining after frame skip.')

		if debug: print(f'Grayscaling and downscaling frames.')
		frames : list[Image.Image] = [ ImageEditor.to_grayscale_pixelated( image, LED16x16.GRID_SIZE, bgc=0, centered=True ) for image in video_frames ]

		if debug:
			print(f'Encoding a total of { len(frames) } frames to debug video.')
			DEBUG_SIZE : int = 48
			head, _ = os.path.split( video_filepath )
			LED16x16.output_debug_video( video_frames, head + f'/debug/{DEBUG_SIZE}.avi', size=DEBUG_SIZE, fps=10, threshold=threshold, debug=False )

		if debug: print(f'Encoding frames.')
		encoded_frames : list[str] = [ LED16x16.image_to_massivememory( img, threshold=threshold, debug=False ) for img in frames ]

		if debug: print(f'Concatenating frames.')
		memory : str = ''.join( encoded_frames )
		padding : str = 'A' * ( MassiveMemory.MAX_CHARACTERS_PER_MEMORY - len(memory) - memory.count('\n') )

		if debug: print(f'Completed process.')
		return memory + padding

class LED32x32:
	pass
