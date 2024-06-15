
import traceback
import cv2
import numpy as np
import os

from typing import Any
from PIL import Image
from math import floor

def number_to_nth_str( number : int ) -> str:
	DEFAULT_SUFFIX : str = 'th'
	CUSTOM_SUFFIX : dict[str, str] = { '2' : 'nd', '3' : 'rd' }
	return f'{ number }{ CUSTOM_SUFFIX.get(str(number) ) or DEFAULT_SUFFIX }'

def split_into_chunks( array : list[Any], length : int, pad : Any = None ) -> list[Any]:
	rem : int = len(array) % length
	values = [ array[i:i+length] for i in range(0, len(array), length) ]
	if rem != 0 and pad != None: values.extend([pad]*rem)
	return values

class ImageEditor:

	@staticmethod
	def to_cv2( img : Image.Image ) -> np.ndarray:
		return cv2.cvtColor( np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR )

	@staticmethod
	def to_PIL( img : np.ndarray, mode : str = 'RGB' ) -> Image.Image:
		return Image.fromarray( cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode=mode )

	@staticmethod
	def gray_to_RGB( image : Image.Image ) -> Image.Image:
		return ImageEditor.to_PIL( cv2.cvtColor( ImageEditor.to_cv2( image ), cv2.COLOR_GRAY2RGB) )

	@staticmethod
	def get_center_xy( size : tuple[int, int], padding : int ) -> tuple[int, int]:
		return int(padding / 2) - round( size[0] / 2 ), int(padding / 2) - round( size[1] / 2 )

	@staticmethod
	def to_grayscale_pixelated( image : Image.Image, grid : int = 128, bgc : int = 0, centered : bool = True ) -> Image.Image:
		image = image.convert('L')
		# aspect : float = image.size[0] / image.size[1]
		# xy : tuple[int, int] = ( int(min(grid * aspect, grid)), int(min(grid * aspect, grid)) )
		image.thumbnail( (grid, grid), Image.BILINEAR )
		center : tuple[int, int] = (centered == True) and ImageEditor.get_center_xy( image.size, grid ) or (0, 0)
		img : Image.Image = Image.new('L', (grid, grid), bgc)
		img.paste( image, center )
		return img

	@staticmethod
	def split_image_to_quad_chunks( image : Image.Image, chunk_size : int = 16 ) -> list[Image.Image]:
		image_size : tuple = image.size
		if image_size[0] % chunk_size != 0 and image_size[1] % chunk_size != 0:
			raise ValueError('The video frames do not divide evenly into the chunk size. Cannot proceed.')
		coordinates : list[tuple] = [
			(xindex * chunk_size, yindex * chunk_size, (xindex+1) * chunk_size, (yindex+1) * chunk_size)
			for yindex in range( round(image_size[1] / chunk_size) )
			for xindex in range( round(image_size[0] / chunk_size) )
		]
		return [ image.crop( coordinate ) for coordinate in coordinates ]

class VideoEditor:

	@staticmethod
	def extract_frames_from_video( filepath : str, MAX_FRAMES : int = -1, NTH_FRAME : int = -1 ) -> list[Image.Image]:
		assert os.path.exists( filepath ), f'Video at "{filepath}" does not exist.'
		frames : list[Image.Image] = []
		capture = cv2.VideoCapture( filepath )
		while True:
			success, vframe = capture.read()
			if success == False: break
			frame : Image.Image = Image.fromarray( cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) )
			frames.append(frame)
			if MAX_FRAMES != -1 and len(frames) >= MAX_FRAMES: break
		capture.release()
		if NTH_FRAME != 1: frames : list[Image.Image] = frames[ 0::NTH_FRAME ]
		return frames

	@staticmethod
	def frames_to_video( frames : list[Image.Image], video_size : tuple[int, int], filepath : str, fps : int = 1, debug : bool = False ) -> bool:
		try:
			print(f'Writing debug video of size {str(video_size)} to {filepath}.')
			codec = cv2.VideoWriter_fourcc(*'MPEG')
			video = cv2.VideoWriter(filepath, codec, fps, video_size, 1)
			blank : Image.Image = Image.new('RGBA', video_size, color=0)
			os.makedirs( os.path.split( filepath )[0], exist_ok=True )
			for index, frame in enumerate(frames):
				imtemp = blank.copy()
				imtemp.paste( frame, (0, 0) )
				b4 = cv2.cvtColor( np.array(imtemp), cv2.COLOR_RGB2BGR )
				# if debug: cv2.imwrite( f'{ os.path.split( filepath )[0] }/frame_{index}.jpg', b4 )
				video.write( b4 )
			video.release()
			return True
		except Exception as exception:
			print('Failed to write frames to video:')
			traceback.print_exception( exception )
			return False

	@staticmethod
	def output_debug_video( frames : list[Image.Image], filepath : str, size : int = 64, fps : int = 10, threshold : int = 127, debug : bool = False ) -> None:
		debug_frames =  [ ImageEditor.to_grayscale_pixelated( image, size, bgc=0, centered=True ) for image in frames ]
		debug_frames : list[Image.Image] = [ f.resize((size, size), Image.NEAREST) for f in debug_frames ]
		debug_frames : list[Image.Image] = [ f.point( lambda p: 255 if p > threshold else 0 ) for f in debug_frames ]
		VideoEditor.frames_to_video( debug_frames, (size, size), filepath, fps=fps, debug=debug )

def extract_frames_to_directory( filepath : str, directory : str, NTH_FRAME : int = 5, MAX_FRAMES : int = -1 ) -> None:
	capture = cv2.VideoCapture( filepath )
	length = int( capture.get(cv2.CAP_PROP_FRAME_COUNT) )
	print(f'Total video frames: {length}')
	counter = 0
	saved_counter = 0
	os.makedirs(directory, exist_ok=True)
	while True:
		success, vframe = capture.read()
		if success == False:
			break
		counter += 1
		if NTH_FRAME != -1 and counter % NTH_FRAME != 0:
			continue
		frame = Image.fromarray( cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) ).convert('RGB')
		savepath = os.path.join( directory, f'frame_{counter+1}.jpg' )
		frame.save(savepath)
		saved_counter += 1
		if MAX_FRAMES != -1 and saved_counter < MAX_FRAMES:
			break
	capture.release()

if __name__ == '__main__':
	filepath : str = "C:\\Users\\Declan\\Music\\2024-06-14 22-32-32.mp4"
	directory : str = "C:\\Users\\Declan\\Music\\frames"

	NTH_FRAME = 5
	MAX_FRAMES = -1

	extract_frames_to_directory(filepath, directory, NTH_FRAME=NTH_FRAME, MAX_FRAMES=MAX_FRAMES)
