
import traceback
import cv2
import numpy as np
import os

from typing import Any
from PIL import Image

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
		image.thumbnail( (grid, grid), Image.BILINEAR )
		center : tuple[int, int] = (centered == True) and ImageEditor.get_center_xy( image.size, grid ) or (0, 0)
		img : Image.Image = Image.new('L', (grid, grid), bgc)
		img.paste( image, center )
		return img

class VideoEditor:

	@staticmethod
	def extract_frames_from_video( filepath : str, MAX_FRAMES : int = -1 ) -> list[Image.Image]:
		frames : list[Image.Image] = []
		capture = cv2.VideoCapture( filepath )
		while True:
			success, vframe = capture.read()
			if success == False: break
			frame : Image.Image = Image.fromarray( cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) )
			frames.append(frame)
			if MAX_FRAMES != -1 and len(frames) >= MAX_FRAMES: break
		capture.release()
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
				if debug: cv2.imwrite( f'{ os.path.split( filepath )[0] }/frame_{index}.jpg', b4 )
				video.write( b4 )
			video.release()
			return True
		except Exception as exception:
			print('Failed to write frames to video:')
			traceback.print_exception( exception )
			return False
