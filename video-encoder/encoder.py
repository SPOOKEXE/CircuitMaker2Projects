

'''
Utilizing '!build massivememory'

Iterate top left to bottom right, row by row.
Value is 0 or 1 which represents ON/OFF.
'''

'''
TODO:
fix the bugged math so ik how much room there is remaining
'''

import cv2
import numpy as np

from dataclasses import dataclass
from PIL import Image

class BaseProcessor:
	def preprocess( self, image : Image.Image ) -> Image.Image:
		return image

class GrayscalePixelArtProcessor(BaseProcessor):
	def __init__(self, size : int = 256 ):
		self.SIZE = size
	def preprocess( self, image : Image.Image ) -> list[int]:
		image = image.convert('L')
		image.thumbnail( (self.SIZE, self.SIZE), Image.BILINEAR )
		return [ v > 127 and 1 or 0 for v in np.array( image ).flatten() ]

def extract_frames_from_video( filepath : str, max_frames : int = -1, processor : BaseProcessor = None ) -> list:
	frames : list[Image.Image] = []
	capture = cv2.VideoCapture( filepath )
	while True:
		success, vframe = capture.read()
		if success == False: break
		frame : Image.Image = Image.fromarray( cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) )
		if processor != None: frame = processor.preprocess( frame )
		frames.append(frame)
		if max_frames != -1 and len(frames) >= max_frames: break
	capture.release()
	return frames

@dataclass
class EncoderConfig:
	GRID_SIZE : int
	MAX_FRAMES : int
	NTH_FRAME : int # every nth frame
	MAX_CHARACTERS_PER_MEMORY : int # 2^n addresses * 2^n bits

def encode_to_raw( video_filepath : str, config : EncoderConfig ) -> str:
	processor : BaseProcessor = GrayscalePixelArtProcessor( size=config.GRID_SIZE )
	print('Processing:', video_filepath)
	frames : list[list[int]] = extract_frames_from_video( video_filepath, max_frames=config.MAX_FRAMES, processor=processor )
	frames : list[list[int]] = frames[ 0::config.NTH_FRAME ]
	print('Gathered frames.')
	return str( np.array(frames).flatten().tolist() ).replace(' ', '').replace(',', '')[1:-1]

BS4 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
def number_to_massivememory( number ) -> str:
	d3 = number & 0b111111
	d2 = (number & 0b111111_000000) >> 6
	d1 = (number & 0b111_000000_000000) >> 12
	return BS4[d3] + BS4[d2] + BS4[d1]

def encode_to_massivememory( video_filepath : str, config : EncoderConfig ) -> str:
	processor : BaseProcessor = GrayscalePixelArtProcessor( size=config.GRID_SIZE )
	print('Processing:', video_filepath)
	frames : list[list[int]] = extract_frames_from_video( video_filepath, max_frames=config.MAX_FRAMES, processor=processor )
	frames : list[list[int]] = frames[ 0::config.NTH_FRAME ]
	print('Gathered frames.')

	print('Encoding')
	WHITE : str = number_to_massivememory( int('1111111111111111', 2) )
	BLACK : str = number_to_massivememory( int('0000000000000000', 2) )

	# frames : list[list] = [ [ v==0 and BLACK or WHITE for v in np.array( frame ).flatten().tolist() ] for frame in frames ]
	frames : list[int] = [ v==0 and BLACK or WHITE for v in np.array( frames ).flatten().tolist() ]
	return "".join([str(v) for v in frames])

if __name__ == '__main__':

	config = EncoderConfig(
		GRID_SIZE = 16,
		MAX_FRAMES = 900,
		NTH_FRAME = 3,
		MAX_CHARACTERS_PER_MEMORY = 200000, #pow(2, 12) * pow(2, 16) # 16 addresses * 16 bits
	)

	# data : str = encode_to_raw( 'video-encoder/Strobe_WhiteAndBlack_FastFlash.mp4', config )
	# with open(f'video-encoder/0_raw.txt', 'w') as file:
	# 	file.write( data )

	data : str = encode_to_massivememory( 'video-encoder/Strobe_WhiteAndBlack_FastFlash.mp4', config )
	with open(f'video-encoder/0_memory.txt', 'w') as file:
		file.write( data )
