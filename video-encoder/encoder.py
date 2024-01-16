

'''
Utilizing '!build massivememory'

Iterate top left to bottom right, row by row.
Value is 0 or 1 which represents ON/OFF.
'''

from dataclasses import dataclass
import cv2
import numpy as np

from math import ceil, floor
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

def encode_to_membits( video_filepath : str, config : EncoderConfig ) -> list[list[int]]:
	processor : BaseProcessor = GrayscalePixelArtProcessor( size=config.GRID_SIZE )

	print('Processing:', video_filepath)
	frames : list[list[int]] = extract_frames_from_video( video_filepath, max_frames=config.MAX_FRAMES, processor=processor )
	frames : list[list[int]] = frames[ 0::config.NTH_FRAME ]

	print('Writing Memory Data')
	BITS_PER_FRAME : int = config.GRID_SIZE * config.GRID_SIZE
	FRAMES_PER_MEMORY_BLOCKS : int = floor(config.MAX_CHARACTERS_PER_MEMORY / BITS_PER_FRAME)
	print(BITS_PER_FRAME, config.MAX_CHARACTERS_PER_MEMORY, FRAMES_PER_MEMORY_BLOCKS )

	iterations : int = ceil( len(frames) / FRAMES_PER_MEMORY_BLOCKS )
	return [ frames[index*FRAMES_PER_MEMORY_BLOCKS:index+1*FRAMES_PER_MEMORY_BLOCKS] for index in range(iterations) ]

if __name__ == '__main__':

	config = EncoderConfig(
		GRID_SIZE = 16,
		MAX_FRAMES = 800,
		NTH_FRAME = 3,
		MAX_CHARACTERS_PER_MEMORY = 200000, #pow(2, 12) * pow(2, 16) # 16 addresses * 16 bits
	)

	memory_blocks : str = encode_to_membits( 'video-encoder/BadApple-360p.mp4', config )
	for index, block in enumerate(memory_blocks):
		data : str = str(block).replace(' ', '').replace(',', '').replace('[', '').replace(']', '')
		print(index, len(data))
		with open(f'video-encoder/{index}_memory.txt', 'w') as file:
			file.write( data )
