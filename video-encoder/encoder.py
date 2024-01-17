

'''
Utilizing '!build massivememory'

Iterate top left to bottom right, row by row.
Value is 0 or 1 which represents ON/OFF.
'''

'''
TODO:
fix the bugged math so ik how much room there is remaining
'''

import json
import cv2
import numpy as np

from dataclasses import dataclass
from PIL import Image
from regex import P

class BaseProcessor:
	def preprocess( self, image : Image.Image ) -> Image.Image:
		return image

class GrayscalePixelArtProcessor(BaseProcessor):
	def __init__(self, size : int = 256 ):
		self.SIZE = size
	def preprocess( self, image : Image.Image ) -> list[int]:
		image = image.convert('L')
		image.thumbnail( (self.SIZE, self.SIZE), Image.BILINEAR )
		nimage = Image.new('L', (self.SIZE, self.SIZE), 0)
		nimage.paste( image, (0, 0) )
		return [ v > 127 and 1 or 0 for v in np.array( nimage ).flatten() ]

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

	print('Encoding frames.')
	mm : list = []
	for frame in frames:
		new_row : list = [ ]
		for index in frame[0::config.GRID_SIZE]:
			value = frame[index*config.GRID_SIZE:(index+1)*config.GRID_SIZE]
			num = int("".join([ str(v) for v in value ]), 2)
			encoded = number_to_massivememory( num )
			new_row.append( encoded )
		# new_row.reverse()
		mm.extend(new_row)
	value : str = "".join(mm)
	padd_amnt : int = config.MAX_CHARACTERS_PER_MEMORY - len(value)
	print(f'Padding an additional {padd_amnt} values.')
	return value + ("A" * padd_amnt)

def split_to_chunks( array : list, length : int ) -> list:
	return [ array[i:i+length] for i in range(0, len(array), length) ]

HEX_MAPPED = {
	'0000' : '0', '0001' : '1',
	'0010' : '2', '0011' : '3',
	'0100' : '4', '0101' : '5',
	'0110' : '6', '0111' : '7',
	'1000' : '8', '1001' : '9',
	'1010' : 'A', '1011' : 'B',
	'1100' : 'C', '1101' : 'D',
	'1110' : 'E', '1111' : 'F'
}

def strhex( value : str ) -> str:
	return HEX_MAPPED.get(value)

def encode_to_massmemory( video_filepath : str, config : EncoderConfig ) -> str:
	processor : BaseProcessor = GrayscalePixelArtProcessor( size=config.GRID_SIZE )
	print('Processing:', video_filepath)
	frames : list[list[int]] = extract_frames_from_video( video_filepath, max_frames=config.MAX_FRAMES, processor=processor )
	frames : list[list[int]] = frames[ 0::config.NTH_FRAME ]

	encoded_frames : list = [ ]
	for index, row in enumerate(frames):
		print(index, 'FRAME NUMBER')
		print(row)
		pixels : str = ''.join([str(x) for x in row])
		print(pixels)
		splits : list[str] = split_to_chunks( pixels, 4 )
		print(splits)
		encoded : list[str] = [ strhex( chunk ) for chunk in splits ]
		print(encoded)
		encoded_frames.append( ''.join(encoded) )

	with open('video-encoder/frames.txt', 'w') as file:
		file.write( json.dumps( encoded_frames, indent=4 ) )

	value : str = "".join([ "".join(a) for a in encoded_frames ])
	padd_amnt : int = config.MAX_CHARACTERS_PER_MEMORY - len(value)
	print(f'Padding an additional {padd_amnt} values.')
	return value + ('0' * padd_amnt)

def default_massivememory( FILEPATH : str ) -> None:

	config = EncoderConfig(
		GRID_SIZE = 16,
		MAX_FRAMES = 900,
		NTH_FRAME = 3,
		MAX_CHARACTERS_PER_MEMORY = 200000, #pow(2, 12) * pow(2, 16) # 16 addresses * 16 bits
	)

	data : str = encode_to_massivememory( FILEPATH, config )
	print('Data is encoded: ', len(data))
	with open(f'video-encoder/0_memory.txt', 'w') as file:
		file.write( data )

def default_massmemory( FILEPATH : str ) -> None:
	config = EncoderConfig(
		GRID_SIZE = 8,
		MAX_FRAMES = 1200,
		NTH_FRAME = 3,
		MAX_CHARACTERS_PER_MEMORY = 200000,
	)

	data : str = encode_to_massmemory( FILEPATH, config )
	print('Data is encoded: ', len(data))
	with open(f'video-encoder/1_memory.txt', 'w') as file:
		file.write( data )

if __name__ == '__main__':

	# FILEPATH : str = 'video-encoder/flashing_1.mp4'
	FILEPATH : str = 'video-encoder/badapple_compact.mp4'

	# MASSIVE MEMORY
	# default_massivememory( FILEPATH )

	# MASS MEMORY
	default_massmemory( FILEPATH )

	# with open('video-encoder/white.txt', 'w') as file:
	# 	file.write( 'FFFFFFFFFFFFFFFF0000000000000000' * 62 )
