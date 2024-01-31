
from __future__ import annotations

import numpy as np
import cv2

from dataclasses import dataclass, field
from uuid import uuid4
from PIL import Image
from pyxelate import Pyx, Pal

class BlockEnum:
	Tile = 14

# 14,0,0,0,0,255+0+0???
@dataclass
class Block:
	uid : str = field( default_factory=lambda : uuid4().hex) # only used in python to make connections easier
	blockid : int = -1

	x : int = 0
	y : int = 0
	z : int = 0

	properties : list = field( default_factory=list )
	active : bool = False

	inputs : list[str] = field( default_factory=list )
	outputs : list[str] = field( default_factory=list )

# class ConnectionTools:

# 	@staticmethod
# 	def connect( _ : SaveFile, block0 : Block, block1 : Block ) -> None:
# 		if not (block1.uid in block0.outputs):
# 			block0.outputs.append( block1.uid )
# 			block1.inputs.append( block0.uid )

# 	@staticmethod
# 	def clear_inputs( save : SaveFile, block0 : Block ) -> None:
# 		# scan all input unique ids and remove the corrosponding output from that target block
# 		for uid in block0.inputs:
# 			block = save.get_block_by_uid( uid )
# 			if block == None:
# 				continue
# 			if block0.uid in block.outputs:
# 				block.outputs.remove( block0.uid )
# 		block0.inputs = []

# 	@staticmethod
# 	def clear_outputs( save : SaveFile, block0 : Block ) -> None:
# 		# scan all output unique ids and remove the corrosponding input from that target block
# 		for uid in block0.outputs:
# 			block = save.get_block_by_uid( uid )
# 			if block == None:
# 				continue
# 			if block0.uid in block.inputs:
# 				block.inputs.remove( block0.uid )
# 		# clear this block's outputs
# 		block0.outputs = []

# 	@staticmethod
# 	def clear_connections( save : SaveFile, block0 : Block ) -> None:
# 		ConnectionTools.clear_inputs(save, block0)
# 		ConnectionTools.clear_outputs(save, block0)

class SaveFile:
	blocks : list[Block]
	uids : list[str]
	mapped : dict[str, Block]

	def __init__( self ) -> SaveFile:
		self.blocks = list()
		self.uids = list()
		self.mapped = dict()

	def append_blocks( self, blocks : list[Block] ) -> None:
		for block in blocks:
			if block.uid in self.uids:
				continue
			self.blocks.append( block )
			self.uids.append( block.uid )
			self.mapped[block.uid] = block

	def remove_blocks_by_uid( self, unique_ids : list[str] ) -> None:
		for uid in unique_ids:
			if not (uid in self.uids):
				continue
			block = self.mapped.get(uid)
			self.blocks.remove(block)
			self.uids.remove(uid)
			self.mapped.pop(uid)

	def remove_blocks( self, blocks : list[Block] ) -> None:
		self.remove_blocks_by_uid([ block.uid for block in blocks ])

	def get_block_by_uid( self, uid : str ) -> Block | None:
		if not uid in self.uids:
			return None
		return self.mapped.get( uid )

	def get_blocks_by_uids( self, uids : list[str] ) -> list[Block]:
		blocks : list[Block] = list()
		for uid in uids:
			if not (uid in self.uids):
				continue
			blocks.append( self.mapped.get(uid) )
		return blocks

	def to_save_format( self ) -> str:
		invidual_blocks : list[str] = []
		for block in self.blocks:
			props : str = "+".join([str(item) for item in block.properties])
			invidual_blocks.append(f'{block.blockid},{block.active and 1 or 0},{block.x},{block.y},{block.z},{ props }')
		return ";".join( invidual_blocks ) + "???"

	# def load_save_format( self, save : str ) -> None:
	# 	raise NotImplementedError

class ImageEditor:

	@staticmethod
	def to_cv2( img : Image.Image ) -> np.ndarray:
		return cv2.cvtColor( np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR )

	@staticmethod
	def to_PIL( img : np.ndarray, mode : str = 'RGB' ) -> Image.Image:
		return Image.fromarray( cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode=mode )

class SaveGenerators:

	@staticmethod
	def convert_image_to_savefile( image : Image.Image ) -> str:
		image = image.convert('RGB')
		image.save('pixelart/test.jpg')

		pixels = np.array(image)
		print( len(pixels), len(pixels[0]) )

		pixel_blocks : list[Block] = []
		for y in range(len(pixels)):
			for x in range(len(pixels[0])):
				block = Block(blockid=BlockEnum.Tile, x=x, y=len(pixels)-y-1, z=0, properties=pixels[y][x][:3])
				pixel_blocks.append( block )

		savefile = SaveFile()
		savefile.append_blocks( pixel_blocks )
		return savefile.to_save_format()

	@staticmethod
	def create_pixel_art_simple( image : Image.Image, grid_size : int = 64, ) -> str:
		image.thumbnail((grid_size, grid_size), Image.BILINEAR)
		return SaveGenerators.convert_image_to_savefile( image )

	@staticmethod
	def create_pixel_art_advanced(
		image : Image.Image,
		factor : int = 8,
		pallete : int | list = 256,
		dither : str = 'none',
		max_dims : tuple[int, int] = (256, 256)
	) -> str:
		image.thumbnail(max_dims, Image.BILINEAR)
		pixels = ImageEditor.to_cv2( image )
		pixels = Pyx(factor=factor, palette=pallete, dither=dither).fit_transform(pixels)
		return SaveGenerators.convert_image_to_savefile( Image.fromarray(pixels) )

source : Image.Image = Image.open('pixelart/31tkCSZqN3L_2.png')
save : str = SaveGenerators.create_pixel_art_simple( source, grid_size=32 )
# save : str = SaveGenerators.create_pixel_art_save( source, factor=8, pallete=Pal.GAMEBOY_ORIGINAL, dither='none', max_dims=(512, 512) )
print(len(save))

with open('pixelart/pixel_art_save.txt', 'w') as file:
	file.write(save)
