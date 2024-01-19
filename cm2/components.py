
class MassMemory:
	'''8-bit memory.'''
	pass

class MassiveMemory:
	'''16-bit memory.'''

	BASE64 : str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
	MAX_CHARACTERS_PER_MEMORY : int = 200000

	@staticmethod
	def number_to_massivememory( number : int ) -> str:
		'''Encode the given number to the 16bit custom base64.'''
		d3 : int = (number & 0b111111)
		d2 : int = (number & 0b111111_000000) >> 6
		d1 : int = (number & 0b111111_000000_000000) >> 12
		return MassiveMemory.BASE64[d3] + MassiveMemory.BASE64[d2] + MassiveMemory.BASE64[d1]

	@staticmethod
	def fill_padding( memory : str ) -> str:
		length : int = MassiveMemory.MAX_CHARACTERS_PER_MEMORY - len(memory)
		return memory + ( MassiveMemory.BASE64[0] * length )
