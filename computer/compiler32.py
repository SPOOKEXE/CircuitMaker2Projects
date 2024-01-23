

'''
MassiveMemory Output:
4 LEFT BITS : OPCODE INSTRUCTION
6 CENTER BITS : VALUE/ADDRESS/NONE
6 RIGHT BITS : VALUE/ADDRESS/NONE
'''


from typing import Callable

OPCODES : dict[str, int] = {
	'RESET' :	'0000',
	'READ'	:	'0001',
	'WRITE'	:	'0010',
	'COPY'	:	'0011',
	'FLUSH'	:	'0100',
	'ECHO'	:	'0101',
	'NULL'	:	'0110',
	'NULL'	:	'0111',
	'ADD'	:	'1000',
	'SUB'	:	'1001',
	'NULL'	:	'1010',
	'NULL'	:	'1011',
	'NULL'	:	'1100',
	'NULL'	:	'1101',
	'NULL'	:	'1110',
	'NULL'	:	'1111',
}

OPCODE_ARGS_LENGTH : dict[str, int] = {
	'READ'	: 1,
	'WRITE'	: 2,
	'FLUSH'	: 0,
	'ECHO'	: 1,
	'ADD'	: 3,
	'SUB'	: 3
}

MACROS : dict[str, Callable[[list[str]], str]] = {
	#'MACRO' : (lambda args : f'{[args[0]]}{args[1]}{args[2]}')
}

MACROS_ARGS_LENGTH : dict[str, int] = {
	# 'MACRO' : 3,
}

MAX_VALUE_BIT_SIZE : int = 6
MAX_ADDRESS_BIT_SIZE : int = 6

MIN_MEMORY_ADDRESS : int = 5
MAX_MEMORY_ADDRESS : int = pow(2, MAX_ADDRESS_BIT_SIZE-1)

class CompilerErrors:
	INSTRUCTION_NOT_VALID = 'Instruction {} on line {} is not a valid instruction!'
	INSTRUCTION_ARGS_LENGTH = 'Instruction {} on line {} has the wrong number of arguments!'
	MEMORY_ADDRESS_INVALID = 'A memory reference was not passed in on line {}!'
	MEMORY_ADDRESS_OUT_OF_RANGE = 'The memory address on line {} is out of range! Got {} when max address is 63 and minimum is 7.'
	NOT_A_NUMBER = 'The passed value on line {} is not a number!'
	NUMBER_OUT_OF_RANGE = 'The passed number on line {} is too large for the bit size!'

def expand_macro( args : list[str] ) -> list[str]:
	macro : str = args.pop(0)
	transform : Callable[[list[str]], str] = MACROS.get( macro )
	if transform == None: raise ValueError(f'Macro "{macro}" does not exist!')
	arglen : int = MACROS_ARGS_LENGTH.get( macro ) or 0
	if len(args) != arglen: raise ValueError(f'Macro "{macro}" requires {arglen} arguments but you have passed {len(args)} arguments!')
	return transform(args)

def preprocess( content : str ) -> list[str]:
	content = content.replace('\t', '') # no tabs needed
	lines : list[str] = []
	for line in content.split('\n'):
		if line.strip() == '': continue # ignore empty lines
		if line.find('>>') != -1: line = line[:line.find('>>')] # remove comments
		line = line.replace('\n', '') # remove newlines
		lines.append( line ) # finished preprocessing line
	return lines

class Validators:

	def NUMBER_VALUE( value : str ) -> int:
		'''
		0 = MEMORY ADDRESS
		1 = NOT A NUMBER
		2 = OVER/UNDER BIT SIZE LIMIT
		3 = BINARY
		4 = DECIMAL
		'''
		if '$' in value:
			return 0
		try:
			if value.find('0b') == -1:
				value = int( value, base=10 )
				if value > pow(2, MAX_VALUE_BIT_SIZE): return 2
				return 4
			else:
				value = int( value, base=2 )
				if value > pow(2, MAX_VALUE_BIT_SIZE): return 2
				return 3
		except Exception as exception:
			print( exception )
			return 1

	def MEMORY_ADDRESS( index : int, value : str, is_write : bool = False ) -> None:
		if not '$' in value:
			raise ValueError( CompilerErrors.MEMORY_ADDRESS_INVALID.format(index) )
		try:
			if str( value[1:] ).find('0b') == -1: address_n : int = int( value[1:], base=10 )
			else: address_n : int = int( value[1:], base=2 )
		except:
			raise ValueError( CompilerErrors.MEMORY_ADDRESS_INVALID.format(index) )
		if is_write and (address_n > MAX_MEMORY_ADDRESS or address_n < MIN_MEMORY_ADDRESS):
			raise ValueError( CompilerErrors.MEMORY_ADDRESS_OUT_OF_RANGE.format(index, address_n) )

	def OPCODE_INSTRUCTION( index : int, line : str ) -> tuple[str, list[str]]:
		splits : list[str] = line.split(' ')
		instruction : str = splits.pop(0)
		opcode : str = OPCODES.get( instruction )
		if opcode == None: raise ValueError( CompilerErrors.INSTRUCTION_NOT_VALID.format(instruction, index) )
		arglen : int = OPCODE_ARGS_LENGTH.get(instruction) or 0
		if len(splits) != arglen: raise ValueError( CompilerErrors.INSTRUCTION_ARGS_LENGTH.format(instruction, index) )
		return instruction, splits

def to_binary_format( value : int, BIT_SIZE : int ) -> str:
	return format(value, f'#0{BIT_SIZE+2}b')[2:]

def compile_instruction( index : int, line : str ) -> list[str]:
	print(line)
	instruction, args = Validators.OPCODE_INSTRUCTION( index, line )
	print( instruction, args )
	if instruction == 'RESET':
		return [ f"{OPCODES['RESET']} 000000 000000" ]
	elif instruction == 'READ':
		Validators.MEMORY_ADDRESS( index, args[0] )
		addr0 : str = to_binary_format( int(args[0][1:]), MAX_ADDRESS_BIT_SIZE )
		return [ f"{OPCODES['READ']} {addr0} 000100" ]
	elif instruction == 'WRITE':
		Validators.MEMORY_ADDRESS( index, args[0] )
		numtype : int = Validators.NUMBER_VALUE( args[1] )
		if numtype == 0 or numtype == 1: # memory address or not-a-number
			raise ValueError( CompilerErrors.NOT_A_NUMBER.format(index) )
		if numtype == 2:
			raise ValueError( CompilerErrors.NUMBER_OUT_OF_RANGE.format(index) )
		address : str = to_binary_format( int(args[0][1:]), MAX_ADDRESS_BIT_SIZE )
		value : int = 0
		if numtype == 3:
			value = to_binary_format( args[1], MAX_VALUE_BIT_SIZE ) # binary
		elif numtype == 4:
			value = to_binary_format( int(args[1]), MAX_VALUE_BIT_SIZE ) # decimal
		return [ f"{ OPCODES['WRITE'] } { address } { value }" ]
	elif instruction == 'COPY':
		Validators.MEMORY_ADDRESS( index, args[0] )
		Validators.MEMORY_ADDRESS( index, args[1] )
		addr0 : str = to_binary_format( int(args[0][1:]), MAX_ADDRESS_BIT_SIZE )
		addr1 : str = to_binary_format( int(args[1][1:]), MAX_ADDRESS_BIT_SIZE )
		return [ f"{OPCODES['COPY']} {addr0} {addr1}" ]
	elif instruction == 'FLUSH':
		return [ f"{OPCODES['FLUSH']} 000000 000000" ]
	elif instruction == 'ECHO':
		addr0 : str = to_binary_format( int(args[0][1:]), MAX_ADDRESS_BIT_SIZE )
		return [ f"{OPCODES['FLUSH']} 000000 000000", f"{OPCODES['ECHO']} {addr0} 000000" ]
	elif instruction == 'ADD':
		Validators.MEMORY_ADDRESS( index, args[0] )
		Validators.MEMORY_ADDRESS( index, args[1] )
		Validators.MEMORY_ADDRESS( index, args[2] )
		addr0 : str = to_binary_format( int(args[0][1:]), MAX_ADDRESS_BIT_SIZE )
		addr1 : str = to_binary_format( int(args[1][1:]), MAX_ADDRESS_BIT_SIZE )
		addr2 : str = to_binary_format( int(args[2][1:]), MAX_ADDRESS_BIT_SIZE )
		return [
			f"{OPCODES['COPY']} {addr0} 000000",
			f"{OPCODES['COPY']} {addr1} 000001",
			f"{OPCODES['ADD']} 000000 000000",
			f"{OPCODES['COPY']} 000010 {addr2}"
		]
	elif instruction == 'SUB':
		Validators.MEMORY_ADDRESS( index, args[0] )
		Validators.MEMORY_ADDRESS( index, args[1] )
		Validators.MEMORY_ADDRESS( index, args[2] )
		addr0 : str = to_binary_format( int(args[0][1:]), MAX_ADDRESS_BIT_SIZE )
		addr1 : str = to_binary_format( int(args[1][1:]), MAX_ADDRESS_BIT_SIZE )
		addr2 : str = to_binary_format( int(args[2][1:]), MAX_ADDRESS_BIT_SIZE )
		return [
			f"{OPCODES['COPY']} {addr0} 000000",
			f"{OPCODES['COPY']} {addr1} 000001",
			f"{OPCODES['SUB']} 000000 000001",
			f"{OPCODES['COPY']} 000010 {addr2}"
		]
	raise ValueError(f'Instruction {instruction} on line {index} is currently not supported in compiling!')

def compile( content : str ) -> str:
	machine : list[str] = []
	lines : list[str] = preprocess( content )
	for index, line in enumerate(lines):
		translated : list[str] = compile_instruction( index, line )
		print( translated )
		machine.extend( translated )
	return '\n'.join(machine)

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

def memory_code( compiled : str, padding : bool = True ) -> str:
	memory : str = ''.join([ MassiveMemory.number_to_massivememory( int(line, base=2) ) for line in compiled.replace(' ', '').split('\n') ])
	if padding:
		memory = MassiveMemory.fill_padding(memory)
	return memory

if __name__ == '__main__':

	# print(Validators.NUMBER_VALUE('$0b0000110'))
	# print(Validators.NUMBER_VALUE('0b0000110'))
	# print(Validators.NUMBER_VALUE('6'))

	assembly = compile('''
	RESET			>> reset the memory/registers
	WRITE $8 4		>> write 4 to ADDRESS 8
	WRITE $9 4		>> write 4 to ADDRESS 9
	ADD $8 $9 $10	>> add ADDR-8 and ADDR-9 and write to ADDR-10
	ECHO $10		>> outputs result (8)
	''')
	print(assembly)

	memory = memory_code( assembly, padding=False )
	print(memory[:100])

	# == BINARY CODE ==
	# 0000 000000 000000
	# 0010 001000 000100
	# 0010 001001 000100
	# 0011 001000 000000
	# 0011 001001 000001
	# 1000 000000 000000
	# 0011 000010 001010
	# 0100 000000 000000
	# 0101 001010 000000

	# == MASSIVE MEMORY ENCODED ==
	# EICEJCAIDBJDAAIKCDAAEAKF

	with open( 'computer/assembly_memory.txt', 'w' ) as file:
		file.write( memory )
