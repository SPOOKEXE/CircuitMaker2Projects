
# https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c

'''
requirements:
- mido
- pydantic
'''

from enum import Enum
from pydantic import BaseModel
from midoparse import midi2array
from math import floor

import mido

def map_value( value : float | int, u0 : float | int, v0 : float | int, u1 : float | int, v1 : float | int ) -> float | int:
	return (value - u0) / (v0 - u0) * (v1 - u1) + u1

MIDI_NUMBER_TO_FREQUENCY = [
	27.5,
	29.135,
	30.868,
	32.703,
	34.648,
	36.708,
	38.891,
	41.203,
	43.654,
	46.249,
	48.999,
	51.913,
	55.0,
	58.27,
	61.735,
	65.406,
	69.296,
	73.416,
	77.782,
	82.407,
	87.307,
	92.499,
	97.999,
	103.826,
	110.0,
	116.541,
	123.471,
	130.813,
	138.591,
	146.832,
	155.563,
	164.814,
	174.614,
	184.997,
	195.998,
	207.652,
	220.0,
	233.082,
	246.942,
	261.626,
	277.183,
	293.665,
	311.127,
	329.628,
	349.228,
	369.994,
	391.995,
	415.305,
	440.0,
	466.164,
	493.883,
	523.251,
	554.365,
	587.33,
	622.254,
	659.255,
	698.456,
	739.989,
	783.991,
	830.609,
	880.0,
	932.328,
	987.767,
	1046.502,
	1108.731,
	1174.659,
	1244.508,
	1318.51,
	1396.913,
	1479.978,
	1567.982,
	1661.219,
	1760.0,
	1864.655,
	1975.533,
	2093.005,
	2217.461,
	2349.318,
	2489.016,
	2637.02,
	2793.826,
	2959.955,
	3135.963,
	3322.438,
	3520.0,
	3729.31,
	3951.066,
	4186.009
]

class CM2Blocks:
	SOUND = 7
	DELAY = 16
	BUTTON = 4

class Instrument(Enum):
	SINE = 0
	SQUARE = 1
	TRIANGLE = 2
	SAWTOOTH = 3

class Component(BaseModel):
	pass

class Delay(Component):
	delay : int = 20 # milliseconds

class Sound(Component):
	frequency : float
	instrument : Instrument
	period : int # milliseconds

class Button(Component):
	pass

class MidiTransform:

	@staticmethod
	def midi_to_components(filepath : str) -> list[list[Component] | Delay]:
		mid = mido.MidiFile(filepath, clip=True)
		array = midi2array(mid)
		tracks = [ Button(), ]
		for sequence in array:
			# delay : int = 2
			notes = []
			for note_number, period in enumerate(sequence):
				if period == 0: continue # note doesnt play
				sound = Sound(frequency=MIDI_NUMBER_TO_FREQUENCY[note_number], instrument=Instrument.SAWTOOTH.value, period=period)
				# ticks = round(map_value(period, 0, 1000, 0, 20))
				# delay = max(ticks, delay)
				notes.append(sound)
			tracks.append(notes)
			# tracks.append(Delay(delay=delay))
		return tracks

	@staticmethod
	def components_to_cm2(components : list[Delay | list[Sound] | Button]) -> str:

		connections : list[tuple[int, int]] = []
		component_counter : int = 0

		for index, component in enumerate(components):
			if index == 0: continue # first item has no previous connecitons
			if isinstance(component, list):
				# connect all to previous
				for subindex, _ in enumerate(component):
					connections.append((component_counter, component_counter + 1 + subindex))
				component_counter += len(component)
			else:
				# connect singular to previous
				connections.append((component_counter, component_counter+1))
				component_counter += 1

		save_data : str = ""
		for index, item in enumerate(components):
			z = index % 30
			x = floor(index / 30) * -2
			if isinstance(item, list) is True:
				# list of sound
				for sound_index, sound in enumerate(item):
					save_data = save_data + f'{CM2Blocks.SOUND},0,{x},{sound_index},{z},{sound.frequency}+{sound.instrument.value};'
			elif isinstance(item, Delay) is True:
				# delay
				save_data = save_data + f'{CM2Blocks.DELAY},0,{x},{0},{z},{item.delay};'
			elif isinstance(item, Button) is True:
				save_data = save_data + f'{CM2Blocks.BUTTON},0,{x},{0},{z},;'

		save_data = save_data[:-1] + "?"
		for (a, b) in connections:
			save_data = save_data + f"{a+1},{b+1};"

		save_data = save_data[:-1] + "??"
		return save_data

if __name__ == '__main__':
	components = MidiTransform.midi_to_components('midi/Hall of the Mountain King.mid')
	components = [components[0]] + components[1000:2000]

	output = ""
	for item in components:
		if isinstance(item, list):
			output += str([v.frequency for v in item])
		else:
			output += item.model_dump_json()
		output += "\n"

	with open('components.txt', 'w') as file:
		file.write(output)

	save : str = MidiTransform.components_to_cm2(components)
	with open('save.cm2', 'w') as file:
		file.write(save)
