import numpy as np

from enum import Enum

class BasePalette(Enum):
	"""Palette Enum class with additional helper functions"""

	def __len__(self):
		"""Number of colors in palette"""
		return len(self.value)

	def __iter__(self):
		self.n = 0
		return self

	def __next__(self):
		if self.n < len(self):
			result = self.value[self.n]
			self.n += 1
			return result
		else:
			raise StopIteration

	@classmethod
	def list(self):
		"""list all available Palette names"""
		return list(map(lambda x: x.name, self))

	def from_hex(hex_list):
		"""Generate Pal palette from list of #HEX color values"""
		hex_list = [h.lstrip("#") for h in hex_list]
		return np.array([[tuple(int(h[i:i+2], 16) for i in (0, 2, 4))] for h in hex_list], dtype=float) / 255.0

	def from_rgb(rgb_list):
		"""Generate Pal palette from list of 0-255 [R, G, B] values"""
		return np.array([[rgb] for rgb in rgb_list], dtype=float) / 255.


class Pal(BasePalette):
	"""Enum of common palettes based on https://en.wikipedia.org/wiki/List_of_color_palettes"""

	TELETEXT = [
		[[0.        , 0.        , 0.        ]],
		[[1.        , 0.        , 0.        ]],
		[[0.        , 0.50196078, 0.        ]],
		[[1.        , 1.        , 0.        ]],
		[[0.        , 0.        , 1.        ]],
		[[1.        , 0.        , 1.        ]],
		[[0.        , 1.        , 1.        ]],
		[[1.        , 1.        , 1.        ]]
	]

	BBC_MICRO = TELETEXT

	CGA_MODE4_PAL1 = [
		[[0., 0., 0.]],
		[[1., 1., 1.]],
		[[0., 1., 1.]],
		[[1., 0., 1.]]
	]

	CGA_MODE5_PAL1 = [
		[[0.        , 0.        , 0.        ]],
		[[0.33333333, 1.        , 1.        ]],
		[[1.        , 0.33333333, 0.33333333]],
		[[1.        , 1.        , 1.        ]]
	]

	CGA_MODE4_PAL2 = [
		[[0., 0., 0.]],
		[[0.33333333, 1.        , 0.33333333]],
		[[1.        , 0.33333333, 0.33333333]],
		[[0.33333333, 1.        , 0.33333333]]
	]

	ZX_SPECTRUM = [
		[[0.        , 0.        , 0.        ]],
		[[0.        , 0.15294118, 0.98431373]],
		[[1.        , 0.18823529, 0.08627451]],
		[[1.        , 0.24705882, 0.98823529]],
		[[0.        , 0.97647059, 0.17254902]],
		[[0.        , 0.98823529, 0.99607843]],
		[[1.        , 0.99215686, 0.2       ]],
		[[1.        , 1.        , 1.        ]]
	]

	APPLE_II_LO = [
		[[0.        , 0.        , 0.        ]],
		[[0.52156863, 0.23137255, 0.31764706]],
		[[0.31372549, 0.27843137, 0.5372549 ]],
		[[0.91764706, 0.36470588, 0.94117647]],
		[[0.        , 0.40784314, 0.32156863]],
		[[0.57254902, 0.57254902, 0.57254902]],
		[[0.        , 0.65882353, 0.94509804]],
		[[0.79215686, 0.76470588, 0.97254902]],
		[[0.31764706, 0.36078431, 0.05882353]],
		[[0.92156863, 0.49803922, 0.1372549 ]],
		[[0.57254902, 0.57254902, 0.57254902]],
		[[0.96470588, 0.7254902 , 0.79215686]],
		[[0.        , 0.79215686, 0.16078431]],
		[[0.79607843, 0.82745098, 0.60784314]],
		[[0.60392157, 0.8627451 , 0.79607843]],
		[[1.        , 1.        , 1.        ]]
	]

	APPLE_II_HI = [
		[[0.        , 0.        , 0.        ]],
		[[1.        , 0.        , 1.        ]],
		[[0.        , 1.        , 0.        ]],
		[[1.        , 1.        , 1.        ]],
		[[0.        , 0.68627451, 1.        ]],
		[[1.        , 0.31372549, 0.        ]]
	]

	COMMODORE_64 = [
		[[0.        , 0.        , 0.        ]],
		[[1.        , 1.        , 1.        ]],
		[[0.63137255, 0.30196078, 0.2627451 ]],
		[[0.41568627, 0.75686275, 0.78431373]],
		[[0.63529412, 0.34117647, 0.64705882]],
		[[0.36078431, 0.67843137, 0.37254902]],
		[[0.30980392, 0.26666667, 0.61176471]],
		[[0.79607843, 0.83921569, 0.5372549 ]],
		[[0.63921569, 0.40784314, 0.22745098]],
		[[0.43137255, 0.32941176, 0.04313725]],
		[[0.8       , 0.49803922, 0.4627451 ]],
		[[0.38823529, 0.38823529, 0.38823529]],
		[[0.54509804, 0.54509804, 0.54509804]],
		[[0.60784314, 0.89019608, 0.61568627]],
		[[0.54117647, 0.49803922, 0.80392157]],
		[[0.68627451, 0.68627451, 0.68627451]]
	]

	GAMEBOY_COMBO_UP = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.51764706, 0.25882353, 0.01568627]],
		[[0.9254902 , 0.60392157, 0.32941176]],
		[[0.98823529, 0.98039216, 0.98823529]]
	]
	GAMEBOY_COMBO_DOWN = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.61176471, 0.57254902, 0.95686275]],
		[[0.9254902 , 0.54117647, 0.54901961]],
		[[0.98823529, 0.98039216, 0.6745098 ]]
	]

	GAMEBOY_COMBO_LEFT = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.01568627, 0.19607843, 0.98823529]],
		[[0.48627451, 0.66666667, 0.98823529]],
		[[0.98823529, 0.98039216, 0.98823529]],
		[[0.6745098 , 0.14901961, 0.14117647]],
		[[0.9254902 , 0.54117647, 0.54901961]],
		[[0.29803922, 0.54117647, 0.01568627]],
		[[0.01568627, 0.98039216, 0.01568627]]
	]

	GAMEBOY_COMBO_RIGHT = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.98823529, 0.19607843, 0.01568627]],
		[[0.01568627, 0.98039216, 0.01568627]],
		[[0.98823529, 0.98039216, 0.98823529]]
	]

	GAMEBOY_A_UP = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.6745098 , 0.14901961, 0.14117647]],
		[[0.9254902 , 0.54117647, 0.54901961]],
		[[0.98823529, 0.98039216, 0.98823529]],
		[[0.29803922, 0.54117647, 0.01568627]],
		[[0.01568627, 0.98039216, 0.01568627]],
		[[0.01568627, 0.19607843, 0.98823529]],
		[[0.48627451, 0.66666667, 0.98823529]]
	]

	GAMEBOY_A_DOWN = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.98823529, 0.19607843, 0.01568627]],
		[[0.95686275, 0.99607843, 0.01568627]],
		[[0.98823529, 0.98039216, 0.98823529]]
	]

	GAMEBOY_A_LEFT = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.04313725, 0.02745098, 0.08235294]],
		[[0.55686275, 0.52156863, 0.87058824]],
		[[0.98823529, 0.98039216, 0.98823529]],
		[[0.6745098 , 0.14901961, 0.14117647]],
		[[0.9254902 , 0.54117647, 0.54901961]],
		[[0.51764706, 0.25882353, 0.01568627]],
		[[0.9254902 , 0.60392157, 0.32941176]]
	]

	GAMEBOY_A_RIGHT = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.01568627, 0.19607843, 0.98823529]],
		[[0.01568627, 0.98039216, 0.01568627]],
		[[0.98823529, 0.98039216, 0.98823529]],
		[[0.6745098 , 0.14901961, 0.14117647]],
		[[0.9254902 , 0.54117647, 0.54901961]]
	]

	GAMEBOY_B_UP = [
		[[0.29803922, 0.16470588, 0.01568627]],
		[[0.58039216, 0.47843137, 0.29803922]],
		[[0.76862745, 0.68235294, 0.58039216]],
		[[0.98823529, 0.91764706, 0.89411765]],
		[[0.        , 0.        , 0.        ]],
		[[0.51764706, 0.25882353, 0.01568627]],
		[[0.9254902 , 0.60392157, 0.32941176]]
	]

	GAMEBOY_B_DOWN = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.51764706, 0.25882353, 0.01568627]],
		[[0.95686275, 0.99607843, 0.01568627]],
		[[0.98823529, 0.98039216, 0.98823529]],
		[[0.01568627, 0.19607843, 0.98823529]],
		[[0.48627451, 0.66666667, 0.98823529]],
		[[0.29803922, 0.54117647, 0.01568627]],
		[[0.01568627, 0.98039216, 0.01568627]]
	]

	GAMEBOY_B_LEFT = [
		[[0.01568627, 0.00784314, 0.01568627]],
		[[0.45490196, 0.44705882, 0.45490196]],
		[[0.7372549 , 0.72941176, 0.7372549 ]],
		[[0.98823529, 0.98039216, 0.98823529]]
	]

	GAMEBOY_B_RIGHT = [
		[[0.98823529, 0.98039216, 0.98823529]],
		[[0.95686275, 0.99607843, 0.01568627]],
		[[0.01568627, 0.63529412, 0.64313725]],
		[[0.01568627, 0.00784314, 0.01568627]]
	]

	GAMEBOY_ORIGINAL = [
		[[0.        , 0.24705882, 0.        ]],
		[[0.18039216, 0.45098039, 0.1254902 ]],
		[[0.54901961, 0.74901961, 0.03921569]],
		[[0.62745098, 0.81176471, 0.03921569]]
	]

	GAMEBOY_POCKET = [
		[[0.        , 0.        , 0.        ]],
		[[0.33333333, 0.33333333, 0.33333333]],
		[[0.66666667, 0.66666667, 0.66666667]],
		[[1.        , 1.        , 1.        ]]
	]

	GAMEBOY_VIRTUALBOY = [
		[[0.9372549 , 0.        , 0.        ]],
		[[0.64313725, 0.        , 0.        ]],
		[[0.33333333, 0.        , 0.        ]],
		[[0.        , 0.        , 0.        ]]
	]

	MICROSOFT_WINDOWS_16 = [
		[[0.        , 0.        , 0.        ]],
		[[0.50196078, 0.        , 0.        ]],
		[[0.        , 0.50196078, 0.        ]],
		[[0.50196078, 0.50196078, 0.        ]],
		[[0.        , 0.        , 0.50196078]],
		[[0.50196078, 0.        , 0.50196078]],
		[[0.        , 0.50196078, 0.50196078]],
		[[0.75294118, 0.75294118, 0.75294118]],
		[[0.50196078, 0.50196078, 0.50196078]],
		[[1.        , 0.        , 0.        ]],
		[[0.        , 1.        , 0.        ]],
		[[1.        , 1.        , 0.        ]],
		[[0.        , 0.        , 1.        ]],
		[[1.        , 0.        , 1.        ]],
		[[0.        , 1.        , 1.        ]],
		[[1.        , 1.        , 1.        ]]
	]

	MICROSOFT_WINDOWS_20 = [
		[[0.        , 0.        , 0.        ]],
		[[0.50196078, 0.        , 0.        ]],
		[[0.        , 0.50196078, 0.        ]],
		[[0.50196078, 0.50196078, 0.        ]],
		[[0.        , 0.        , 0.50196078]],
		[[0.50196078, 0.        , 0.50196078]],
		[[0.        , 0.50196078, 0.50196078]],
		[[0.75294118, 0.75294118, 0.75294118]],
		[[0.75294118, 0.8627451 , 0.75294118]],
		[[0.65098039, 0.79215686, 0.94117647]],
		[[1.        , 0.98431373, 0.94117647]],
		[[0.62745098, 0.62745098, 0.64313725]],
		[[0.50196078, 0.50196078, 0.50196078]],
		[[1.        , 0.        , 0.        ]],
		[[0.        , 1.        , 0.        ]],
		[[1.        , 1.        , 0.        ]],
		[[0.        , 0.        , 1.        ]],
		[[1.        , 0.        , 1.        ]],
		[[0.        , 1.        , 1.        ]],
		[[1.        , 1.        , 1.        ]]
	]

	MICROSOFT_WINDOWS_PAINT = [
		[[0.        , 0.        , 0.        ]],
		[[1.        , 1.        , 1.        ]],
		[[0.48235294, 0.48235294, 0.48235294]],
		[[0.74117647, 0.74117647, 0.74117647]],
		[[0.48235294, 0.04705882, 0.00784314]],
		[[1.        , 0.14509804, 0.        ]],
		[[0.48235294, 0.48235294, 0.00392157]],
		[[1.        , 0.98431373, 0.00392157]],
		[[0.        , 0.48235294, 0.00784314]],
		[[0.00784314, 0.97647059, 0.00392157]],
		[[0.        , 0.48235294, 0.47843137]],
		[[0.00784314, 0.99215686, 0.99607843]],
		[[0.00392157, 0.0745098 , 0.47843137]],
		[[0.01568627, 0.19607843, 1.        ]],
		[[0.48235294, 0.09803922, 0.47843137]],
		[[1.        , 0.25098039, 0.99607843]],
		[[0.47843137, 0.22352941, 0.00392157]],
		[[1.        , 0.47843137, 0.22352941]],
		[[0.48235294, 0.48235294, 0.21960784]],
		[[1.        , 0.98823529, 0.47843137]],
		[[0.00784314, 0.22352941, 0.22352941]],
		[[0.01176471, 0.98039216, 0.48235294]],
		[[0.        , 0.48235294, 1.        ]],
		[[1.        , 0.17254902, 0.48235294]]
	]

	PICO_8 = [
		[[0.        , 0.        , 0.        ]],
		[[0.11372549, 0.16862745, 0.3254902 ]],
		[[0.49411765, 0.14509804, 0.3254902 ]],
		[[0.        , 0.52941176, 0.31764706]],
		[[0.67058824, 0.32156863, 0.21176471]],
		[[0.37254902, 0.34117647, 0.30980392]],
		[[0.76078431, 0.76470588, 0.78039216]],
		[[1.        , 0.94509804, 0.90980392]],
		[[1.        , 0.        , 0.30196078]],
		[[1.        , 0.63921569, 0.        ]],
		[[1.        , 0.9254902 , 0.15294118]],
		[[0.        , 0.89411765, 0.21176471]],
		[[0.16078431, 0.67843137, 1.        ]],
		[[0.51372549, 0.4627451 , 0.61176471]],
		[[1.        , 0.46666667, 0.65882353]],
		[[1.        , 0.8       , 0.66666667]]
	]

	MSX = [
		[[0.        , 0.        , 0.        ]],
		[[0.24313725, 0.72156863, 0.28627451]],
		[[0.45490196, 0.81568627, 0.49019608]],
		[[0.34901961, 0.33333333, 0.87843137]],
		[[0.50196078, 0.4627451 , 0.94509804]],
		[[0.7254902 , 0.36862745, 0.31764706]],
		[[0.39607843, 0.85882353, 0.9372549 ]],
		[[0.85882353, 0.39607843, 0.34901961]],
		[[1.        , 0.5372549 , 0.49019608]],
		[[0.8       , 0.76470588, 0.36862745]],
		[[0.87058824, 0.81568627, 0.52941176]],
		[[0.22745098, 0.63529412, 0.25490196]],
		[[0.71764706, 0.4       , 0.70980392]],
		[[0.8       , 0.8       , 0.8       ]],
		[[1.        , 1.        , 1.        ]]
	]

	MONO_OBRADINN_IBM = [
		[[0.18039216, 0.18823529, 0.21568627]],
		[[0.92156863, 0.89803922, 0.80784314]]
	]

	MONO_OBRADINN_MAC = [
		[[0.2       , 0.2       , 0.09803922]],
		[[0.89803922, 1.        , 1.        ]]
	]

	MONO_BJG = [
		[[0.9372549 , 1.        , 0.95686275]],
		[[0.17254902, 0.05882353, 0.2       ]]
	]

	MONO_BW = [
		[[0., 0., 0.]],
		[[1., 1., 1.]]
	]

	# https://superuser.com/questions/361297/what-colour-is-the-dark-green-on-old-fashioned-green-screen-computer-displays/1206781#1206781
	MONO_PHOSPHOR_AMBER = [
		[[0.15686275, 0.15686275, 0.15686275]],
		[[1.        , 0.69019608, 0.        ]]
	]

	MONO_PHOSPHOR_LTAMBER = [
		[[0.15686275, 0.15686275, 0.15686275]],
		[[1.        , 0.8       , 0.        ]]
	]

	MONO_PHOSPHOR_GREEN1 = [
		[[0.15686275, 0.15686275, 0.15686275]],
		[[0.2       , 1.        , 0.        ]]
	]

	MONO_PHOSPHOR_GREEN2 = [
		[[0.15686275, 0.15686275, 0.15686275]],
		[[0         , 1.        , 0.2       ]]
	]

	MONO_PHOSPHOR_GREEN3 = [
		[[0.15686275, 0.15686275, 0.15686275]],
		[[0         , 1.        , 0.4       ]]
	]

	MONO_PHOSPHOR_APPLE = [
		[[0.15686275, 0.15686275, 0.15686275]],
		[[0.2       , 1.        , 0.2       ]]
	]
	APPLE_II_MONO = MONO_PHOSPHOR_APPLE

	MONO_PHOSPHOR_APPLEC = [
		[[0.15686275, 0.15686275, 0.15686275]],
		[[0.4       , 1.        , 0.4       ]]
	]
	APPLE_II_MONOC = MONO_PHOSPHOR_APPLEC
