import os
from PIL import Image
size = (16,12)
framerate = 15
maxFrames = 0

framesDir = "E:/Video/bad apple/frames/30fps/2bpp"
files = os.listdir(framesDir)

if maxFrames == 0:
	maxFrames = len(files)

chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

for y in range(size[1]):
	frame = 0
	string = ""
	while frame < min(len(files)-1,maxFrames):
		for frameNum in range(min(len(files),maxFrames)-frame):
			frame += 1
			if frameNum % 20 == 0:
				print(f"Frame {frame}")

			file = files[frame-1]
			img = Image.open(os.path.join(framesDir, file))
			resized = img.resize(size)
			px = resized.load()
			frameString = ""
			for x in range(16):
				if px[x,y] == 1:
					frameString = "1" + frameString
				else:
					frameString = "0" + frameString

			p0 = int(frameString[10:16],2)
			p1 = int(frameString[4:10],2)
			p2 = int(frameString[0:4],2)
			b64Bytes = chars[p0] + chars[p1] + chars[p2]
			string += b64Bytes
	print(string)

	string = string.ljust(12288, "A")

	with open(f"output_{y}.txt", "w") as f:
		f.write(string)
print(f"Run at {framerate} tps for correct speed.")
