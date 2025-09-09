import glob
from PIL import Image

im = Image.open("dataset/0/0.png")

pix = im.load()

print(pix)