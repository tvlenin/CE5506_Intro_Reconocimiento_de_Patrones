
from PIL import ImageTk, Image, ImageDraw
import PIL
from Tkinter import *
import glob, os
from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

if(len(sys.argv) < 2):
    print("Need one argument")
    print("Example : python test_Mnist.py saved/new")
    sys.exit("exit")
if(sys.argv[1] == "saved"):
    model = load_model('my_model_60000.h5')
else:
    model = load_model('my_model.h5')

width = 200
height = 200
center = height//2
white = (0,0, 0)
green = (0,128,0)

def save():
    filename = "image.jpg"
    image1.save(filename)
    size = 28, 28
    file, ext = os.path.splitext("image.jpg")
    im = PIL.Image.open("image.jpg")
    im.thumbnail(size)
    im.save(file + ".png", "PNG")
#predict code
    im = imread("image.png", as_grey="False" )
    arr = np.array(im)
    pr = model.predict_classes(arr.reshape(1,28,28,1))
    print(pr)
    plt.imshow(im.reshape(28,28))
    plt.show()



def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="white",width=5)
    draw.line([x1, y1, x2, y2],fill="white",width=5)

root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()
image1 = PIL.Image.new("RGB", (height, height), white)
draw = ImageDraw.Draw(image1)
cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)
button=Button(text="save",command=save)
button.pack()





root.mainloop()
