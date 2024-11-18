# coding = utf-8
import argparse
import os
import img2pdf
from PIL import Image

IMG_SUFFIX = ['png', 'jpg', 'jpeg', 'gif', 'ico', 'PNG', 'JPG', 'JPEG']

'''
# opening from filename
with open("name.pdf","wb") as f:
	f.write(img2pdf.convert('test.jpg'))

# opening from file handle
with open("name.pdf","wb") as f1, open("test.jpg") as f2:
	f1.write(img2pdf.convert(f2))

# opening using pathlib
with open("name.pdf","wb") as f:
	f.write(img2pdf.convert(pathlib.Path('test.jpg')))

# using in-memory image data
with open("name.pdf","wb") as f:
	f.write(img2pdf.convert("\x89PNG...")

# multiple inputs (variant 1)
with open("name.pdf","wb") as f:
	f.write(img2pdf.convert("test1.jpg", "test2.png"))

# multiple inputs (variant 2)
with open("name.pdf","wb") as f:
	f.write(img2pdf.convert(["test1.jpg", "test2.png"]))
'''

def image2pdf_func(filename, savename):
    # opening image
    image = Image.open(filename)
    # converting into chunks using img2pdf
    pdf_bytes = img2pdf.convert(image.filename)
    # opening or creating pdf file
    file = open(savename, "wb")
    # writing pdf files with chunks
    file.write(pdf_bytes)
    # closing image file
    image.close()
    # closing pdf file
    file.close()

# multi-layer folder creation
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_img_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath.split('.')[-1] in IMG_SUFFIX:
                ret.append(os.path.join(root, filespath))
    return ret

# Read all the pdf files in a certain path
# Save all the pdf files to folders containing same name of pdf names
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type = str, default = './202411 - second - 1117 (beijing)/市内的士发票C.jpeg', help = 'inpath')
    parser.add_argument('--outpath', type = str, default = './202411 - second - 1117 (beijing)/市内的士发票C.pdf', help = 'outpath')
    opt = parser.parse_args()
    
    image2pdf_func(opt.inpath, opt.outpath)          # single image file
