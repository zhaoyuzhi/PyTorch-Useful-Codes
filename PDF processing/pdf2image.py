import argparse
import os
import tempfile
from pdf2image import convert_from_path, convert_from_bytes

def extract(filename, outputDir):
    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(filename, output_folder = path)
        for index, img in enumerate(images):
            # create the folder
            midname = filename.split('\\')[-1][:-4]
            folderpath = os.path.join(outputDir, midname)
            if os.path.exists(folderpath):
                print("Re-arrange the folder")
                raise SystemExit
            else:
                os.mkdir(folderpath)
            # save the image
            imgname = str(index) + '.jpg'
            imgpath = os.path.join(folderpath, imgname)
            img.save(imgpath)

def get_pdf_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-3:] == 'pdf':
                ret.append(os.path.join(root, filespath))
    return ret

# Read all the pdf files in a certain path
# Save all the pdf files to folders containing same name of pdf names
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type = str, default = 'sample.pdf', help = 'single pdf file name')
    parser.add_argument('--inpath', type = str, default = './sampleout', help = 'inpath including many pdf files')
    parser.add_argument('--outpath', type = str, default = './', help = 'outpath')
    opt = parser.parse_args()

    filelist = get_pdf_files(opt.inpath)
    for index, pdf in enumerate(filelist):
        print('Now processing %d-th pdf file' % (index))
        extract(pdf, opt.outpath)
    
    # extract(opt.filename, opt.outputDir)          # single pdf file
    