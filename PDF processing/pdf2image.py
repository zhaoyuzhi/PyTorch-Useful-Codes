import argparse
import os
import tempfile
from pdf2image import convert_from_path, convert_from_bytes

'''
with tempfile.TemporaryDirectory() as path:
    images_from_path = convert_from_path('/home/belval/example.pdf', output_folder=path)
'''

def pdf2image_func(filename, output_dir):
    # create the folder
    check_path(output_dir)
    # get the name of the input file
    input_file_name = filename.split(os.sep)[-1].split('.')[0]
    # convert the pdf to image
    with tempfile.TemporaryDirectory() as path:
        # extract images from multi-page pdf
        images_from_path = convert_from_path(filename, output_folder = path)
        # save the image
        for index, img in enumerate(images_from_path):
            imgname = input_file_name + '_' + str(index + 1) + '.jpg'
            imgpath = os.path.join(output_dir, imgname)
            img.save(imgpath)

# multi-layer folder creation
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

    '''
    filelist = get_pdf_files(opt.inpath)
    for index, pdf in enumerate(filelist):
        print('Now processing %d-th pdf file' % (index))
        pdf2image_func(pdf, opt.outpath)
    '''
    
    # pdf2image_func(opt.filename, opt.outputDir)          # single pdf file
    