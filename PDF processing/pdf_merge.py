import argparse
import os
from PyPDF2 import PdfReader, PdfWriter

def pdf_merger(folder_path, save_path):
    # get all PDF files in folder_path
    pdf_lst = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    # or you can define the PDF files
    pdf_lst = ['滴滴出行行程报销单A.pdf', '滴滴出行行程报销单B.pdf', '滴滴电子发票A.pdf', '滴滴电子发票B.pdf', '市内的士发票C.pdf', '高速发票1.pdf', '高速发票2.pdf', '高速发票3.pdf', '高速发票4.pdf', '高速发票5.pdf', '酒店发票.pdf', '机票发票1.pdf', '机票发票2.pdf']
    print(pdf_lst)

    # concat all PDF files
    pdf_lst = [os.path.join(folder_path, filename) for filename in pdf_lst]
    
    # merge PDFs
    file_merger = PdfWriter()
    for pdf in pdf_lst:
        file_merger.append(pdf)

    # save to new file
    file_merger.write(save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type = str, default = '202411 - second - 1117 (beijing)', help = 'inpath including many pdf files')
    parser.add_argument('--save_path', type = str, default = '202411 - second - 1117 (beijing)/output.pdf', help = 'outpath')
    opt = parser.parse_args()

    pdf_merger(opt.folder_path, opt.save_path)
    