import os

# single-layer folder creation
def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

# multi-layer folder creation
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    output_path = "E:\\Deblur\\123"
    check_path(output_path)
    