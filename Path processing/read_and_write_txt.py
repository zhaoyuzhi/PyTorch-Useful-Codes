import os

# Use the following command if there is Chinese
# file = open(filename, mode, encoding='utf-8')

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# read a txt expect EOF
def text_readlines(filename):
    # try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

if __name__ == '__main__':

    '''
    text_save(fullname, "./ILSVRC2012_train_224.txt")
    #text_save(jpgname, "")
    print("successfully saved")
    '''
    '''
    a = text_readlines("C:\\Users\\ZHAO Yuzhi\\Desktop\\code\\Colorization\\ILSVRC2012_train_name.txt")
    print(len(a))
    '''
