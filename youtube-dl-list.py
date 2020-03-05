import os
import youtube_dl

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
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def download(url):
    ydl_opts = {
        'format': 'bestvideo'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.download([url])

filelist = text_readlines('videolist6.txt')
print(filelist)

for i, item in enumerate(filelist):
    print('The %d-th video named %s' % (i, item))
    download(item)
