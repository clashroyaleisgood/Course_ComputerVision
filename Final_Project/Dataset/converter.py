import cv2
import os

'''
place folder like:
|-- Dataset/
|   |-- bull/
|       |-- bull_o_a.gif  # stereo images
|       |-- bull_o_d.jpg  # disparity map
|-- converter.py

splitGIF() will create
|-- Dataset/
|   |-- bull/
        |-- stereo/
            |-- bull_1.jpg
            |-- bull_2.jpg
            |-- ...
'''

def convert(folder_name):
    '''
    conver all {folder_name}/[*ppm, *pgm] 
           to  {folder_name + "_new"}/[*png]
    
    * Note that: folder_name should be the same level to this .py file
    '''
    print(f'Converting {folder_name}/ to {folder_name}_new/')

    # ImageDirectory = path.join('dataset', 'example', folder_name)
    ImageDirectory = os.path.join(os.path.dirname(__file__), folder_name)
    NewDirectory   = os.path.join(os.path.dirname(__file__), folder_name + '_new')
    os.makedirs(NewDirectory, exist_ok=True)

    for filename in os.listdir(ImageDirectory):
        fullpath = os.path.join(ImageDirectory, filename)
        image = None
        if filename.split('.')[-1] == 'ppm':  # RGB color
            image = cv2.imread(fullpath)
        elif filename.split('.')[-1] == 'pgm':  # Gray scale
            image = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
        NewFilename = f'{filename.rsplit(".", 1)[0]}.jpg'  # split from right to left, split up to once

        cv2.imwrite(os.path.join(NewDirectory, NewFilename), image)

def splitGIF(folder_name):
    '''
    split *.gif
    to images/{folder_name}_[1, 2, 3, ...].jpg
    '''

    folderpath = os.path.join(os.path.dirname(__file__), folder_name)
    filename = os.listdir(folderpath)[0]
    fullpath = os.path.join(folderpath, filename)

    print(f'Spliting {filename} to {filename}/stereo/*.jpg')

    folderpath_stored = os.path.join(folderpath, 'stereo')
    os.makedirs(folderpath_stored, exist_ok=True)

    counter = 1

    # image = cv2.imread(os.path.join(folderpath, filename))
    cap = cv2.VideoCapture(fullpath)

    while True:
        ret, image = cap.read()
        if ret:
            cv2.imwrite(os.path.join(folderpath_stored,
                        f'{folder_name}_{counter}.jpg'), image)  # jpg is good enough
            counter += 1
            # cv2.imshow('Window', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            break

    cap.release()

def compress(folder_path, dsize):
    for filename in os.listdir(folder_path):
        fullpath = os.path.join(folder_path, filename)
        print(f'relace {fullpath} to size ({dsize[1], dsize[0]})')
        image = cv2.imread(fullpath)
        image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fullpath, image)

if __name__ == '__main__':
    # convert('tsukuba')
    # splitGIF('bull')
    compress('Final_Project\\Dataset\\self_laptop\\unrect', (1000, 750))
