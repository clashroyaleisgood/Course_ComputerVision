import cv2
import os

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

if __name__ == '__main__':
    convert('tsukuba')
