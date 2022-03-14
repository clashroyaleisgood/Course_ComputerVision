import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

image_row = 0
image_col = 0
IMAGE_FOLDER_PATH = [r'test/bunny/', r'test/star/', r'test/venus/']

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image

def SVD_inv(A):
    '''
    A: m x n matrix
       3 x 2 for [[0 0], [0 0], [0 0]]
    '''
    m, n = A.shape
    u, s, v = np.linalg.svd(A)  # mxm, mxn, nxn
    l = abs(m - n)
    s = np.append(1/s, [0]*l)
    Ainv = v.T @ np.diag(s)[:n, :m] @ u.T

    return Ainv

def get_LightSource(filepath):
    Light = []
    with open(filepath, 'r') as f:
        for line in f:
            single_light = line.split()[1][1:-1].split(',')
            single_light = np.array([int(e) for e in single_light])
            norm = np.linalg.norm(single_light)
            single_light = single_light.astype('float64') / norm

            Light += [single_light]
    return np.array(Light)

def iter_bmp_paths(filepath):
    for i in range(1, 7):
        yield f'{filepath}pic{i}.bmp'

def get_ImageMatrix(filepaths):
    '''
    return: (rows, cols, 6)
    '''
    I = []
    for filepath in filepaths:
        bmp = read_bmp(filepath)
        bmp = bmp.astype('float64')
        bmp /= 255
        I += [bmp]
    return np.stack(I, axis=2)

def get_Mask(filepaths):
    '''
    SHOULD be excuted after get_ImageMatrix()
    return : (rows, cols) -> bool
    '''
    Mask = np.zeros((image_row, image_col), dtype=np.bool_)

    for filepath in filepaths:
        bmp = read_bmp(filepath)
        Mask = np.greater(bmp, 0) | Mask
    return Mask

def get_Normal(Linv, Images):
    '''
    Linv: LightInverse
    Images: Images(rows, cols, 6)
    return: (rows, cols, 3)
    '''
    Normal = []
    for y in range(image_row):
        RowNormal = []
        for x in range(image_col):
            KdN = Linv @ Images[y][x]
            norm = np.linalg.norm(KdN)
            N = KdN / norm if norm > 0.000001 else KdN

            RowNormal += [N]
        Normal += [RowNormal]
    Normal = np.array(Normal)
    return Normal

def get_Gradientxy(Normal):
    '''
    Normal: (rows, cols, 3)
    return: (rows, cols, 2)
                         [dz/dx, dz/dy]
    '''
    Gradient = []
    for y in range(image_row):
        RowGradient = []
        for x in range(image_col):
            Na, Nb, Nc = Normal[y][x]
            dzdx = -Na/Nc if Nc > 0.000001 else 0
            dzdy = -Nb/Nc if Nc > 0.000001 else 0
            RowGradient += [[dzdx, dzdy]]
        Gradient += [RowGradient]
    Gradient = np.array(Gradient)
    return Gradient

def Reconstruct(Gradient, Mask):
    '''
    Gradient: (rows, cols, 2)
                           [dz/dx, dz/dy]
    return: (rows, cols) -> depth map
    '''
    # Start from top down
    Surface = np.zeros((image_row, image_col))

    for y in range(1, image_row):
        for x in range(1, image_col):
            if not Mask[y][x]:
                continue
            # Compute from left
            # Z = Z(x-1, y) + dz/dx(x-1, y)
            Surface[y][x] += Surface[y][x-1] + Gradient[y][x-1][0]
            # Compute from top
            # Z = Z(x, y-1) + dy/dx(x, y-1)
            Surface[y][x] += Surface[y-1][x] - Gradient[y-1][x][1]
            Surface[y][x] /= 2

    Surface2 = np.zeros((image_row, image_col))
    for y in range(image_row-2, 0, -1):
        for x in range(image_col-2, 0, -1):
            if not Mask[y][x]:
                continue
            # Compute from right
            # Z = Z(x+1, y) - dz/dx(x, y)
            Surface2[y][x] += Surface2[y][x+1] - Gradient[y][x][0]
            # Compute from down
            # Z = Z(x, y+1) - dy/dx(x, y)
            Surface2[y][x] += Surface2[y+1][x] + Gradient[y][x][1]
            Surface2[y][x] /= 2

    return (Surface + Surface2) / 2

def ReconstructC(Gradient, Mask):
    '''
    Reconstruct FROM CENTER
    Gradient: (rows, cols, 2)
                           [dz/dx, dz/dy]
    return: (rows, cols) -> depth map
    '''
    x_center = image_col // 2
    y_center = image_row // 2

    Surface = np.zeros((image_row, image_col))
    # mid -> right
    for x in range(x_center+1, image_col):
        if not Mask[y_center][x]:
            continue
        Surface[y_center][x] = Surface[y_center][x-1] + Gradient[y_center][x-1][0]
    # mid -> down
    for x in range(x_center-1, 0, -1):
        if not Mask[y_center][x]:
            continue
        Surface[y_center][x] = Surface[y_center][x+1] - Gradient[y_center][x][0]
    # mid -> left
    for y in range(y_center+1, image_row):
        if not Mask[y][x_center]:
            continue
        Surface[y][x_center] = Surface[y-1][x_center] - Gradient[y-1][x_center][1]
    # mid -> up
    for y in range(y_center-1, 0, -1):
        if not Mask[y][x_center]:
            continue
        Surface[y][x_center] = Surface[y+1][x_center] + Gradient[y][x_center][1]
    
    # mid -> right_down
    for y in range(y_center+1, image_row):
        for x in range(x_center+1, image_col):
            if not Mask[y][x]:
                continue
            Surface[y][x] = (
                Surface[y][x-1] + Gradient[y][x-1][0] + # form Left
                Surface[y-1][x] - Gradient[y-1][x][1]   # from Up
            ) / 2
    # mid -> right_up
    for y in range(y_center-1, 0, -1):
        for x in range(x_center+1, image_col):
            if not Mask[y][x]:
                continue
            Surface[y][x] = (
                Surface[y][x-1] + Gradient[y][x-1][0] + # from Left
                Surface[y+1][x] + Gradient[y][x][1]     # from Down
            ) / 2
    # mid -> left_up
    for y in range(y_center-1, 0, -1):
        for x in range(x_center-1, 0, -1):
            if not Mask[y][x]:
                continue
            Surface[y][x] = (
                Surface[y][x+1] - Gradient[y][x][0] +   # form Right
                Surface[y+1][x] + Gradient[y][x][1]     # from Down
            ) / 2
    # mid -> left_down
    for y in range(y_center+1, image_row):
        for x in range(x_center-1, 0, -1):
            if not Mask[y][x]:
                continue
            Surface[y][x] = (
                Surface[y][x+1] - Gradient[y][x][0] +   # form Right
                Surface[y-1][x] - Gradient[y-1][x][1]   # from Up
            ) / 2
    return Surface


if __name__ == '__main__':
    FolderPath = IMAGE_FOLDER_PATH[0]
    LightPath = f'{FolderPath}/LightSource.txt'
    LightSource = get_LightSource(LightPath)
    LightInverse = SVD_inv(LightSource)

    ImageMatrix = get_ImageMatrix(iter_bmp_paths(FolderPath))
    Mask = get_Mask(iter_bmp_paths(FolderPath))

    N = get_Normal(LightInverse, ImageMatrix)
    # normal_visualization(N)

    G = get_Gradientxy(N)
    Z = Reconstruct(G, Mask)
    depth_visualization(Z)
    # showing the windows of all visualization function
    plt.show()