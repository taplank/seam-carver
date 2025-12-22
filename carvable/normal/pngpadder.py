# import edge_detector
import numpy as np 
import os 
import cv2 
os.chdir(os.getcwd())

def grab_img(name):
    image = cv2.imread(str(name))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray 

def png_padder(img, curimg, horiz):
    shape = np.shape(img)
    curshape = np.shape(curimg)
    new_img = []
    if curshape == shape:
        return curimg 
    if horiz:
        for r in range(0, curshape[0]):
            row = list(curimg[r])
            row.extend([0]*(shape[1]-curshape[1]))
            new_img.append(row)
        return np.array(new_img)
    else:
        for c in range(0, curshape[1]):
            col = list(curimg[:, c])
            col.extend([0]*(shape[0]-curshape[0]))
            new_img.append(col)
        return np.transpose(np.array(new_img))

def main():
    horiz = False
    img = grab_img('carvable.png')
    shape = np.shape(img)
    img_list = 0
    if horiz == True:
        for j in range(shape[1]):
            new_img = png_padder(img, cv2.imread(f"cachevert/carved_{j}.png", cv2.IMREAD_GRAYSCALE), True)
            filename = f"cachehorizpad/carved_{j}.png"
            cv2.imwrite(filename, np.array(new_img, dtype=np.uint8))
    else:
        for j in range(shape[0]):
            new_img = png_padder(img, cv2.imread(f"cachevert/carved_{j}.png", cv2.IMREAD_GRAYSCALE), False)
            filename = f"cachevertpad/carved_{j}.png"
            cv2.imwrite(filename, np.array(new_img, dtype=np.uint8))

if __name__ == "__main__":
    main()

