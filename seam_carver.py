# import edge_detector
import numpy as np 
import os 
import cv2 
#for video generation
import subprocess
os.chdir(os.getcwd())
path_to_cache = "carvable/normal"
path_to_img = 'carvable/carvable.png'

#make video
def make_video_from_frames(frames_dir, out_path, fps, full_w, full_h, start_number=1):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-start_number", str(start_number),
        "-i", os.path.join(frames_dir, "carved_%d.png"),
        "-vf", f"pad={full_w}:{full_h}:0:0:black,format=yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        out_path
    ]
    subprocess.run(cmd, check=True)


def gen_vals(img):
    # today we will use the Sobel operator
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    vals = []
    test_matrix = 0
    test = False
    #convert to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    old_img = img
    img = np.pad(img, 1, 'edge')
    for row in range(0, len(old_img)):
        current_row_vals = []
        for col in range(0, len(old_img[row])):              
            matrix = np.array(img[row:row+3, col:col+3])
            o_x = np.sum(g_x*matrix) 
            o_y = np.sum(g_y*matrix)
            #o_x = g_x*test_matrix
            #o_y = g_y*test_matrix
            g = np.sqrt(o_x*o_x + o_y*o_y)
            #get row of vals
            current_row_vals.append(g)
        #make list of rows = col of rows = matrix
        vals.append(current_row_vals)
    return np.array(vals) 

    # run edge detector on image
    # run the edge detection to get points for each pixel 
def remove_seam(img, seam):
    new_img = []
    for r in range(len(img)):
        row = list(img[r])
        row.pop(seam[r])
        new_img.append(row)
    #combine these into an image 
    return np.array(new_img)

def carve_seam(table, img):
    rows = len(table)
    cols = len(table[0])
    final_table = []
    prev_row = None
    for r in range(0, rows):
        row = table[r]
        if r==0:
            final_table.append(list(row))
            prev_row = table[0]
        else:
            row_final = []
            prev_row = final_table[r-1]
            row = table[r]
            if len(prev_row) != len(row):
                return "Error"
            # for each square in this row
            # note that indexes are left indented
            for i in range(0, cols):
                # set min val = infinity 
                min_prev = float('inf')
                for j in range(i-1, i+2):
                    if j>=0 and j<cols:
                        if prev_row[j]<min_prev:
                            #iterate through each val, if it's less than update min_val 
                            min_prev = prev_row[j]
                    #append min_val+row_val to row_final
                row_final.append(min_prev + row[i])
            final_table.append(row_final)
    path = []
    for r in range(0, rows):
        row = final_table[rows-1-r]
        if r == 0:
            min_valdex = row.index(min(row))
            path.append(min_valdex)
        left = max(0, min_valdex-1)
        right = min(cols, min_valdex+2)
        if r != rows-1: 
            next_row = final_table[rows-1-(r+1)]
            min_val = min(next_row[left:right])
            min_valdex = next_row[left:right].index(min_val) + left
            path.append(min_valdex)

    path.reverse()
    finmg = remove_seam(img, path)
    return finmg

def main():
    result = input("horiz (horizontal) or vert (vertical) or both? Reply exactly.")
    if result == 'horiz':
        horiz = True
    else:
        horiz = False
    img = cv2.imread(path_to_img)
    shape = np.shape(img)
    truimg = img
    graymg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not horiz:
        img = np.transpose(img, axes=(1,0,2))
    #we do our calcs in transpose form bc I didn't feel like writing out the a[:, b] for the entire carve_seam code.
    #whenever we display the image we transpose back, if it's vert

    #Also you might be wondering what the variable axes does.
    #axes is the var that sets where the new axes come from. it maps the new axes (0, 1, 2) to the old axes, which here are (1, 0, 2)
    #so axis 1 --> axis 0, axis 0 --> axis 1, axis 2 --> axis 2 (stays same bc its a color channel)
    image_cache = {0:truimg}
    if not horiz:
        for i in range(0, shape[0]):
            print(i)
            print(np.shape(truimg))
            image_cache[i] = truimg
            filename = f"{path_to_cache}/vert/carved_{i}.png"
            cv2.imwrite(filename, np.array(truimg, dtype=np.uint8)) 
            vals = gen_vals(img)
            img = carve_seam(vals, img)
            truimg = np.transpose(img.copy(), axes=(1,0,2))
        make_video_from_frames(
            frames_dir=f"{path_to_cache}/vert",
            out_path=f"{path_to_cache}/vert.mp4",
            fps=30,
            full_w=shape[1],
            full_h=shape[0],
            start_number=0
        )
    if horiz:
        for i in range(0, shape[1]):
            print(i)
            print(np.shape(truimg))
            image_cache[i] = truimg
            filename = f"{path_to_cache}/horiz/carved_{i}.png"
            cv2.imwrite(filename, np.array(truimg, dtype=np.uint8)) 
            vals = gen_vals(img)
            img = carve_seam(vals, img)
            truimg = img
        #make the video from the carved stuff 
        make_video_from_frames(
            frames_dir=f"{path_to_cache}/horiz",
            out_path=f"{path_to_cache}/horiz/horiz.mp4",
            fps=30,
            full_w=shape[1],
            full_h=shape[0],
            start_number=0
        )


if __name__ == "__main__":
    main()

