import cv2
import random,os
import numpy as np
import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np


def sharpen_image(image):
    r = np.random.randint(10000)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(Folder_name+"/Sharpen-"+str(r)+ Extension, image)

def emboss_image(image):
    r = np.random.randint(10000)
    kernel_emboss_1=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    image = cv2.filter2D(image, -1, kernel_emboss_1)+128
    cv2.imwrite(Folder_name + "/Emboss-" +str(r)+ Extension, image)

def edge_image(image,ksize):
    r = np.random.randint(10000)
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    cv2.imwrite(Folder_name + "/Edge-"+str(ksize)+str(r) + Extension, image)

def addeptive_gaussian_noise(image):
    r = np.random.randint(10000)
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    cv2.imwrite(Folder_name + "/Addeptive_gaussian_noise-" +str(r)+ Extension, image)

def salt_image(image,p,a):
    r = np.random.randint(10000)
    noisy=image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    cv2.imwrite(Folder_name + "/Salt-"+str(p)+"*"+str(a) +str(r)+ Extension, image)

def paper_image(image,p,a):
    r = np.random.randint(10000)
    noisy=image
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name + "/Paper-" + str(p) + "*" + str(a) +str(r)+ Extension, image)

def salt_and_paper_image(image,p,a):
    r = np.random.randint(10000)
    noisy=image
    #salt
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1

    #paper
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name + "/Salt_And_Paper-" + str(p) + "*" + str(a)+str(r) + Extension, image)

def contrast_image(image,contrast):
    r = np.random.randint(10000)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/Contrast-" + str(contrast) +str(r)+ Extension, image)

def edge_detect_canny_image(image,th1,th2):
    r = np.random.randint(10000)
    image = cv2.Canny(image,th1,th2)
    cv2.imwrite(Folder_name + "/Edge Canny-" + str(th1) + "*" + str(th2)+str(r) + Extension, image)

def grayscale_image(image):
    r = np.random.randint(10000)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(Folder_name + "/Grayscale-" +str(r)+ Extension, image)

def multiply_image(image,R,G,B):
    image=image*[R,G,B]
    r = np.random.randint(10000)
    cv2.imwrite(Folder_name+"/Multiply-"+str(R)+"*"+str(G)+"*"+str(B)+str(r)+Extension, image)
def gausian_blur(image,blur):
    r = np.random.randint(10000)
    image = cv2.GaussianBlur(image,(5,5),blur)
    cv2.imwrite(Folder_name+"/GausianBLur-"+str(blur)+str(r)+Extension, image)
def averageing_blur(image,shift):
    r = np.random.randint(10000)
    image=cv2.blur(image,(shift,shift))
    cv2.imwrite(Folder_name + "/AverageingBLur-" + str(shift) +str(r)+ Extension, image)
def median_blur(image,shift):
    r = np.random.randint(10000)
    image=cv2.medianBlur(image,shift)
    cv2.imwrite(Folder_name + "/MedianBLur-" + str(shift) +str(r)+ Extension, image)
def bileteralBlur(image,d,color,space):
    r = np.random.randint(10000)
    image = cv2.bilateralFilter(image, d,color,space)
    cv2.imwrite(Folder_name + "/BileteralBlur-"+str(d)+"*"+str(color)+"*"+str(space)+ str(r)+Extension, image)
def erosion_image(image,shift):
    r = np.random.randint(10000)
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name + "/Erosion-"+"*"+str(shift) + str(r)+Extension, image)
def dilation_image(image,shift):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name + "/Dilation-" + "*" + str(shift)+ str(r)+Extension, image)
def opening_image(image,shift):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(Folder_name + "/Opening-" + "*" + str(shift)+ str(r)+Extension, image)
def closing_image(image, shift):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(Folder_name + "/Closing-" + "*" + str(shift) +str(r)+ Extension, image)
def morphological_gradient_image(image, shift):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(Folder_name + "/Morphological_Gradient-" + "*" + str(shift) + str(r)+Extension, image)
def top_hat_image(image, shift):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite(Folder_name + "/Top_Hat-" + "*" + str(shift) + str(r)+Extension, image)
def black_hat_image(image, shift):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(Folder_name + "/Black_Hat-" + "*" + str(shift) + str(r)+Extension, image)

def resize_image(image,w,h):
    image=cv2.resize(image,(w,h))
    cv2.imwrite(Folder_name+"/Resize-"+str(w)+"*"+str(h)+Extension, image)
def crop_image(image,y1,y2,x1,x2):
    image=image[y1:y2,x1:x2]
    cv2.imwrite(Folder_name+"/Crop-"+str(x1)+str(x2)+"*"+str(y1)+str(y2)+Extension, image)
def padding_image(image,topBorder,bottomBorder,leftBorder,rightBorder,color_of_border=[0,0,0]):
    image = cv2.copyMakeBorder(image,topBorder,bottomBorder,leftBorder,
        rightBorder,cv2.BORDER_CONSTANT,value=color_of_border)
    cv2.imwrite(Folder_name + "/padd-" + str(topBorder) + str(bottomBorder) + "*" + str(leftBorder) + str(rightBorder) + Extension, image)
def flip_image(image,dir):
    image = cv2.flip(image, dir)
    cv2.imwrite(Folder_name + "/flip-" + str(dir)+Extension, image)
def superpixel_image(image,segments):
    seg=segments

    def segment_colorfulness(image, mask):
        # split the image into its respective RGB components, then mask
        # each of the individual RGB channels so we can compute
        # statistics only for the masked region
        (B, G, R) = cv2.split(image.astype("float"))
        R = np.ma.masked_array(R, mask=mask)
        G = np.ma.masked_array(B, mask=mask)
        B = np.ma.masked_array(B, mask=mask)

        # compute rg = R - G
        rg = np.absolute(R - G)

        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        # compute the mean and standard deviation of both `rg` and `yb`,
        # then combine them
        stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))

        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)

    orig = cv2.imread(image)
    vis = np.zeros(orig.shape[:2], dtype="float")

    # load the image and apply SLIC superpixel segmentation to it via
    # scikit-image
    image = io.imread(image)
    segments = slic(img_as_float(image), n_segments=segments,
                    slic_zero=True)
    for v in np.unique(segments):
        # construct a mask for the segment so we can compute image
        # statistics for *only* the masked region
        mask = np.ones(image.shape[:2])
        mask[segments == v] = 0

        # compute the superpixel colorfulness, then update the
        # visualization array
        C = segment_colorfulness(orig, mask)
        vis[segments == v] = C
    # scale the visualization image from an unrestricted floating point
    # to unsigned 8-bit integer array so we can use it with OpenCV and
    # display it to our screen
    vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")

    # overlay the superpixel colorfulness visualization on the original
    # image
    alpha = 0.6
    overlay = np.dstack([vis] * 3)
    output = orig.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # cv2.imshow("Visualization", vis)
    cv2.imwrite(Folder_name + "/superpixels-" + str(seg) + Extension, output)
def invert_image(image,channel):
    # image=cv2.bitwise_not(image)
    image=(channel-image)
    cv2.imwrite(Folder_name + "/invert-"+str(channel)+Extension, image)
def add_light(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/light-"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dark-" + str(gamma) + Extension, image)
def add_light_color(image, color, gamma=1.0):
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/light_color-"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dark_color" + str(gamma) + Extension, image)
def saturation_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/saturation-" + str(saturation) + Extension, image)
def hue_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/hue-" + str(saturation) + Extension, image)
class Data_augmentation:
    def __init__(self, path, image_name):
        self.path = path
        self.name = image_name
        self.image = cv2.imread(path + image_name)

    def noisy(self, img, noise_type="gauss"):
        if noise_type == "gauss":
            image = img.copy()
            mean = 0
            st = 0.7
            gauss = np.random.normal(mean, st, image.shape)
            gauss = gauss.astype('uint8')
            image = cv2.add(image, gauss)
            return image
        elif noise_type == "sp":
            image = img.copy()
            prob = 0.05
            if len(image.shape) == 2:
                black = 0
                white = 255
            else:
                colorspace = image.shape[2]
                if colorspace == 3:  # RGB
                    black = np.array([0, 0, 0], dtype='uint8')
                    white = np.array([255, 255, 255], dtype='uint8')
                else:  # RGBA
                    black = np.array([0, 0, 0, 255], dtype='uint8')
                    white = np.array([255, 255, 255, 255], dtype='uint8')
            probs = np.random.random(image.shape[:2])
            image[probs < (prob / 2)] = black
            image[probs > 1 - (prob / 2)] = white
            return image

    def colorjitter(self,img, cj_type="b"):
        '''
        ### Different Color Jitter ###
        img: image
        cj_type: {b: brightness, s: saturation, c: constast}
        '''
        if cj_type == "b":
            # value = random.randint(-50, 50)
            value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if value >= 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = np.absolute(value)
                v[v < lim] = 0
                v[v >= lim] -= np.absolute(value)

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            return img

        elif cj_type == "s":
            # value = random.randint(-50, 50)
            value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if value >= 0:
                lim = 255 - value
                s[s > lim] = 255
                s[s <= lim] += value
            else:
                lim = np.absolute(value)
                s[s < lim] = 0
                s[s >= lim] -= np.absolute(value)

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            return img

        elif cj_type == "c":
            brightness = 10
            contrast = random.randint(40, 100)
            dummy = np.int16(img)
            dummy = dummy * (contrast / 127 + 1) - contrast + brightness
            dummy = np.clip(dummy, 0, 255)
            img = np.uint8(dummy)
            return img

    def rotate(self, image, angle=90, scale=1.0):
        w = image.shape[1]
        h = image.shape[0]
        # rotate matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        # rotate
        image = cv2.warpAffine(image, M, (w, h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def image_augment(self, save_path):
        img = self.image.copy()
        img_flip = self.flip(img, vflip=True, hflip=False)
        img_rot = self.rotate(img)
        img_gaussian = self.noisy(img)
        img_salt_and_pepper = self.noisy(img, "sp")
        img_color_jit_b=self.colorjitter(img, "b")
        img_color_jit_c = self.colorjitter(img, "c")
        img_color_jit_sh = self.colorjitter(img, "s")

        name_int = self.name[:len(self.name) - 4]
        cv2.imwrite(save_path + '%s' % str(name_int) + '_vflip.jpg', img_flip)
        cv2.imwrite(save_path + '%s' % str(name_int) + '_rot.jpg', img_rot)
        cv2.imwrite(save_path + '%s' % str(name_int) + '_GaussianNoise.jpg', img_gaussian)
        cv2.imwrite(save_path + '%s' % str(name_int) + 'img_salt_and_pepper.jpg', img_salt_and_pepper)
        cv2.imwrite(save_path + '%s' % str(name_int) + 'img_color_jit_b.jpg', img_color_jit_b)
        cv2.imwrite(save_path + '%s' % str(name_int) + 'img_color_jit_c.jpg', img_color_jit_c)
        cv2.imwrite(save_path + '%s' % str(name_int) + 'img_color_jit_sh.jpg', img_color_jit_sh)

def first_block(image):
    resize_image(image,450,400)
    crop_image(image,0,300,100,450)#(y1,y2,x1,x2)(bottom,top,left,right)
    crop_image(image,100,300,100,350)#(y1,y2,x1,x2)(bottom,top,left,right)
    padding_image(image,100,100,100,100)#(y1,y2,x1,x2)(bottom,top,left,right)
    flip_image(image,0)#horizontal
    flip_image(image,1)#vertical
    flip_image(image,-1)#both
    invert_image(image,150)
    add_light(image,1.5)
    add_light(image,0.3)
    add_light_color(image,255,1.5)
    add_light_color(image,200,2.0)
    saturation_image(image,150)
    saturation_image(image,200)
    hue_image(image,50)
    hue_image(image,100)
def second_block(image):
    multiply_image(image, 1.25, 1.25, 1.25)
    multiply_image(image, 1.5, 1, 1)
    gausian_blur(image, 0.25)
    gausian_blur(image, 0.50)
    gausian_blur(image, 4)
    averageing_blur(image, 5)
    averageing_blur(image, 4)
    averageing_blur(image, 6)
    median_blur(image, 3)
    median_blur(image, 5)
    median_blur(image, 7)
    bileteralBlur(image, 25, 100, 100)
    bileteralBlur(image, 40, 75, 75)
    erosion_image(image, 1)
    erosion_image(image, 3)
    dilation_image(image, 1)
    dilation_image(image, 3)
    opening_image(image, 3)
    opening_image(image, 5)
    closing_image(image, 1)
    closing_image(image, 3)
    morphological_gradient_image(image, 10)
    morphological_gradient_image(image, 15)
    top_hat_image(image, 200)
    top_hat_image(image, 300)
    black_hat_image(image, 300)
    black_hat_image(image, 500)
def third_block(image):
    sharpen_image(image)
    emboss_image(image)
    edge_image(image, 1)
    edge_image(image, 3)
    addeptive_gaussian_noise(image)
    salt_image(image, 0.5, 0.009)
    salt_image(image, 0.5, 0.09)
    salt_image(image, 0.5, 0.9)
    paper_image(image, 0.5, 0.009)
    paper_image(image, 0.5, 0.09)
    paper_image(image, 0.5, 0.9)
    salt_and_paper_image(image, 0.5, 0.009)
    salt_and_paper_image(image, 0.5, 0.09)
    salt_and_paper_image(image, 0.5, 0.9)
    edge_detect_canny_image(image, 100, 200)
    edge_detect_canny_image(image, 200, 400)
    grayscale_image(image)

from tqdm import tqdm
import shutil
def create_test_folder():
    for path in os.listdir('stl10/train/'):
        for root, _, files in os.walk("stl10/train/"+path+"/"):
            try:
                os.mkdir("stl10/test_generated/"+path)
            except:
                pass

            original = r"C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\test_generated\""
            target = r'C:\Users\Ron\Desktop\Test_2\my_csv_file.csv'
            shutil.move(original, target)
    ""

    def create_test():
        x = 0
        for path in os.listdir('stl10/train/'):
            for root, _, files in os.walk("stl10/train/"+path+"/"):
                try:
                    os.mkdir("stl10/test_generated/"+path)
                except:
                    pass
                tot = len(files)
                sample = int(tot * 0.3)
                randomlist = random.sample(range(0, tot), sample)
                print(randomlist)
                print(sample)
                x+=len(randomlist)
                for ifile in randomlist:
                    original = r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\train'+ "\\" + path+ "\\"+files[ifile]
                    target = r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\test_generated'+ "\\" + path+ "\\"+files[ifile]
                    shutil.move(original, target)
        print(x)

def move_all():
    for path in os.listdir('stl10/train/'):
        for root, _, files in os.walk("stl10/train/" + path + "/"):
            for file in files:
                original = r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\train' + "\\" + path + "\\" + \
                           file
                target = r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\aug' + "\\" + path + "\\" + \
                         file
                shutil.move(original, target)

def balance_test():
    x = 0
    for path in tqdm(os.listdir('challenge_dataset_augmented/train_augmented/')):
        for root, _, files in os.walk("challenge_dataset_augmented/train_augmented/" + path + "/"):
            tot = len(files)
            sample = int(tot * 0.3)
            randomlist = random.sample(range(0, tot), sample)
            x += len(randomlist)
            for ifile in randomlist:
                if (os.path.exists( r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\challenge_dataset_augmented\train_augmented'+"\\"+path+"\\"+files[ifile])):
                    origin=(( r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\challenge_dataset_augmented\train_augmented'+"\\"+path+"\\"+files[ifile]))
                    target = r'challenge_dataset_augmented\test_generated' + "\\" + path + "\\" +  files[ifile]
                    shutil.move(origin, target)
    print(x)
if __name__ == "__main__":
    Extension=".jpg"
    for path in tqdm(os.listdir('stl10/train/')):
        for root, _, files in os.walk("stl10/train/"+path+"/"):
            try:
                os.mkdir("stl10/aug/"+path)
            except:
                pass
            Folder_name = "stl10/aug/" + path + "/"
            for file in files:
                if len(os.listdir(Folder_name)) < 500:
                    image_file= root+file
                    image=cv2.imread(image_file)
                    first_block(image)
                    second_block(image)
                    third_block(image)
            if len(os.listdir(Folder_name)) < 500:
                while len(os.listdir(Folder_name)) < 500:
                    random = np.random.randint(len(os.listdir(Folder_name)))
                    file = files[random]
                    image_file= root+file
                    image=cv2.imread(image_file)
                    first_block(image)
                    second_block(image)
                    third_block(image)

    for path in os.listdir('stl10/train/'):
        for root, _, files in os.walk("stl10/train/" + path + "/"):
            for file in files:
                original = r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\train' + "\\" + path + "\\" + \
                           file
                target = r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\aug' + "\\" + path + "\\" + \
                         file
                shutil.move(original, target)

    for path in os.listdir('stl10/aug/'):
        for root, _, files in os.walk("stl10/aug/" + path + "/"):
            try:
                os.mkdir("stl10/test_generated/" + path)
            except:
                pass
            tot = len(files)
            sample = int(tot * 0.3)
            randomlist = random.sample(range(0, tot), sample)
            for ifile in randomlist:
                original = r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\aug' + "\\" + path + "\\" + \
                           files[ifile]
                target = r'C:\Users\DELL\Desktop\UNI TN\Applied Machine Learning\Convolutional Neural Netowork - Torch\stl10\test_generated' + "\\" + path + "\\" + \
                         files[ifile]
                shutil.move(original, target)





