#%%
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import glob
import cv2
import pandas as pd
import pptk
import module_visualization


class pointcloud:

    def __init__(self):
        self.points = []
        self.intensity = []
        self.rgb = []
        self.vtk = module_visualization.visualization_library

    def modes(self,histogram):
        """Returns min index of histogram values of a image
        GS edits:
            1. which is "high_1" mean?
            2. can you make the high_1 as a more descriptive name?
        """

        high_1 = 0
        ind_1 = 0
        high_2 = 0
        ind_2 = 0

        for i in range (len(histogram)):
            if histogram[i] > high_1:
                high_1 = histogram[i]
                ind_1 = i
        for i in range(len(histogram)):
            if histogram[i] > high_2 and histogram[i] <= high_1:
                high_2 = histogram[i]
                ind_2 = i

        return min (ind_1, ind_2)

    def thresholding(self, img_temp, gray):
        """Applies thresholding to the B Scans to lower noise
        argvs:
            what is img_tmp
            what ia gray for image or data?
        returns:
            what is the output?
        """

        # blur gray image, threshold it, convert threshold back to bgr for the template
        img_blur= cv2.medianBlur(gray, 3)

        # Thresholding value chose based on heuritics
        ret, thresh = cv2.threshold(img_blur, 90, 255, cv2.THRESH_BINARY) 
        template = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        #converts b and r channels in the template image to 0 vectors 
        template[:, :,[0,2]] = 0

        plt.imshow(template)

        return template

    def loadPath(self, path, checker):
        """Loads the image if checker is equal to 1
        GS edits: why we need this function?
        argvs:

        returns:

        """

        os.chdir(path)
        Nfile = int((len(os.listdir()))/2) + 1 
        temp_data = np.zeros((Nfile, 512, 512, 3))
        idx = 1

        for file in os.listdir():

            if file.endswith(".jpg"):
                img_temp = cv2.imread(file)
                gray_img = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

                if checker == 1:
                    img_temp = self.thresholding(img_temp,gray_img)

                temp_data [idx, :,:] = img_temp
                idx = idx + 1
        
        print("temp_data file shape .f{}".format(temp_data.shape))

        return temp_data

    def datapreparation(self, data , scale_img = 1/4 , scale_x = 0.05688, scale_y= 0.1, scale_z = 0.01459 ):
        """Modifies the data from pixel scale to mm scale by using experimentally determined scaling factors
        Right hand axis rule for determining X Y Z axis
        argvs:
            scale_img = 1/4 , downsampling the image
            scale_x = 0.05688 mm , width per pixel of full width OCT scan
            scale_y = 0.01 mm ,  total distance moved by the OCT laser while taking a volumetric scan divided by total number of images captured
                for total length = 12.8mm, and 128 scans , scale_y = 12.8/128
            scale_z = 0.01459 mm , depth per pixel of full width OCT scan
        returns:
            output:
        """
        
        df_oct = pd.DataFrame()

        for i in range(data.shape[0]):
            img_temp = data[i,:,:]

            mask = np.zeros((512, 512,3))

            # Remove points above pixel height h. h = 76 for this case
            img_temp[0:76,0:512] = mask[0:76,0:512]

            img_temp = cv2.resize(img_temp, None, fx = scale_img, fy = scale_img)         #resize image
            gray = cv2.cvtColor(img_temp.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            if i == 1:
                print("img {}".format(img_temp.shape))
                #plt.imshow(img_temp)
            
            #exract r,g,b channel 2d matricies  
            r = img_temp[:,:,0]
            g = img_temp[:,:,1]
            b = img_temp[:,:,2]

            #create a linear index array that is the length of the total image pixels 
            idx_final = list(range(0,gray.shape[0] * gray.shape[0])) 

            #creates a row, col index for the each pixel in the image that would allot it in a matrix of img_temp.shape
            [row , col] = np.unravel_index(idx_final,gray.shape,'F')

            if i == 1:
                print("col {}".format(col.shape))
                
            x_temp = row /scale_img *scale_x
            z_temp = -1*col/scale_img*scale_z
            y_temp = -np.ones(x_temp.shape)*i*scale_y

            #TODO Error could be here in the way we are defining the intensity. Shape of gray.flatten() is out of bound of image
            pcd_data = [x_temp, y_temp, z_temp, r.flatten(), g.flatten(), b.flatten(), gray.flatten()] # x, y, z  r,g,b, intensity values
             
            pcd_data = np.asarray(pcd_data) 
            pcd_data = pcd_data.transpose() # taking transpose
            df_temp = pd.DataFrame(pcd_data) # convert to a 7 channel dataframe. Each channel is idx_final long

            df_oct = pd.concat([df_oct,df_temp])

        return df_oct

    def dfConvert (self,df_oct, check):
        """Extracts the XYZ and RGB and Intensity value from the dataframe of pointcloud
        """

        # read all rows in columns 0,1,2 (xyz)
        df_xyz= df_oct.iloc[:, 0:3]

        # read all rows in column 3,4,5 (r,g,b)
        df_rgb = df_oct.iloc[:,3:6]

        # read all rows in column 6 (intensity)
        df_intensity = df_oct.iloc[:, 6]
        
        if check == 1:

            # mask out all parts of the dataframe that have 0 intensity
            mask = df_intensity != 0

            # apply mask to the position and label(rgb) dataframes
            p = df_xyz[mask]
            l = df_rgb[mask]

            return p , mask , l

        return df_xyz, df_intensity, df_rgb    

    def testPointCloud (self):

        path_dir= r"/Users/rp/Library/CloudStorage/Box-Box/Home Folder rp247/Private/1. Documents_220111/Duke/1. BTL/1b. Photoablation_Acoustics/PAA_code_data/data"
       

        path = "./angle_test/angle_test/220922/2/"
        os.chdir(path_dir)
        cwd = os.getcwd()
        print(cwd)

        data = self.loadPath (path,1)

        df_oct = self.datapreparation(data)

        df_xyz, df_intensity, df_rgb = self.dfConvert(df_oct,1)

        #saving point cloud
        np.save("pc_pts.npy",df_xyz)
        np.save("pc_rgb.npy",df_rgb)
        np.save("pc_gray_intensity.npy",df_intensity)

        df_xyz = df_xyz.to_numpy()
        df_rgb = df_rgb.to_numpy()
        df_intensity = df_intensity.to_numpy()

        print("pc_rgb shape = ", df_rgb.shape)
        print("pc_intesity shape = ", df_intensity.shape)
        print("pc_pts shape = ", df_xyz.shape)

        return df_xyz, df_intensity, df_rgb

      

if __name__ == "__main__" :

    pc = pointcloud()
    vtk_pc = module_visualization.visualization_library()

    #test 1
    df_xyz, df_intensity, df_rgb = pc.testPointCloud()

    vtk_pc.colorized_pointcloud_show(points_3d_npy=df_xyz , vector_color_npy= df_rgb)




        

# %%
