
"""
Owner: Ravi Prakash 
experiment: 2-DoF stage raster scanning of phantom mounted
NOTE: Please do not modify this before checking with the owner

TODO: synchronize the code with the functions
  
"""

import sys
import os

import module_robot_stage
import thorlabs_apt

# from System import *
# import os, sys, clr
# import pyvisa

import time
import numpy as np
import matplotlib.pyplot as plt
import module_oct_serial

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

class raster_scan:

    def __init__(self) :
        
        # intializes the tumorID device and thorlabs stage
        print("initialize the sensor modules")

        # robot stage
        input("Press Enter to move start home process of stage ")
        self.stage = module_robot_stage.controlstage()

        # oct stage
        self.oct = module_oct_serial.octserial()

        # tumorid module
        # self.tumorid.clear_error()
        # self.tumorid = tumorID()

    def RobotStageTest(self):
        """single step mannual robot stage testing
        """
        # test movement of stage
        while(True):
            x_pos_stage = int(input("enter x position (0 to 300 mm)"))
            y_pos_stage = int(input("enter y position (0 to 300 mm)"))

            # For phantom scaning, X axis stage hardstop at 270mm    
            if x_pos_stage >= 270 or y_pos_stage >= 270:
                print("in correct position to the maximum range, exit")
                exit()

            # print("input x_pos_stage = ", x_pos_stage)
            # print("input y_pos_stage = ", y_pos_stage)
            self.stage.move_to(x_pos_stage, y_pos_stage)
            self.stage.get_position()

    def TumorIDDeviceTest(self):
        """test the tumorid with the robot stage
        argvs:
            input x-y position values
        returns:
            stage move to the target position
        """

        # test movement of stage
        x_pos_stage = int(input("enter x position"))
        y_pos_stage = int(input("enter y position"))
        self.stage.move_to(x_pos_stage, y_pos_stage)

        # test working of tumorID 
        integration_time = .5
        wavelength_signal = self.tumorid.preCapture(integration_time)
        self.tumorid.turn_laser_ON(250/1000) # miliampere
        spectrum_signal = self.tumorid.capture()
        self.tumorid.turn_laser_OFF()

        # show the data
        plt.plot(wavelength_signal, spectrum_signal)

    def StageRasterInROI(self, x_pos_stage, y_pos_stage):
        """acutal raster scanning with the stage and the TumorID
        within a Region-of-interest
        """

        # basic setting
        path_of_tumorid_data = r"./data_test/data_raster_scan"

        # tumorID setup
        tumorID_current = 180
        integration_time = 0.5
        wavelength_signal = self.tumorid.preCapture(integration_time)

        # TODO: explain the offset_x_stage and offset_y_stage
        # stage setup
        offset_x_stage = 11.2
        offset_y_stage = 11.2
        x_initial_stage = x_pos_stage
        y_initial_stage = y_pos_stage
        x_final_pos = x_initial_stage + offset_x_stage
        y_final_pos = y_initial_stage + offset_y_stage

        # test the stage with random positions
        # set the x position
        idx_x_pos = x_initial_stage
        while idx_x_pos < x_final_pos:

            # set the y position
            idx_y_pos = y_initial_stage

            while idx_y_pos < y_final_pos:

                # move to the target position
                self.stage.move_to(idx_x_pos, idx_y_pos)
                time.sleep(10)

                # turning on the laser and capturing the spot
                self.tumorid.turn_laser_ON(tumorID_current / 1000)

                # turning on the camera and capturing an image
                # TODO: add the camera module herein.
                # TODO: add the laser spot detection algorithm

                # process the tumorid data
                spectrum_signal = self.tumorid.capture()
                self.tumorid.turn_laser_OFF()
                filename = "point (x,y) {},{}".format(idx_x_pos, idx_y_pos)
                self.tumorid.save(spectrum_signal, wavelength_signal, path_of_tumorid_data, filename, 1)

                idx_y_pos = idx_y_pos + 1
            
            idx_x_pos = idx_x_pos + 1
            print("x and y value {}, {}".format(idx_x_pos, idx_y_pos))
            time.sleep(15)

    def OCTRasterInROI(self, x_init_pos_stage, y_init_pos_stage, x_final_pos_stage, y_final_pos_stage):
        """acutal raster scanning with the stage and the OCT within a Region-of-interest
        one stage-step -> one oct scan
        """

        # global setting
        range_max_stage_x = 270
        range_max_stage_y = 140

        # Offset of 10 mm with 1.2 mm overlap window as OCT scan width during volumetric scan is 11.2 mm
        x_stage_step = 10
        y_stage_step = 10
        x_initial_stage = x_init_pos_stage
        y_initial_stage = y_init_pos_stage
        x_final_pos = x_final_pos_stage
        y_final_pos = y_final_pos_stage

        # test the stage with random positions + set the x position
        idx_x_pos = x_initial_stage
        input("Press Enter to Start")
        counter_stage = 0
        counter_scan = 0
        while idx_x_pos <= x_final_pos:

            # set the y position
            idx_y_pos = y_initial_stage
            while idx_y_pos <= y_final_pos:

                # move to the target position
                self.stage.move_to(idx_x_pos, idx_y_pos)
                counter_stage = counter_stage +1
                counter_scan = counter_scan + 1
                if counter_stage == 1:
                    input("Press Enter to Start Once Position Reached")
                file_name = " xy_"+str(idx_x_pos)+"_"+str(idx_y_pos)
                print("Time to scan and wait")
                #time.sleep(30)

                
                oct_token = self.oct.octvolscan(file_name)

                if oct_token:
                    print("Scan successfull")
                else:
                    print("Scan failed")

                idx_y_pos = idx_y_pos + y_stage_step
                if idx_y_pos >= range_max_stage_y:
                    print("out of bound y value, exiting")
                    exit()

                # Press Enter for next step
                #input("Press Enter for Next Step")
            
            idx_x_pos = idx_x_pos + x_stage_step
            # move to the start y of new row position
            self.stage.move_to(idx_x_pos,y_init_pos_stage)
            #wait for it to reach the start position
            time.sleep(30)
            print("x and y value {}, {}".format(idx_x_pos, idx_y_pos))

            if idx_x_pos >= range_max_stage_x:
                print("out of bound x value, exiting")
                exit()
           # time.sleep(15)

    def OCTPhantomScan(self, flag_mode = "single"):
        """defines the start and stop location of stage for scanning over phantom mounted on OCT
        """

        if flag_mode == "oct":
            input("Press Enter to remote oct serial port")
            file_name = " xy_" + str(voltage_x_pos) + "_" + str(voltage_y_pos)
            print("start remote oct scanning")
            oct_token = self.oct.octvolscan(file_name)
            if oct_token:
                print("Scan successfull")
            else:
                print("Scan failed")

        if flag_mode == "roi":
            """stage-oct raster scanning in a roi
            """
            # unit: mm
            x_init_pos_stage = 190
            y_init_pos_stage = 0
            x_final_pos_stage = 250
            y_final_pos_stage = 120
            print("check the current ROI")
            self.OCTRasterInROI(x_init_pos_stage, y_init_pos_stage, x_final_pos_stage, y_final_pos_stage)
            print("Scan Complete")

        if flag_mode == "single":
            """single point oct scanning
            """
            # move to the target position
            voltage_x_pos = 200
            voltage_y_pos = 100
            input("check the input voltage for the current oct scan")
            self.stage.move_to(voltage_x_pos, voltage_y_pos)
            input("Press Enter to remote oct serial port")
            file_name = " xy_" + str(voltage_x_pos) + "_" + str(voltage_y_pos)
            print("start remote oct scanning")
            oct_token = self.oct.octvolscan(file_name)
            if oct_token:
                print("Scan successfull")
            else:
                print("Scan failed")

    def StageRasterSimultion(self, list_of_x_voltages = [], list_of_y_voltages = [] ):
        """simulate the raster scanning before running the study.
        argvs:
            list_of_x_voltages: N x 1
            list_of_y_voltages: N x 1
            3D coordinate axis:
        returns:
            simulation visualizatoin in the {world} glboal coordinate system
        """
        # inputs
        if len(list_of_x_voltages) == 0 and len(list_of_y_voltages) == 0:
            list_of_inputs = 0.5 * np.random.randn(10, 2) + 1.0
            list_of_x_voltages = np.transpose(np.asarray([list_of_inputs[:,0]]))
            list_of_y_voltages = np.transpose(np.asarray([list_of_inputs[:,1]]))

        # define the convertion ratio
        ratio_voltage_to_mm = 1.0

        # define the original 3D position
        x_pos_origin = 0.0 
        y_pos_origin = 0.0

        # define the X-axis vector
        # define the Y-axis vector
        vec_x_direction = np.array([1.0, 0.0, 0.0])
        vec_y_direction = np.array([0.0, 1.0, 0.0])

        # return the 3D coordinates
        list_of_x_pos_mm = list_of_x_voltages * ratio_voltage_to_mm
        list_of_y_pos_mm = list_of_y_voltages * ratio_voltage_to_mm
        list_of_z_pos_mm = np.zeros(list_of_x_pos_mm.shape)
        list_of_xy_unit_3d = np.hstack([list_of_x_pos_mm, list_of_y_pos_mm, list_of_z_pos_mm])
        print("list_of_xy_unit_3d = ", list_of_xy_unit_3d.shape)

        # vis the programs
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(list_of_xy_unit_3d[:, 0], list_of_xy_unit_3d[:, 1], list_of_xy_unit_3d[:, 2], c='b', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim(0, 350)
        ax.set_ylim(0, 350)
        ax.set_zlim(0, 1)

        # show the 3D frame
        len_frame_scale = 0.5
        arrow_prop_dict = dict(mutation_scale = 10, arrowstyle='->', shrinkA=0, shrinkB=0)
        a = Arrow3D([0, len_frame_scale], [0, 0], [0, 0], **arrow_prop_dict, color='r')
        ax.add_artist(a)
        a = Arrow3D([0, 0], [0, len_frame_scale], [0, 0], **arrow_prop_dict, color='b')
        ax.add_artist(a)
        a = Arrow3D([0, 0], [0, 0], [0, len_frame_scale], **arrow_prop_dict, color='g')
        ax.add_artist(a)

        plt.show()

        return list_of_xy_unit_3d

if __name__ == "__main__":

    # test-1: stage movements
    class_test = raster_scan()

    # manual input stage coordinates
    class_test.RobotStageTest()

    # simulation
    # class_test.StageRasterSimultion()

    # test-2: OCT phantom scan
    # class_test.OCTPhantomScan(flag_mode="single")