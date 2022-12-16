
#%% 
# VTK visualization code

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import cv2
import time
import vtk
import random
import scipy.io
import sys
import scipy.io
from scipy import interpolate

class visualization_library():

    def __init__(self):
        """define the global data paths
        """
        a = 1

    def colorized_pointcloud_show(self, points_3d_npy, vector_color_npy):
        '''visualize colorized point cloud
        argv:
            points_3d_npy: 3D numpy array to describe the surface points
            vector_color_npy: colorized vector associated with 3D colorized point cloud.

        return:
            vtk visualization scene (interactive GUI).
        '''

        global ren, renWin, iren, Rotating, Panning, Zooming
        Rotating = 0
        Panning = 0
        Zooming = 0

        # get colorized point cloud
        actor_color_pointcloud = self.actor_color_pointcloud(points_3d_npy, vector_color_npy)

        # Set the camera object in unit: mm
        pts_cen = np.mean(points_3d_npy, axis=0) # mean along the row
        dis_cen = 0.15
        self.cam = vtk.vtkCamera()
        self.cam.SetPosition(pts_cen[0] + dis_cen, pts_cen[1] + dis_cen, pts_cen[2] + dis_cen)
        self.cam.SetFocalPoint(pts_cen[0], pts_cen[1], pts_cen[2])

        # Renderer
        ren = vtk.vtkRenderer()
        ren.SetBackground(.2, .3, .4)
        ren.ResetCamera()
        ren.SetActiveCamera(self.cam)
        ren.AddActor(actor_color_pointcloud)

        # Render window
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)

        # Interactor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetInteractorStyle(None)
        iren.SetRenderWindow(renWin)
        iren.AddObserver("LeftButtonPressEvent", ButtonEvent)
        iren.AddObserver("LeftButtonReleaseEvent", ButtonEvent)
        iren.AddObserver("MiddleButtonPressEvent", ButtonEvent)
        iren.AddObserver("MiddleButtonReleaseEvent", ButtonEvent)
        iren.AddObserver("RightButtonPressEvent", ButtonEvent)
        iren.AddObserver("RightButtonReleaseEvent", ButtonEvent)
        iren.AddObserver("MouseMoveEvent", MouseMove)
        iren.AddObserver("KeyPressEvent", Keypress)

        iren.Initialize()
        renWin.Render()
        iren.Start()

    def actor_color_pointcloud(self, npy_data, vec_color):

        # from npy to array
        x = npy_data[:, 0]
        y = npy_data[:, 1]
        z = npy_data[:, 2]

        # converting 1 channel numpy color vector to 3 by padding with zeros

        color_pad  = np.zeros((vec_color.shape[0],2))
       # color_pad  = np.ones((vec_color.shape[0],2))
        vec_color = np.append(vec_color,color_pad,axis = 1)


        # Set up the point and vertices
        Points = vtk.vtkPoints()
        Vertices = vtk.vtkCellArray()

        # Set up the color objects
        Colors = vtk.vtkUnsignedCharArray()
        Colors.SetNumberOfComponents(3) # 3 for 3 channel
        Colors.SetName("Colors")
        length = int(len(x))

        # Set up the point and vertice object
        for i in range(length):
            p_x = x[i]
            p_y = y[i]
            p_z = z[i]
            id = Points.InsertNextPoint(p_x, p_y, p_z)
            Vertices.InsertNextCell(1)
            Vertices.InsertCellPoint(id)
            #test
           
            if len(vec_color) > 3: # a color vector
                Colors.InsertNextTuple3(vec_color[i][0], vec_color[i][1], vec_color[i][2])
            else:
                Colors.InsertNextTuple3(vec_color[0], vec_color[1], vec_color[2])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(Points)
        polydata.SetVerts(Vertices)
        polydata.GetPointData().SetScalars(Colors)  # Set the color points for the problem
        polydata.Modified()

        # Set up the actor and mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

'''
Define the rendering objects 
'''
# Handle the mouse button events.
def ButtonEvent(obj, event):
    global Rotating, Panning, Zooming
    if event == "LeftButtonPressEvent":
        Rotating = 1
    elif event == "LeftButtonReleaseEvent":
        Rotating = 0
    elif event == "MiddleButtonPressEvent":
        Panning = 1
    elif event == "MiddleButtonReleaseEvent":
        Panning = 0
    elif event == "RightButtonPressEvent":
        Zooming = 1
    elif event == "RightButtonReleaseEvent":
        Zooming = 0

# General high-level logic
def MouseMove(obj, event):

    global Rotating, Panning, Zooming
    global iren, renWin, ren
    lastXYpos = iren.GetLastEventPosition()
    lastX = lastXYpos[0]
    lastY = lastXYpos[1]

    xypos = iren.GetEventPosition()
    x = xypos[0]
    y = xypos[1]

    center = renWin.GetSize()
    centerX = center[0]/2.0
    centerY = center[1]/2.0

    if Rotating:
        Rotate(ren, ren.GetActiveCamera(), x, y, lastX, lastY,
               centerX, centerY)
    elif Panning:
        Pan(ren, ren.GetActiveCamera(), x, y, lastX, lastY, centerX,
            centerY)
    elif Zooming:
        Dolly(ren, ren.GetActiveCamera(), x, y, lastX, lastY,
              centerX, centerY)

def Keypress(obj, event):
    key = obj.GetKeySym()
    if key == "e":
        obj.InvokeEvent("DeleteAllObjects")
        sys.exit()
    elif key == "w":
        Wireframe()
    elif key =="s":
        Surface()

def Rotate(renderer, camera, x, y, lastX, lastY, centerX, centerY):
    camera.Azimuth(lastX-x)
    camera.Elevation(lastY-y)
    camera.OrthogonalizeViewUp()
    renWin.Render()

# Pan translates x-y motion into translation of the focal point and
# position.
def Pan(renderer, camera, x, y, lastX, lastY, centerX, centerY):
    FPoint = camera.GetFocalPoint()
    FPoint0 = FPoint[0]
    FPoint1 = FPoint[1]
    FPoint2 = FPoint[2]

    PPoint = camera.GetPosition()
    PPoint0 = PPoint[0]
    PPoint1 = PPoint[1]
    PPoint2 = PPoint[2]

    renderer.SetWorldPoint(FPoint0, FPoint1, FPoint2, 1.0)
    renderer.WorldToDisplay()
    DPoint = renderer.GetDisplayPoint()
    focalDepth = DPoint[2]

    APoint0 = centerX+(x-lastX)
    APoint1 = centerY+(y-lastY)

    renderer.SetDisplayPoint(APoint0, APoint1, focalDepth)
    renderer.DisplayToWorld()
    RPoint = renderer.GetWorldPoint()
    RPoint0 = RPoint[0]
    RPoint1 = RPoint[1]
    RPoint2 = RPoint[2]
    RPoint3 = RPoint[3]

    if RPoint3 != 0.0:
        RPoint0 = RPoint0/RPoint3
        RPoint1 = RPoint1/RPoint3
        RPoint2 = RPoint2/RPoint3

    camera.SetFocalPoint( (FPoint0-RPoint0)/2.0 + FPoint0,
                          (FPoint1-RPoint1)/2.0 + FPoint1,
                          (FPoint2-RPoint2)/2.0 + FPoint2)
    camera.SetPosition( (FPoint0-RPoint0)/2.0 + PPoint0,
                        (FPoint1-RPoint1)/2.0 + PPoint1,
                        (FPoint2-RPoint2)/2.0 + PPoint2)
    renWin.Render()

# Dolly converts y-motion into a camera dolly commands.
def Dolly(renderer, camera, x, y, lastX, lastY, centerX, centerY):
    dollyFactor = pow(1.02,(0.5*(y-lastY)))
    if camera.GetParallelProjection():
        parallelScale = camera.GetParallelScale()*dollyFactor
        camera.SetParallelScale(parallelScale)
    else:
        camera.Dolly(dollyFactor)
        renderer.ResetCameraClippingRange()

    renWin.Render()

# Wireframe sets the representation of all actors to wireframe.
def Wireframe():
    actors = ren.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextItem()
    while actor:
        actor.GetProperty().SetRepresentationToWireframe()
        actor = actors.GetNextItem()

    renWin.Render()

# Surface sets the representation of all actors to surface.
def Surface():
    actors = ren.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextItem()
    while actor:
        actor.GetProperty().SetRepresentationToSurface()
        actor = actors.GetNextItem()
    renWin.Render()

def Bracket2npy(vtx):
    npy_data = np.zeros((len(vtx), 3))
    for i in range(len(vtx)):
        npy_data[i, 0] = vtx[i][0]
        npy_data[i, 1] = vtx[i][1]
        npy_data[i, 2] = vtx[i][2]

    return npy_data

if __name__ == "__main__":

    # guangshen
    class_obj = visualization_library()
    path = "C:/1. Documents_220111/Duke/1. BTL/1a.Registration/Code/Stage/ControlFiles/"
    # Test fast point cloud visualization
    pts_pointcloud = np.load(path+ "pc_pts.npy")
    vec_color = np.load(path + "pc_color.npy")
    print("vec_color shape = ", vec_color.shape)
    print("pts_pointcloud shape = ", pts_pointcloud.shape)
    class_obj.colorized_pointcloud_show(points_3d_npy=pts_pointcloud, vector_color_npy=vec_color)

# %%
