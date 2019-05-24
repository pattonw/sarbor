import vtk
import numpy as np
from collections import OrderedDict
from vtk.util.colors import lime_green, slate_grey


def octree_to_sparse_vtk_volume(volume, cutoff=0.5, resolution=[1, 1, 1]):
    """
    Generate a vtk volume from the blockwise sparse octree. This is a hacky
    solution that iterates over evey pixel and checks if any of its neighboring
    pixels are on the other side of the cutoff, meaning a surface would have
    to be drawn between them. It then creates a vtk unstructured grid out of
    a list of points and voxels.
    """

    def get_points(volume, i, j, k, points, lower_bound, resolution):
        vertices = [
            (i + 0, j + 0, k + 0),
            (i + 1, j + 0, k + 0),
            (i + 1, j + 1, k + 0),
            (i + 0, j + 1, k + 0),
            (i + 0, j + 0, k + 1),
            (i + 1, j + 0, k + 1),
            (i + 1, j + 1, k + 1),
            (i + 0, j + 1, k + 1),
        ]
        i = len(points)
        indices = []
        for vertex in vertices:
            global_coords = tuple(
                [(vertex[i] + lower_bound[i]) * resolution[i] for i in range(3)]
            )
            point_value, point_index = points.get(global_coords, (None, None))
            if point_index is None:
                points[global_coords] = (volume[vertex], i)
                indices.append(i)
                i += 1
            else:
                indices.append(point_index)
        return indices

    vertices = OrderedDict()
    voxels = []

    leaves = {tuple(leaf.bounds[0]): leaf for leaf in volume.iter_leaves()}
    for lower_bound, leaf in leaves.items():
        # pad the block with zeros
        leaf_data = np.pad(
            leaf.data, [[0, 1], [0, 1], [0, 1]], "constant", constant_values=0
        )
        cases = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]
        for case in cases:
            # fill in padded values from appropriate leaves if those leaves exist
            case_lower_bound = tuple(
                [lower_bound[i] + case[i] * leaf.data.shape[i] for i in range(3)]
            )
            if case_lower_bound in leaves:
                leaf_data[
                    tuple(
                        map(
                            slice,
                            [0 if c == 0 else -1 for c in case],
                            [-1 if c == 0 else None for c in case],
                        )
                    )
                ] = leaves[case_lower_bound].data[
                    tuple(map(slice, [0, 0, 0], [None if c == 0 else 1 for c in case]))
                ]

        if any((leaf_data > cutoff).reshape(-1)):
            # create a numpy array, with 8 copies of itself allong
            # the 0th axis. Each copy represents a shift somewhere between
            # 0,0,0 and 1,1,1. Thus if you take the minimum/maximum along
            # the 0th axis, you get the minimum/maximum of a pixel and its
            # neighbors.
            square = np.array(
                [
                    leaf_data[:-1, :-1, :-1],
                    leaf_data[:-1, :-1, 1:],
                    leaf_data[:-1, 1:, :-1],
                    leaf_data[:-1, 1:, 1:],
                    leaf_data[1:, :-1, :-1],
                    leaf_data[1:, :-1, 1:],
                    leaf_data[1:, 1:, :-1],
                    leaf_data[1:, 1:, 1:],
                ]
            )
            edge_voxels = np.where(
                (square.min(axis=0) < cutoff) * (square.max(axis=0) > cutoff)
            )
            for i, j, k in zip(*edge_voxels):
                indices = get_points(
                    leaf_data, i, j, k, vertices, lower_bound, resolution=resolution
                )
                voxels += [indices]

    volume = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    scalars = vtk.vtkFloatArray()

    def mkVtkIdList(it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    # Load the point, cell, and data attributes.
    for coord, value in vertices.items():
        v, i = value
        points.InsertPoint(i, coord)
        scalars.InsertTuple1(i, v)
    for i in range(len(voxels)):
        volume.InsertNextCell(12, mkVtkIdList(voxels[i]))

    # We now assign the pieces to the vtkPolyData.
    volume.SetPoints(points)
    del points
    volume.GetPointData().SetScalars(scalars)
    del scalars

    return volume


def contour_sparse_vtk_volume(volume, cutoff):
    # Use marching cubes over a volume
    Marching = vtk.vtkContourFilter()
    Marching.SetInputData(volume)
    Marching.SetValue(0, cutoff)
    Marching.Update()
    return Marching


def decimate_mesh(contour, target_reduction=0.1):
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(contour.GetOutput())
    decimate.SetTargetReduction(target_reduction)
    decimate.Update()
    return decimate
  

def write_to_stl(vtk_volume, filename):
    # Write a volume to a .stl file
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(filename)
    stlWriter.SetInputConnection(vtk_volume.GetOutputPort())
    stlWriter.Write()


def read_from_stl(filename):
    # Read a volume from a .stl file
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    return reader


def visualize_mesh(contour, cutoff=0.5):
    # visualize a volume
    aTransform = vtk.vtkTransform()
    aTransform.Scale(1, 1, 1)
    transFilter = vtk.vtkTransformFilter()
    transFilter.SetInputConnection(contour.GetOutputPort())
    transFilter.SetTransform(aTransform)
    aShrinker = vtk.vtkShrinkPolyData()
    aShrinker.SetShrinkFactor(1)
    aShrinker.SetInputConnection(transFilter.GetOutputPort())
    aMapper = vtk.vtkPolyDataMapper()
    aMapper.ScalarVisibilityOff()
    aMapper.SetInputConnection(transFilter.GetOutputPort())
    Triangles = vtk.vtkActor()
    Triangles.SetMapper(aMapper)
    Triangles.GetProperty().SetDiffuseColor(lime_green)
    Triangles.GetProperty().SetOpacity(0.2)
    Triangles.SetScale

    # Create the Renderer, RenderWindow, and RenderWindowInteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(640, 480)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Add the actors to the renderer
    ren.AddActor(Triangles)

    # Set the background color.
    ren.SetBackground(slate_grey)

    # Position the camera.
    ren.ResetCamera()
    ren.GetActiveCamera().Dolly(1.2)
    ren.GetActiveCamera().Azimuth(30)
    ren.GetActiveCamera().Elevation(20)
    ren.ResetCameraClippingRange()

    iren.Initialize()
    renWin.Render()
    iren.Start()
