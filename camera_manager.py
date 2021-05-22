#!/usr/bin/env python

from __future__ import print_function

import glob
import sys

try:
    sys.path.append(glob.glob('.\\Windows\\CARLA_0.9.11\\PythonAPI\\carla\\dist\\carla-0.9.11-py3.7-win-amd64.egg')[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

import weakref
import numpy as np
import math

# ==============================================================================
# -- Global variables ----------------------------------------------------------
# ==============================================================================

HEIGHT = 280
WIDTH = 400

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction=2.2):
        self.sen = []
        self.sensor = None
        self._parent = parent_actor
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), carla.AttachmentType.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=3.0), carla.Rotation(pitch=-10.0)), carla.AttachmentType.Rigid),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), carla.AttachmentType.SpringArm)]
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(WIDTH))
                bp.set_attribute('image_size_y', str(HEIGHT))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            item.append(bp)
        self.index = None
        self.transform_index = None
        self.main_image = None
        self.cam_image = None
        self.dep_image = None
        self.seg_image = None
        self.x = None

        #####  Camera Settings  #####
        self.fov = math.pi/4
        self.f = WIDTH/(2 * math.tan(self.fov))
        self.c_u = WIDTH/2
        #############################

    def x_from_depth(self):
        """
        Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.
        """
        depth = self.dep_image * 1000
        H, W = np.shape(depth)
        X = np.zeros((H, W))
        # Compute x and y coordinates
        for i in range(H):
            for j in range(W):
                X[i, j] = ((j+1 - self.c_u)*depth[i, j]) / self.f

        return X

    def set_sensor(self, index, pos_index, notify=True):
        self.transform_index = pos_index
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensors[index][-1],
            self._camera_transforms[pos_index][0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[pos_index][1])
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image, index, pos_index))
        self.sen.append(self.sensor)
        self.index = index

    @staticmethod
    def _parse_image(weak_self, image, sen_index, pos_index):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            return
        else:
            image.convert(self.sensors[sen_index][1]) # (280, 400, 4)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            if sen_index == 2:
                # segmentation image
                array = array[:, :, :3]
                seg_array = np.dot(array[:, :, :3], [1, 1, 1])
                self.seg_image = seg_array
            elif sen_index == 1:
                # depth image
                array = array.astype(np.float32)
                # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
                dep_array = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
                dep_array /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
                # global dep_image
                self.dep_image = dep_array
                self.x = self.x_from_depth()
            elif sen_index == 0:
                # rgb image
                array = array[:, :, :3]
                if pos_index == 0 :
                    # global main_image
                    self.main_image = array
                elif pos_index == 1 :
                    # global cam_image
                    self.cam_image = array