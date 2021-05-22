#!/usr/bin/env python

from __future__ import print_function

import glob
import sys

try:
    sys.path.append(glob.glob('.\\Windows\\CARLA_0.9.11\\PythonAPI\\carla\\dist\\carla-0.9.11-py3.7-win-amd64.egg')[0])
except IndexError:
    pass

import carla
import random

# ==============================================================================
# -- Other Helper Classes ------------------------------------------------------
# ==============================================================================

from camera_manager import CameraManager

# ==============================================================================
# -- Global variables ----------------------------------------------------------
# ==============================================================================

initial_state = {"x": -7.530000, "y": 121.209999, 
                "z": 0.500000, "pitch": 0.000000, 
                "yaw": 89.999954, "roll": 0.000000}

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            sys.exit(1)
        self.vehicle = None
        self.camera_manager = None
        self.players = [None, None]
        self.players_location = [[5, 45, 0], [5, 70, 0]]
        self.restart()

    def restart(self):

        self.world.unload_map_layer(carla.MapLayer.All)
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.a2'))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '224,0,0')
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # Spawn the vehicle.
        while self.vehicle is None :#or self.player2 is None or self.player3 is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_point = carla.Transform(
                carla.Location(x=initial_state["x"], y=initial_state["y"], z=initial_state["z"]), 
                carla.Rotation(yaw=initial_state["yaw"]),
            )
            self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        # spawing objects
        for player_id in range(1):
            while self.players[player_id] is None:
                spawn_point = carla.Transform(
                    carla.Location(
                        x=initial_state["x"] + self.players_location[player_id][0], 
                        y=initial_state["y"] + self.players_location[player_id][1],
                        z=initial_state["z"],
                    ), 
                    carla.Rotation(yaw=initial_state["yaw"] + self.players_location[player_id][2]),
                )
                self.players[player_id] = self.world.try_spawn_actor(blueprint, spawn_point)

        # Set up the sensors.
        self.camera_manager = CameraManager(self.vehicle)
        # self.camera_manager.set_sensor(0, 0, notify=False)    # main camera
        self.camera_manager.set_sensor(0, 1, notify=False)    # front camera
        self.camera_manager.set_sensor(1, 1, notify=False)    # depth camera
        self.camera_manager.set_sensor(2, 1, notify=False)    # segmentation camera
        print("[+] World setup completed")

    def destroy_sensors(self):
        if self.camera_manager is not None:
            sen = self.camera_manager.sen
            for s in sen:
                s.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None

    def destroy(self):
        self.destroy_sensors()
        if self.vehicle is not None:
            self.vehicle.destroy()
        actors = self.players
        for actor in actors:
            if actor is not None:
                actor.destroy()