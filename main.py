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

import argparse
import random
import math
import weakref
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import threading

# ==============================================================================
# -- Other Helper Classes ------------------------------------------------------
# ==============================================================================

from world import World
from pid_control import VehiclePIDController
from DWA import DWA, Config
from lane import Lane, Lanes_Image
from object_detection import ObjectDetection

# ==============================================================================
# -- Global variables ----------------------------------------------------------
# ==============================================================================

initial_state = {"x": -7.530000, "y": 121.209999, 
                "z": 0.500000, "pitch": 0.000000, 
                "yaw": 89.9999, "roll": 0.000000}

HEIGHT = 280
WIDTH = 400
world = None
object_detection = None
t1 = None

# ==============================================================================
# -- transform image lanes into real x-y lane  ---------------------------------
# ==============================================================================

def transform(point, state):
    x = point[0]
    y = point[1]
    r = math.sqrt((x*x) + (y*y))
    theta = math.atan2(y, x) - (state.rotation.yaw - 90.0)*(math.pi/180.0)
    x = r*math.cos(theta) + state.location.x
    y = r*math.sin(theta) + state.location.y
    return [x, y]

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    fig = plt.figure()
    goal = np.array([-7.530000, 321.210205])
    # plt.plot(goal[0], goal[1], 'ro')
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        global world
        world = World(client.get_world(), args)
        # world.vehicle.set_autopilot(True)
        global object_detection
        object_detection = ObjectDetection(world)
        global t1
        t1 = threading.Thread(target=object_detection.detect, name='t1')
        t1.start()

        # Path Planning part
        controller = VehiclePIDController(
            world.vehicle,
            args_lateral={'K_P':1.0,'K_D':0.07,'K_I':0.02},
            args_longitudinal={'K_P':1.0,'K_D':0.09, 'K_I':0.2},
        )
        dwa = DWA(goal)
        config = Config()
        local_state = np.array([initial_state['x'], initial_state['y'], (90) * math.pi / 180.0, 0.0, 0.0])

        # plt.xlabel('Y Direction')
        # plt.ylabel('X Direction')

        # Simulation Starts
        curr_lanes = []
        objects = []
        while True:
            if world.camera_manager.cam_image is not None:
                cv2.imshow('camera image', world.camera_manager.cam_image)

            state = world.vehicle.get_transform()
            local_state[0] = state.location.x
            local_state[1] = state.location.y
            local_state[2] = (state.rotation.yaw) * math.pi / 180.0

            # Lane detection
            if world.camera_manager.seg_image is not None:
                image_lanes = Lanes_Image(world.camera_manager.seg_image)
                lines = image_lanes.get_lines()
                
                if world.camera_manager.dep_image is not None:
                    z = world.camera_manager.dep_image*1000
                    x = world.camera_manager.x
                    res = np.zeros((HEIGHT, WIDTH))
                    for line in lines:
                        x1, y1, x2, y2 = line.astype(int)

                        if z[y1][x1] >= 300 or z[y2][x2]>=300:
                            continue

                        p1 = transform([x[y1][x1], z[y1][x1]], state)
                        p2 = transform([x[y2][x2], z[y2][x2]], state)
                        cv2.line(res, (x1, y1), (x2, y2), (255,0,0), 2)

                        # Filtering lanes
                        lane = Lane(p1[0], p1[1], p2[0], p2[1])
                        if lane.type != 'N':
                            idx = lane.find_similar_lane(curr_lanes)
                            if idx == -1:
                                # no similar line
                                curr_lanes.append(lane)
                            else:
                                # similar with idx line
                                curr_lanes[idx] = lane.merge_lane(curr_lanes[idx])

                    cv2.imshow('result', res)

                    # plt.clf()
                    for obj_cord in object_detection.coordinates:
                        # print(obj_cord[0], obj_cord[1])
                        p = transform([obj_cord[0], obj_cord[1]], state)
                        objects.append(p)

                    print("Current lanes are : {}".format(len(curr_lanes)))
                    print("Current object are : {}".format(len(objects)))

                    # Drawing object and lanes
                    # for lane in curr_lanes:
                    #     if lane.weight > 2:
                    #         # print(lane.x1, lane.x2, lane.y1, lane.y2)
                    #         plt.plot(state.location.x, state.location.y, 'ro')
                    #         plt.plot([lane.x1, lane.x2], [lane.y1, lane.y2])
                    # for obj_cord in objects:
                    #     plt.plot(obj_cord[0], obj_cord[1], 'ro')

            u, predicted_trajectory = dwa.dwa_control(local_state, np.array([]), [])
            px = local_state[0]
            py = local_state[1]
            local_state = dwa.motion(local_state, u, config.dt)  # simulate robot

            # Ploting trajactory of car-robot
            # plt.plot([local_state[0], px], [local_state[1], py])
            destination = carla.Location(
                x=local_state[0], 
                y=local_state[1], 
                z=state.location.z,
            )
            
            # move vehicle
            waypoint = world.world.get_map().get_waypoint(destination)
            control_signal = controller.run_step(local_state[3], waypoint)
            world.vehicle.apply_control(control_signal)
            # time.sleep(config.dt/2)

            # edge case to reach at goal
            dist_to_goal = math.hypot(local_state[0] - goal[0], local_state[1] - goal[1])
            if dist_to_goal <= config.robot_radius:
                print("Goal Reached!!")
                break

            # debug
            # cv2.waitKey(1)
            # plt.pause(0.05)
            # print(self._vehicle.get_location().y - initial_state['y'], self._state[1])
            # break
    finally:
        # plt.show()
        if world is not None:
            print("Destroying World")
            world.destroy()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='Amit',
        help='actor role name (default: "Amit")')
    args = argparser.parse_args()
    print('[+] Listening to server {}:{}'.format(args.host, args.port))
    try:
        game_loop(args)
    except KeyboardInterrupt:
        # plt.show()
        global world, object_detection, t1
        object_detection.shutdown = True
        t1.join()
        world.destroy()

        print('\nCarla is closing...')
        
if __name__ == '__main__':
    main()