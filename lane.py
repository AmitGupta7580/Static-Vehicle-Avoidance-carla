#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2
import math

# ==============================================================================
# -- Global variables ----------------------------------------------------------
# ==============================================================================

HEIGHT = 280
WIDTH = 400

# ==============================================================================
# -- Lanes ---------------------------------------------------------------------
# ==============================================================================

class Lane:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.slope = 0
        self.intercept = 0
        self.theta = 0
        self.type = None
        self.weight = 1
        self.calculate_slope()

        ####  Merging Parameters  ####
        self.intercept_threshold = 2
        self.similarity_threshold = 4
        ##############################

    def calculate_slope(self):
        if self.x1 == self.x2:
            # vertical lane
            self.slope = float("Inf")
            self.intercept = self.x1
            self.type = 'V'
        else:
            self.slope = (self.y1 - self.y2)/(self.x1 - self.x2)
        self.theta = math.atan(self.slope)
        if self.theta > (math.pi)/3 or self.theta < -(math.pi)/3:
            # vertical
            self.intercept = (self.x1 + self.x2)/2
            self.type = 'V'
            self.x1 = self.intercept
            self.x2 = self.intercept
        elif self.theta < (math.pi)/6 and self.theta > -(math.pi)/6:
            # horizontal
            self.intercept = (self.y1 + self.y2)/2
            self.type = 'H'
            self.y1 = self.intercept
            self.y2 = self.intercept
        else:
            # not countable line
            self.type = 'N'

    def find_similar_lane(self, lanes):
        for idx in range(len(lanes)):
            lane = lanes[idx]
            if lane.type == self.type: # of same slope
                if abs(lane.intercept - self.intercept) < self.intercept_threshold: # lie under same intercept threshold
                    if self.type == 'V':
                        if (min(self.y1, self.y2) <= max(lane.y1, lane.y2) and min(self.y1, self.y2) >= min(lane.y1, lane.y2)) or \
                            (max(self.y1, self.y2) <= max(lane.y1, lane.y2) and max(self.y1, self.y2) >= min(lane.y1, lane.y2)):
                            cords = [self.y1, self.y2, lane.y1, lane.y2]
                            cords.sort()
                            similarity = cords[2]-cords[1]
                            if similarity > self.similarity_threshold:
                                return idx
                    elif self.type == 'H':
                        if (min(self.x1, self.x2) <= max(lane.x1, lane.x2) and min(self.x1, self.x2) >= min(lane.x1, lane.x2)) or \
                            (max(self.x1, self.x2) <= max(lane.x1, lane.x2) and max(self.x1, self.x2) >= min(lane.x1, lane.x2)):
                            cords = [self.x1, self.x2, lane.x1, lane.x2]
                            cords.sort()
                            similarity = cords[2]-cords[1]
                            if similarity > self.similarity_threshold:
                                return idx
        return -1

    def merge_lane(self, lane):
        self.intercept = (self.weight*self.intercept + lane.weight*lane.intercept)/(self.weight + lane.weight)
        self.weight = self.weight + lane.weight
        if self.type == 'V':
            y_1 = min(min(self.y1, self.y2), min(lane.y1, lane.y2))
            y_2 = max(max(self.y1, self.y2), max(lane.y1, lane.y2))
            self.y1 = y_1
            self.y2 = y_2

        elif self.type == 'H':
            x_1 = min(min(self.x1, self.x2), min(lane.x1, lane.x2))
            x_2 = max(max(self.x1, self.x2), max(lane.x1, lane.x2))
            self.x1 = x_1
            self.x2 = x_2
            
        return self

class Lanes_Image:
    def __init__(self, seg_image):
        self.seg_image = seg_image

    def get_lines(self):
        # road_mask = np.zeros((HEIGHT, WIDTH))
        # road_mask[world.camera_manager.seg_image == 320] = 1
        # road_mask[0:130,:] = 0
        lane_boundary_mask = np.zeros((HEIGHT, WIDTH)).astype(np.uint8)
        lane_boundary_mask[self.seg_image == 511] = 255
        lane_boundary_mask[0:130,:] = 0
            
        # cv2.imshow('lane_boundary_mask', lane_boundary_mask)
        # cv2.imshow('road_mask', road_mask)

        edges = cv2.Canny(lane_boundary_mask, 0, 150)
        # cv2.imshow('edges', edges)

        lines = cv2.HoughLinesP(edges, rho=5, theta=np.pi/180, threshold=70, minLineLength=70, maxLineGap=80)

        lines = lines.reshape((-1, 4))
        return self.merge_lane_lines(lines)
        
    def get_slope_intecept(self, lines):
        slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0] + 0.001)
        intercepts = ((lines[:, 3] + lines[:, 1]) - slopes * (
        lines[:, 2] + lines[:, 0])) / 2
        return slopes, intercepts

    def merge_lane_lines(self, lines):
        # Step 0: Define thresholds
        slope_similarity_threshold = 0.1
        intercept_similarity_threshold = 40
        min_slope_threshold = 0.3
        clusters = []
        current_inds = []
        itr = 0
    
        # Step 1: Get slope and intercept of lines
        slopes, intercepts = self.get_slope_intecept(lines)
    
        # Step 2: Determine lines with slope less than horizontal slope threshold.
        slopes_horizontal = np.abs(slopes) > min_slope_threshold

        # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
        for slope, intercept in zip(slopes, intercepts):
            in_clusters = np.array([itr in current for current in current_inds])
            if not in_clusters.any():
                slope_cluster = np.logical_and(slopes < (slope+slope_similarity_threshold), slopes > (slope-slope_similarity_threshold))
                intercept_cluster = np.logical_and(intercepts < (intercept+intercept_similarity_threshold), intercepts > (intercept-intercept_similarity_threshold))
                inds = np.argwhere(slope_cluster & intercept_cluster & slopes_horizontal).T
                if inds.size:
                    current_inds.append(inds.flatten())
                    clusters.append(lines[inds])
            itr += 1
        
        # Step 4: Merge all lines in clusters using mean averaging
        merged_lines = [np.mean(cluster, axis=1) for cluster in clusters]
        merged_lines = np.array(merged_lines).reshape((-1, 4))

        return merged_lines