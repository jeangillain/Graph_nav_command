# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface for graph nav with options to download/upload a map and to navigate a map. """
from bosdyn.geometry import EulerZXY
import argparse
import grpc
import logging
import math
import os
import sys
import time

import math
import numpy as np
from bosdyn.api import geometry_pb2
from bosdyn.api import power_pb2
from bosdyn.api import robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.api.graph_nav import map_pb2
from bosdyn.api.graph_nav import nav_pb2
import bosdyn.client.channel
from bosdyn.client.power import safe_power_off, PowerClient, power_on
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, LeaseWallet, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose,SE2Velocity 
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
import bosdyn.client.util
import google.protobuf.timestamp_pb2
import glob
import graph_nav_util
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_a_tform_b
from bosdyn.api import geometry_pb2
VELOCITY_CMD_DURATION = 0.6  # seconds


class GraphNavInterface(object):
    """GraphNav service command line interface."""

    def __init__(self, robot,robot_1, upload_path):
        self._robot = robot
        self._master_robot = robot
        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create the lease client with keep-alive, then acquire the lease.
        self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)
        self._lease_wallet = self._lease_client.lease_wallet
        try:
            self._lease = self._lease_client.acquire()
        except ResourceAlreadyClaimedError as err:
            print("The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds.")
            os._exit(1)
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
        
        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)
        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()
        self._waypoint_to_timestamp = []
        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        self._command_dictionary = {
            '1': self._get_localization_state,
            '2': self._set_initial_localization_fiducial,
            '3': self._set_initial_localization_waypoint,
            '4': self._list_graph_waypoint_and_edge_ids,
            '5': self._upload_graph_and_snapshots,
            '6': self._navigate_to,
            '7': self._navigate_route,
            '8': self._navigate_to_anchor,
            '9': self._clear_graph,
            '0': self._navigate_all,
            'a' : self._navigate_all_no_rotate,
        }
    def _issue_robot_command(self, command, endtime=None):
        """Check that the lease has been acquired and motors are powered on before issuing a command.

        Args:
            command: RobotCommand message to be sent to the robot.
            endtime: Time (in the local clock) that the robot command should stop.
        """

        self._robot_command_client.robot_command(command, end_time_secs=endtime)
    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        print('Got localization: \n%s' % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print('Got robot state in kinematic odometry frame: \n%s' % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges, self._waypoint_to_timestamp = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id)
            

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print("Loaded graph has {} waypoints and {} edges".format(
                len(self._current_graph.waypoints), len(self._current_graph.edges)))
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(self._upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(self._upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(lease=self._lease.lease_proto,
                                                       graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print("Uploaded {}".format(waypoint_snapshot.id))
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print("Uploaded {}".format(edge_snapshot.id))
        self._waypoint_to_timestamp  = graph_nav_util.sort_waypoints_chrono(self._current_graph)
        # The upload is complete! Check that the robot is localized to the graph,
        # and it if is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print("Upload complete! The robot is currently not localized to the map; please localize", \
                   "the robot using commands (2) or (3) before attempting a navigation command.")
    
    def _navigate_to_anchor(self, *args):
        """Navigate to a pose in seed frame, using anchors."""
        # The following options are accepted for arguments: [x, y], [x, y, yaw], [x, y, z, yaw],
        # [x, y, z, qw, qx, qy, qz].
        # When a value for z is not specified, we use the current z height.
        # When only yaw is specified, the quaternion is constructed from the yaw.
        # When yaw is not specified, an identity quaternion is used.
        graph = self._graph_nav_client.download_graph()#maybe not necessary

        print("what are the args ? ", args)
        if len(args) < 1 or len(args[0]) not in [2, 3, 4, 7]:
            print("Invalid arguments supplied.")
            return
        seed_T_goal = SE3Pose(float(args[0][0]), float(args[0][1]), 0.0, Quat())
        print("seed_T_goal: ", seed_T_goal)
        if len(args[0]) in [4, 7]:
            seed_T_goal.z = float(args[0][2])
        else:
            localization_state = self._graph_nav_client.get_localization_state()
            if not localization_state.localization.waypoint_id:
                print("Robot not localized")
                return
            seed_T_goal.z = localization_state.localization.seed_tform_body.position.z

        if len(args[0]) == 3:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][2]))
        elif len(args[0]) == 4:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][3]))
        elif len(args[0]) == 7:
            seed_T_goal.rot = Quat(w=float(args[0][3]), x=float(args[0][4]), y=float(args[0][5]),
                                   z=float(args[0][6]))
            print(type(seed_T_goal))
        self._lease = self._lease_wallet.get_lease()
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        # Stop the lease keepalive and create a new sublease for graph nav.
        self._lease = self._lease_wallet.advance()
        sublease = self._lease.create_sublease()
        self._lease_keepalive.shutdown()
        nav_to_cmd_id = None
        # Navigate to the destination.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                max_vel_linear = geometry_pb2.Vec2(x=0.1, y=0.1)
                max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                           angular=0.1)
                travel_params = self._graph_nav_client.generate_travel_params(0.0,0.0,geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2))
                nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
                    seed_T_goal.to_proto(), 1.0, leases=[sublease.lease_proto],
                    command_id=nav_to_cmd_id,travel_params=travel_params)
                print(" seed_T_goal proto : ", seed_T_goal.to_proto())
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)
        time.sleep(1)
        self._lease = self._lease_wallet.advance()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

        # Update the lease and power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        print("args navigate to : ", args)
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        self._lease = self._lease_wallet.get_lease()
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        # Stop the lease keep-alive and create a new sublease for graph nav.
        self._lease = self._lease_wallet.advance()
        sublease = self._lease.create_sublease()
        self._lease_keepalive.shutdown()
        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                max_vel_linear = geometry_pb2.Vec2(x=0.5, y=0.5)
                max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                           angular=0.5)
                travel_params = self._graph_nav_client.generate_travel_params(0.0,0.0,geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2))
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   leases=[sublease.lease_proto],
                                                                   command_id=nav_to_cmd_id,travel_params =travel_params)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)
        list_rotation = []
        #list_rotation.append([0,0.0,math.pi/2])
        list_rotation.append([math.pi/2,0.0,0])
        list_rotation.append([0,0,math.pi/2])
        list_rotation.append([0,-math.pi/2,0])
        list_rotation.append([0,math.pi/2,0])
        #list_rotation.append([0,math.pi/2,0])
        #list_rotation.append([0,-math.pi/2,0])
        
        for elem in list_rotation:
            sublease = self._lease.create_sublease()
            print("elem : ", elem)
            orientation = EulerZXY(elem[0],elem[1],elem[2])
            print("hello")
            cmd = RobotCommandBuilder.synchro_stand_command(body_height=0,
                                                    footprint_R_body=orientation)
            self._robot_command_client.robot_command(cmd, end_time_secs=2,lease=sublease.lease_proto)
            time.sleep(1)  # Sleep for half a second to allow for command execution.

        self._lease = self._lease_wallet.advance()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)
        print("ça se passe quand tout ça ? ")
        # Update the lease and power off the robot if appropriate.
        """if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)"""

    def _navigate_all(self, *args):
        """Navigate through a specific route of waypoints."""
        self._lease = self._lease_wallet.advance()
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return
        for waypoint in self._waypoint_to_timestamp: 
            waypoint_id = waypoint[0]
            print("waypoint_id : " , waypoint_id)

            waypoint_id = graph_nav_util.find_unique_waypoint_id(
                waypoint_id, self._current_graph, self._current_annotation_name_to_wp_id)
            sublease = self._lease.create_sublease()
            self._lease_keepalive.shutdown()
            nav_to_cmd_id = None
            # Navigate to the destination waypoint.
            is_finished = False
            while not is_finished:
                # Issue the navigation command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                try:
                    max_vel_linear = geometry_pb2.Vec2(x=0.5, y=0.5)
                    max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                            angular=0.5)
                    travel_params = self._graph_nav_client.generate_travel_params(0.0,0.0,geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2))
                    destination_waypoint = graph_nav_util.find_unique_waypoint_id(
                    waypoint_id, self._current_graph, self._current_annotation_name_to_wp_id)

                    nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                    leases=[sublease.lease_proto],
                                                                    command_id=nav_to_cmd_id,travel_params =travel_params)
                except ResponseError as e:
                    print("Error while navigating {}".format(e))
                    break
                time.sleep(.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the navigation command is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_to_cmd_id)
            list_rotation = []
            list_rotation.append([0,0,math.pi/2])
            list_rotation.append([0,0,-math.pi/2])
            list_rotation.append([0,-math.pi/2,0])
            list_rotation.append([0,math.pi/2,0])
            if is_finished: 
                for elem in list_rotation:
                    print("allez on est ou là ? ")
                    print("elem : ", elem)

                    self._lease = self._lease_wallet.advance()
                    sublease = self._lease.create_sublease()
                    orientation = EulerZXY(elem[0],elem[1],elem[2])
                    print("hello")
                    print(orientation)
                    cmd = RobotCommandBuilder.synchro_stand_command(body_height=0,
                                                            footprint_R_body=orientation)
                    self._robot_command_client.robot_command(cmd, end_time_secs=2,lease=sublease.lease_proto)
                    time.sleep(2)  # Sleep for half a second to allow for command execution.

    def _navigate_all_no_rotate(self, *args):
        """Navigate through a specific route of waypoints."""
        self._lease = self._lease_wallet.advance()
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return
        for waypoint in self._waypoint_to_timestamp: 
            waypoint_id = waypoint[0]
            waypoint_id = graph_nav_util.find_unique_waypoint_id(
                waypoint_id, self._current_graph, self._current_annotation_name_to_wp_id)
            sublease = self._lease.create_sublease()
            self._lease_keepalive.shutdown()
            nav_to_cmd_id = None
            # Navigate to the destination waypoint.
            is_finished = False
            while not is_finished:
                # Issue the navigation command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                try:
                    max_vel_linear = geometry_pb2.Vec2(x=0.5, y=0.5)
                    max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                            angular=0.5)
                    travel_params = self._graph_nav_client.generate_travel_params(0.0,0.0,geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2))
                    destination_waypoint = graph_nav_util.find_unique_waypoint_id(
                    waypoint_id, self._current_graph, self._current_annotation_name_to_wp_id)

                    nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                    leases=[sublease.lease_proto],
                                                                    command_id=nav_to_cmd_id,travel_params =travel_params)
                except ResponseError as e:
                    print("Error while navigating {}".format(e))
                    break
                time.sleep(.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the navigation command is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_to_cmd_id)
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)
            
        
        
    def _navigate_route(self,*args):
        graph = self._graph_nav_client.download_graph()#maybe not necessary
        print(self._waypoint_to_timestamp)
        for waypoint in self._waypoint_to_timestamp: 
            waypoint_id = waypoint[0]
            #print("waypoint_id : ", [waypoint_id])
            
            self._navigate_to([waypoint_id])
            #print(self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
            
        # Power on the robot and stand it up.
        

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph(lease=self._lease.lease_proto)

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10)  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have not status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True
        else:
            # Navigation command is not complete yet.
            return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None
    def relative_moves(self,yaw,roll, pitch,frame_name,robot_command_client, robot_state_client, stairs=False):
        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Get position of body relative to the odom
        odom_tform_body = get_a_tform_b(
            transforms,
            ODOM_FRAME_NAME,
            BODY_FRAME_NAME
        )
        roll, pitch, yaw = euler_from_quaternion(odom_tform_body.rot.x,odom_tform_body.rot.y,odom_tform_body.rot.z,odom_tform_body.rot.w)

        footprints =[]
        footprints.append( bosdyn.geometry.EulerZXY( roll=roll, pitch=0,yaw=0))
        footprints.append( bosdyn.geometry.EulerZXY( roll=-roll,pitch=0,yaw=0))
        footprints.append( bosdyn.geometry.EulerZXY( pitch=-roll,roll=0,yaw=0))

        footprints.append(bosdyn.geometry.EulerZXY( pitch=pitch,roll=0,yaw=0))
        end_time = 40
        for i in range(4):
            params = RobotCommandBuilder.mobility_params(body_height=0.0,
                                                            footprint_R_body=footprints[i])
            robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(goal_x=odom_tform_body.x, goal_y=odom_tform_body.y, goal_heading=yaw,
            frame_name=frame_name,params=params)
            cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                        end_time_secs=time.time() + end_time)
            time.sleep(2)
    def return_lease(self):
        """Shutdown lease keep-alive and return lease."""
        self._lease_keepalive.shutdown()
        self._lease_client.return_lease(self._lease)

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(RobotCommandBuilder.safe_power_off_command(),
                                                     end_time_secs=time.time())
        self.return_lease()

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print("""
            Options:
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (3) Initialize localization to a specific waypoint (must be exactly at the waypoint).
            (4) List the waypoint ids and edge ids of the map on the robot.
            (5) Upload the graph and its snapshots.
            (6) Navigate to. The destination waypoint id is the second argument.
            (7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
            (8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
                [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
                When a value for z is not specified, we use the current z height.
                When only yaw is specified, the quaternion is constructed from the yaw.
                When yaw is not specified, an identity quaternion is used.
            (9) Clear the current graph.
            (0) Create a full path of the downloaded path. (This is not optimised we could say that we should go to the nearest waypoint and not the first.)
            (a) Execute full path without rotation at speed x :0.5, y:0.5 and angular 0.5
            (q) Exit.
            """)
            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
                self._on_quit()
                break

            if req_type not in self._command_dictionary:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)

    def get_anchor_pos(self,id):
        graph = self._graph_nav_client.download_graph()
        for elem in graph.anchoring.anchors: 
            if elem.id == id: 
                pos = elem.seed_tform_waypoint.position
                return [pos.x, pos.y, pos.z]
        
def main(argv):
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-u', '--upload-filepath',
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    #bosdyn.client.util.add_common_arguments(parser)
    options = parser.parse_args(argv)

    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
    robot = sdk.create_robot('192.168.80.3')
    robot.authenticate('user', 'upsa43jm7vnf')
    sdk_1 = bosdyn.client.create_standard_sdk('RobotCommandMaster')
    robot_1 = sdk.create_robot('192.168.80.3')
    robot_1.authenticate('user', 'upsa43jm7vnf')
    graph_nav_command_line = GraphNavInterface(robot,robot_1, options.upload_filepath)
    try:
        graph_nav_command_line.run()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(exc)
        print("Graph nav command line client threw an error.")
        graph_nav_command_line.return_lease()
        return False


if __name__ == '__main__':
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
