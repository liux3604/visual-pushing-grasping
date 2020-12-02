#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from Robot_sim import Robot
from trainer import Trainer
from logger import Logger
# import utils_sim as utils
import utils_gp_sim as utils_gp
from action import Process_Actions
import shared
import copy

def main(args):
    # --------------- Setup options ---------------
    is_sim = args.is_sim
    # Directory containing 3D mesh files (.obj) for simulation
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None
    obj_model_dir = os.path.abspath(args.obj_model_dir) if is_sim else None

    # Number of objects to add to simulation
    num_obj = args.num_obj if is_sim else None
    # IP and port to robot arm as TCP client (UR5)
    tcp_host_ip = args.tcp_host_ip if not is_sim else None
    tcp_port = args.tcp_port if not is_s-im else None
    # IP and port to robot arm as real-time client (UR5)
    rtc_host_ip = args.rtc_host_ip if not is_sim else None
    rtc_port = args.rtc_port if not is_sim else None
    if is_sim:
        # Cols: min max, Rows: x y z (workspace limits in robot coordinates)
        args.workspace_limits = np.asarray([[0.2,, 0.7], [-0.25, 0.25], [0.0002, 0.2]])
        '''
        np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        '''
    else:
        args.workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])
    # Meters per pixel of heightmap
    heightmap_size = 224    # Workspace heightmap size
    args.heightmap_resolution = (args.workspace_limits[0][1] - args.workspace_limits[0][0]) / heightmap_size  # in meters

    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    # 'reactive' (supervised learning) or 'reinforcement' (Q-learning)
    method = args.method
    # Use immediate rewards (from change detection) for pushing?
    push_rewards = args.push_rewards if method == 'reinforcement' else None
    future_reward_discount = args.future_reward_discount
    # Use prioritized experience replay?
    experience_replay = args.experience_replay
    # Use handcrafted grasping algorithm when grasping fails too many times?
    heuristic_bootstrap = args.heuristic_bootstrap
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only

    # -------------- Testing options --------------
    is_testing = args.is_testing
    # Maximum number of test runs per case/scenario
    max_test_trials = args.max_test_trials
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    # Continue logging from previous session
    continue_logging = args.continue_logging
    logging_directory = os.path.abspath(
        args.logging_directory) if continue_logging else os.path.abspath('logs')
    # Save visualizations of FCN predictions? 0.6s per training step if True
    save_visualizations = args.save_visualizations
    # Set random seed
    np.random.seed(random_seed)

    global robot
    robot = Robot(workspace_limits=args.workspace_limits,
                  num_of_obj=args.num_obj,
                  obj_dir="/home/song/visual-pushing-grasping/objects",
                  is_eval=is_testing)
    global trainer
    trainer = Trainer(method, push_rewards, future_reward_discount, is_testing, load_snapshot, snapshot_file, force_cpu)
    global logger
    logger = Logger(continue_logging, logging_directory)

    # Save camera intrinsics and pose
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale)
    # Save heightmap parameters
    logger.save_heightmap_info(args.workspace_limits, args.heightmap_resolution)

    # load execution info and RL variables of last executed pre-loaded log
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    shared.no_change_count = [0, 0]

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    action_thread = Process_Actions(args, trainer, logger, robot)
    action_thread.daemon = True
    action_thread.start()
    global exit_called
    exit_called = False
    
    is_start_new_training_flag = True
    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' %('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # take snapshot
        color_img, depth_img, shared.color_heightmap, shared.valid_depth_heightmap = utils_gp.get_heightmap(
                                                                                    robot=robot,
                                                                                    heightmap_resolution=args.heightmap_resolution,
                                                                                    workspace_limits=args.workspace_limits)
        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, shared.color_heightmap, shared.valid_depth_heightmap, '0')

        # Making a deep copy for the cases of table_empty_flag=True
        color_heightmap_training = copy.deepcopy(shared.color_heightmap)
        valid_depth_heightmap_training = copy.deepcopy(shared.valid_depth_heightmap)

        # Make sure simulation is still stable (if not, reset simulation)
        sim_stable_flag = True
        if is_sim and (not robot.check_sim_ok()):
            print('Restarting simulation: Simulation unstable.')
            sim_stable_flag = False
            robot.stop_sim()
            robot.restart_sim()
            shared.no_change_count = [0, 0]
            # take_snapshot(robot, args)
            # take snapshot
            color_img, depth_img, shared.color_heightmap, shared.valid_depth_heightmap = utils_gp.get_heightmap(
                                                                                    robot=robot,
                                                                                    heightmap_resolution=args.heightmap_resolution,
                                                                                    workspace_limits=args.workspace_limits)
            # Save RGB-D images and RGB-D heightmaps
            logger.save_images(trainer.iteration, color_img, depth_img, '0')
            logger.save_heightmaps(trainer.iteration, shared.color_heightmap, shared.valid_depth_heightmap, '0')

        # Reset simulation or pause real-world training if table is empty
        table_empty_flag = False
        if isTableEmpty(args):
            table_empty_flag = True
            shared.no_change_count = [0, 0]
            if is_sim:
                robot.stop_sim()
                robot.restart_sim()
            else:
                robot.restart_real()
            # take_snapshot(robot, args)
                        # take snapshot
            color_img, depth_img, shared.color_heightmap, shared.valid_depth_heightmap = utils_gp.get_heightmap(
                                                                                    robot=robot,
                                                                                    heightmap_resolution=args.heightmap_resolution,
                                                                                    workspace_limits=args.workspace_limits)
            # Save RGB-D images and RGB-D heightmaps
            logger.save_images(trainer.iteration, color_img, depth_img, '0')
            logger.save_heightmaps(trainer.iteration, shared.color_heightmap, shared.valid_depth_heightmap, '0')

        if not exit_called:
            # Run forward pass with network to get affordances
            shared.push_predictions, shared.grasp_predictions, _ = trainer.forward(
                                                                    color_heightmap = shared.color_heightmap, 
                                                                    depth_heightmap = shared.valid_depth_heightmap, 
                                                                    object_mass = shared.object_mass, 
                                                                    is_volatile = True,
                                                                    specific_rotation = -1)
            # Execute best primitive action on robot in another thread
            shared.action_semaphore.release()

        # Run training iteration in current thread (aka training thread)
        if sim_stable_flag and 'prev_color_heightmap' in locals():
            # Detect changes
            change_detected = detect_changes(valid_depth_heightmap_training, prev_valid_depth_heightmap, change_threshold=300)
            update_no_change_counct(change_detected, prev_primitive_action)

            # Calculate the expected reward value and current one-time reward value for training.
            expected_reward, current_one_time_reward = trainer.calculate_reward_values(
                                                                primitive_action = prev_primitive_action, 
                                                                grasp_success = prev_grasp_success, 
                                                                change_detected = change_detected, 
                                                                next_color_heightmap = color_heightmap_training, 
                                                                next_depth_heightmap = valid_depth_heightmap_training, 
                                                                prev_object_mass = prev_object_mass)
            # Logging
            trainer.expected_reward_log.append([expected_reward])
            logger.write_to_log('expected_reward(Largest Q-value)', trainer.expected_reward_log)
            trainer.current_one_time_reward_log.append([current_one_time_reward])
            logger.write_to_log('current_one_time_reward', trainer.current_one_time_reward_log)

            # Backpropagate
            trainer.backprop(color_heightmap=prev_color_heightmap, 
                             depth_heightmap=prev_valid_depth_heightmap, 
                             primitive_action=prev_primitive_action, 
                             best_pix_ind=prev_best_pix_ind, 
                             expected_reward=expected_reward, 
                             prev_object_mass=prev_object_mass)

            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9998, trainer.iteration), 0.1) if explore_rate_decay else 0.5

            # Do sampling for experience replay
            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    if method == 'reactive':
                        # random.randint(1, 2) # 2
                        sample_reward_value = 0 if current_one_time_reward == 1 else 1
                    elif method == 'reinforcement':
                        sample_reward_value = 0 if current_one_time_reward == 0.5 else 0.5
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    if method == 'reactive':
                        sample_reward_value = 0 if current_one_time_reward == 1 else 1
                    elif method == 'reinforcement':
                        sample_reward_value = 0 if current_one_time_reward == 1 else 1

                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.current_one_time_reward_log)[1:trainer.iteration, 0] == sample_reward_value, np.asarray(
                    trainer.executed_action_log)[1:trainer.iteration, 0] == sample_primitive_action_id))

                if sample_ind.size > 0:
                    # Find sample with highest surprise value
                    if method == 'reactive':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] - (1 - sample_reward_value))
                    elif method == 'reinforcement':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] - np.asarray(trainer.expected_reward_log)[sample_ind[:, 0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

                    # Compute forward pass with sample
                    with torch.no_grad():
                        sample_push_predictions, sample_grasp_predictions, _ = trainer.forward(
                                                                                            color_heightmap=sample_color_heightmap, 
                                                                                            depth_heightmap=sample_depth_heightmap, 
                                                                                            object_mass=shared.object_mass, 
                                                                                            is_volatile=True,
                                                                                            specific_rotation=-1)

                    # Load next sample RGB-D heightmap
                    next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration+1)))
                    next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration+1)), -1)
                    next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000

                    sample_push_success = sample_reward_value == 0.5
                    sample_grasp_success = sample_reward_value == 1
                    sample_change_detected = sample_push_success
                    # new_sample_expected_reward, _ = trainer.calculate_reward_values(sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected, sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap)

                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
                    trainer.backprop(color_heightmap=sample_color_heightmap, 
                                     depth_heightmap=sample_depth_heightmap, 
                                     primitive_action=sample_primitive_action, 
                                     best_pix_ind=sample_best_pix_ind, 
                                     expected_reward=trainer.expected_reward_log[sample_iteration],
                                     prev_object_mass=prev_object_mass)

                    # Recompute prediction value and label for replay buffer
                    if sample_primitive_action == 'push':
                        trainer.predicted_value_log[sample_iteration] = [
                            np.max(sample_push_predictions)]
                        # trainer.expected_reward_log[sample_iteration] = [new_sample_expected_reward]
                    elif sample_primitive_action == 'grasp':
                        trainer.predicted_value_log[sample_iteration] = [
                            np.max(sample_grasp_predictions)]
                        # trainer.expected_reward_log[sample_iteration] = [new_sample_expected_reward]

                else:
                    print(
                        'Not enough prior training samples. Skipping experience replay.')

            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration,
                                      trainer.model, method)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        shared.training_semaphore.acquire()

        if exit_called:
            break

        # Save information for next training step
        is_start_new_training_flag = False
        prev_color_heightmap = shared.color_heightmap.copy()
        prev_valid_depth_heightmap = shared.valid_depth_heightmap.copy()
        prev_grasp_success = shared.grasp_success
        prev_primitive_action = shared.primitive_action
        prev_best_pix_ind = shared.best_pix_ind
        prev_object_mass = shared.object_mass
        trainer.iteration += 1
        print('Time elapsed: %f' % (time.time()-iteration_time_0))

# Get color and depth heightmap
def take_snapshot(robot, args):
    # Get latest RGB-D image
    color_img, depth_img = robot.get_camera_data()
    # Apply depth scale from calibration
    depth_img = depth_img * robot.cam_depth_scale

    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    shared.color_heightmap, shared.depth_heightmap = utils_gp.get_heightmap(
                                                    color_img=color_img,
                                                    depth_img=depth_img,
                                                    cam_intrinsics=robot.cam_intrinsics,
                                                    cam_pose=robot.cam_pose,
                                                    workspace_limits=args.workspace_limits,
                                                    heightmap_resolution=args.heightmap_resolution)

    shared.valid_depth_heightmap = shared.depth_heightmap.copy()
    shared.valid_depth_heightmap[np.isnan(shared.valid_depth_heightmap)] = 0
    # Save RGB-D images and RGB-D heightmaps
    logger.save_images(trainer.iteration, color_img, depth_img, '0')
    logger.save_heightmaps(trainer.iteration, shared.color_heightmap, shared.valid_depth_heightmap, '0')

# Detect changes between depth_heightmap and prev_depth_heightmap
def detect_changes(curr_valid_depth_heightmap, prev_valid_depth_heightmap, change_threshold=300):
    depth_diff = abs(curr_valid_depth_heightmap - prev_valid_depth_heightmap)
    depth_diff[np.isnan(depth_diff)] = 0
    depth_diff[depth_diff > 0.3] = 0
    depth_diff[depth_diff < 0.01] = 0
    depth_diff[depth_diff > 0] = 1
    change_value = np.sum(depth_diff)
    change_detected = change_value > change_threshold
    print('Change detected: %r (value: %d)' %(change_detected, change_value))
    return change_detected

def update_no_change_counct(change_detected, prev_primitive_action):
    if change_detected:
        if prev_primitive_action == 'push':
            shared.no_change_count[0] = 0
        elif prev_primitive_action == 'grasp':
            shared.no_change_count[1] = 0
    else:
        if prev_primitive_action == 'push':
            shared.no_change_count[0] += 1
        elif prev_primitive_action == 'grasp':
            shared.no_change_count[1] += 1

def isTableEmpty(args):
    ''' Reset simulation or pause real-world training if table is empty by inspecting the table image according to an empty_threshold. 
        If yes, reposition the objects.
        Return true if table is reset.
    '''
    table_isEmpty = False
    if args.is_sim and shared.no_change_count[0] + shared.no_change_count[1] > 3:
        print('Restarting simulation: Failed too many times in simulation. ')
        # If at end of test run, re-load original weights (before test run)
        if args.is_testing:
            trainer.model.load_state_dict(torch.load(snapshot_file))
        table_isEmpty = True

    if args.is_sim and (not len(robot.obj_target_handles)):
        print('Restarting simulation: No object handles left in the scene.')
        table_isEmpty = True

    stuff_count = np.zeros(shared.valid_depth_heightmap.shape)
    stuff_count[shared.valid_depth_heightmap > 0.005] = 1
    total_num_pixels = np.sum(stuff_count)
    empty_threshold = 10 if (args.is_sim and args.is_testing) else 300

    if args.is_sim and total_num_pixels < empty_threshold:
        print('Warning: Camera detects not enough objects in view (value: %d)!.' % (total_num_pixels))
        table_isEmpty = True
    if not args.is_sim and total_num_pixels < empty_threshold:
        print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (total_num_pixels))
        # time.sleep(30)
        table_isEmpty = True

    if table_isEmpty:
        trainer.clearance_log.append([trainer.iteration])
        logger.write_to_log('clearance', trainer.clearance_log)
        if args.is_testing and len(trainer.clearance_log) >= max_test_trials:
            # Exit after training thread (backprop and saving labels)
            global exit_called
            exit_called = True
    
    return table_isEmpty


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim',         dest='is_sim',          action='store_true',    default=False, help='run in simulation?')
    parser.add_argument('--obj_mesh_dir',   dest='obj_mesh_dir',    action='store',    default='objects/blocks', help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--obj_model_dir',  dest='obj_model_dir',   action='store',    default='objects/models', help='directory containing 3D model files (.ttm) of objects to be added to simulation')
    parser.add_argument('--num_obj',        dest='num_obj',         type=int,           action='store', default=10, help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip',    dest='tcp_host_ip',     action='store',     default='100.127.7.223', help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port',       dest='tcp_port',        type=int, action='store', default=30002, help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip',    dest='rtc_host_ip',     action='store', default='100.127.7.223', help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port',       dest='rtc_port',        type=int, action='store', default=30003, help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--random_seed',    dest='random_seed',     type=int, action='store', default=1234, help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu',            dest='force_cpu',       action='store_true', default=False, help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method',     dest='method', action='store', default='reinforcement', help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False, help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False, help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False, help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30, help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False, help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False, help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False, help='save visualizations of FCN predictions?')
    # Run main program with specified arguments
    args = parser.parse_args()
    try:
        main(args)
    finally:
        robot.stop_sim()