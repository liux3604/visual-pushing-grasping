import threading
import shared
import numpy as np
import cv2
class Process_Actions(threading.Thread):
    def __init__(self, args, trainer, logger, robot):
        threading.Thread.__init__(self)
        self.args = args
        self.explore_prob = 0.5 if not args.is_testing else 0.0
        self.trainer = trainer
        self.logger = logger
        self.robot = robot

    def run(self):
        while True:
            shared.action_semaphore.acquire()
            # Determine whether grasping or pushing should be executed based on network predictions
            best_push_conf = np.max(shared.push_predictions)
            best_grasp_conf = np.max(shared.grasp_predictions)
            print('Primitive confidence scores: %f (push), %f (grasp)' %
                  (best_push_conf, best_grasp_conf))
            shared.primitive_action = 'grasp'
            explore_actions = False
            if not self.args.grasp_only:
                if self.args.is_testing and self.args.method == 'reactive':
                    if best_push_conf > 2*best_grasp_conf:
                        shared.primitive_action = 'push'
                else:
                    if best_push_conf > best_grasp_conf:
                        shared.primitive_action = 'push'
                explore_actions = np.random.uniform() < self.explore_prob
                # Exploitation (do best action) vs exploration (do other action)
                if explore_actions:
                    print('Strategy: explore (exploration probability: %f)' % (
                        self.explore_prob))
                    shared.primitive_action = 'push' if np.random.randint(
                        0, 2) == 0 else 'grasp'
                else:
                    print('Strategy: exploit (exploration probability: %f)' % (
                        self.explore_prob))
            self.trainer.is_exploit_log.append(
                [0 if explore_actions else 1])
            self.logger.write_to_log(
                'is-exploit', self.trainer.is_exploit_log)

            # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
            # NOTE: typically not necessary and can reduce final performance.
            if self.args.heuristic_bootstrap and shared.primitive_action == 'push' and shared.no_change_count[0] >= 2:
                print(
                    'Change not detected for more than two pushes. Running heuristic pushing.')
                shared.best_pix_ind = self.trainer.push_heuristic(
                    shared.valid_depth_heightmap)
                shared.no_change_count[0] = 0
                predicted_value = shared.push_predictions[shared.best_pix_ind]
                use_heuristic = True
            elif self.args.heuristic_bootstrap and shared.primitive_action == 'grasp' and shared.no_change_count[1] >= 2:
                print(
                    'Change not detected for more than two grasps. Running heuristic grasping.')
                shared.best_pix_ind = self.trainer.grasp_heuristic(
                    shared.valid_depth_heightmap)
                shared.no_change_count[1] = 0
                predicted_value = shared.grasp_predictions[shared.best_pix_ind]
                use_heuristic = True
            else:
                use_heuristic = False

                # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                if shared.primitive_action == 'push':
                    shared.best_pix_ind = np.unravel_index(
                        np.argmax(shared.push_predictions), shared.push_predictions.shape)
                    predicted_value = np.max(shared.push_predictions)
                elif shared.primitive_action == 'grasp':
                    shared.best_pix_ind = np.unravel_index(
                        np.argmax(shared.grasp_predictions), shared.grasp_predictions.shape)
                    predicted_value = np.max(shared.grasp_predictions)
            self.trainer.use_heuristic_log.append(
                [1 if use_heuristic else 0])
            self.logger.write_to_log(
                'use-heuristic', self.trainer.use_heuristic_log)

            # Save predicted confidence value
            self.trainer.predicted_value_log.append([predicted_value])
            self.logger.write_to_log('predicted-value',
                                     self.trainer.predicted_value_log)

            # Compute 3D position of pixel
            print('Action: %s at (%d, %d, %d)' % (shared.primitive_action,
                                                  shared.best_pix_ind[0], shared.best_pix_ind[1], shared.best_pix_ind[2]))
            best_rotation_angle = np.deg2rad(
                shared.best_pix_ind[0]*(360.0/self.trainer.model.num_rotations))
            best_pix_x = shared.best_pix_ind[2]
            best_pix_y = shared.best_pix_ind[1]
            primitive_position = [best_pix_x * self.args.heightmap_resolution + self.args.workspace_limits[0][0], best_pix_y * self.args.heightmap_resolution +
                                  self.args.workspace_limits[1][0], shared.valid_depth_heightmap[best_pix_y][best_pix_x] + self.args.workspace_limits[2][0]]

            # If pushing, adjust start position, and make sure z value is safe and not too low
            # or shared.primitive_action == 'place':
            if shared.primitive_action == 'push':
                finger_width = 0.02
                safe_kernel_width = int(
                    np.round((finger_width/2)/self.args.heightmap_resolution))
                local_region = shared.valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, shared.valid_depth_heightmap.shape[0]), max(
                    best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, shared.valid_depth_heightmap.shape[1])]
                if local_region.size == 0:
                    safe_z_position = self.args.workspace_limits[2][0]
                else:
                    safe_z_position = np.max(
                        local_region) + self.args.workspace_limits[2][0]
                primitive_position[2] = safe_z_position

            # Save executed primitive
            if shared.primitive_action == 'push':
                self.trainer.executed_action_log.append(
                    [0, shared.best_pix_ind[0], shared.best_pix_ind[1], shared.best_pix_ind[2]])  # 0 - push
            elif shared.primitive_action == 'grasp':
                self.trainer.executed_action_log.append(
                    [1, shared.best_pix_ind[0], shared.best_pix_ind[1], shared.best_pix_ind[2]])  # 1 - grasp
            self.logger.write_to_log('executed-action',
                                     self.trainer.executed_action_log)

            # Visualize executed primitive, and affordances
            if self.args.save_visualizations:
                push_pred_vis = self.trainer.get_prediction_vis(
                    shared.push_predictions, shared.color_heightmap, shared.best_pix_ind)
                self.logger.save_visualizations(
                    self.trainer.iteration, push_pred_vis, 'push')
                cv2.imwrite('visualization.push.png', push_pred_vis)
                grasp_pred_vis = self.trainer.get_prediction_vis(
                    shared.grasp_predictions, shared.color_heightmap, shared.best_pix_ind)
                self.logger.save_visualizations(
                    self.trainer.iteration, grasp_pred_vis, 'grasp')
                cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

            # Initialize variables that influence reward
            shared.push_success = False
            shared.grasp_success = False

            # Execute primitive
            if shared.primitive_action == 'push':
                shared.push_success = self.robot.push(primitive_position, best_rotation_angle, self.args.workspace_limits)
                print('Push: %r' %(shared.push_success))
            elif shared.primitive_action == 'grasp':
                # Avoid collision with floor
                primitive_position[2] = max(primitive_position[2] - 0.015, self.args.workspace_limits[2][0] + 0.02)
                shared.grasp_success, return_obj_handle, simulation_fail= self.robot.grasp( position=primitive_position, 
                                                                                            rot_angle=best_rotation_angle,
                                                                                            place_motion=False)
                if shared.grasp_success:
                    self.robot.remove_object(return_obj_handle, self.robot.obj_target_handles)
                
                print('----- Grasp----- : %r' %(shared.grasp_success))
                
            shared.training_semaphore.release()
