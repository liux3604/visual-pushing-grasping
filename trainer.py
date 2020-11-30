import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from models import reactive_net, reinforcement_net
from scipy import ndimage
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, method, push_rewards, future_reward_discount,
                 is_testing, load_snapshot, snapshot_file, force_cpu):

        self.method = method

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print(
                "CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional classification network for supervised learning
        if self.method == 'reactive':
            self.model = reactive_net(self.use_cuda)

            # Initialize classification loss
            push_num_classes = 3  # 0 - push, 1 - no change push, 2 - no loss
            push_class_weights = torch.ones(push_num_classes)
            push_class_weights[push_num_classes - 1] = 0
            if self.use_cuda:
                self.push_criterion = CrossEntropyLoss2d(push_class_weights.cuda()).cuda()
            else:
                self.push_criterion = CrossEntropyLoss2d(push_class_weights)
            grasp_num_classes = 3  # 0 - grasp, 1 - failed grasp, 2 - no loss
            grasp_class_weights = torch.ones(grasp_num_classes)
            grasp_class_weights[grasp_num_classes - 1] = 0
            if self.use_cuda:
                self.grasp_criterion = CrossEntropyLoss2d(
                    grasp_class_weights.cuda()).cuda()
            else:
                self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights)

        # Fully convolutional Q network for deep reinforcement learning
        elif self.method == 'reinforcement':
            self.model = reinforcement_net(self.use_cuda)
            self.push_rewards = push_rewards
            self.future_reward_discount = future_reward_discount

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduce=False)  # Huber loss
            if self.use_cuda:
                self.criterion = self.criterion.cuda()

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' %
                  (snapshot_file))

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.expected_reward_log = []
        self.current_one_time_reward_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []

    # Pre-load execution info and RL variables

    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(
            transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.expected_reward_log = np.loadtxt(os.path.join(
            transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.expected_reward_log = self.expected_reward_log[0:self.iteration]
        self.expected_reward_log.shape = (self.iteration, 1)
        self.expected_reward_log = self.expected_reward_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(
            transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.current_one_time_reward_log = np.loadtxt(os.path.join(
            transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.current_one_time_reward_log = self.current_one_time_reward_log[0:self.iteration]
        self.current_one_time_reward_log.shape = (self.iteration, 1)
        self.current_one_time_reward_log = self.current_one_time_reward_log.tolist()
        self.use_heuristic_log = np.loadtxt(os.path.join(
            transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration, 1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(
            transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration, 1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(
            transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0], 1)
        self.clearance_log = self.clearance_log.tolist()

    # Compute forward pass through model to compute affordances/Q

    def forward(self, color_heightmap, depth_heightmap, object_mass, is_volatile=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:, :, c] = (input_depth_image[:, :, c] - image_mean[c])/image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

        # Pass input data through model
        output_prob, state_feat = self.model.forward(input_color_data, input_depth_data, object_mass, is_volatile, specific_rotation)

        if self.method == 'reactive':

            # Return affordances (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = F.softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[:, 0, (padding_width/2):(
                        color_heightmap_2x.shape[0]/2 - padding_width/2), (padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = F.softmax(output_prob[rotate_idx][1], dim=1).cpu().data.numpy()[:, 0, (padding_width/2):(
                        color_heightmap_2x.shape[0]/2 - padding_width/2), (padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, F.softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy(
                    )[:, 0, (padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2), (padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, F.softmax(output_prob[rotate_idx][1], dim=1).cpu().data.numpy(
                    )[:, 0, (padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2), (padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif self.method == 'reinforcement':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:, 0, int(padding_width/2):int(
                        color_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, int(padding_width/2):int(
                        color_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:, 0, int(padding_width/2):int(
                        color_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, int(padding_width/2):int(
                        color_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return push_predictions, grasp_predictions, state_feat

    # Calculate the expected reward value and current one-time reward value.
    def calculate_reward_values(self, 
                                primitive_action, 
                                grasp_success, 
                                change_detected, 
                                next_color_heightmap, 
                                next_depth_heightmap, 
                                prev_object_mass):

        if self.method == 'reactive':
            # Compute label value
            expected_reward = 0
            if primitive_action == 'push' and not change_detected:
                expected_reward = 1
            elif primitive_action == 'grasp' and not grasp_success:
                expected_reward = 1

            print('Label value: %d' % (expected_reward))
            return expected_reward, expected_reward

        elif self.method == 'reinforcement':

            # Compute current reward
            current_onetime_reward = 0
            if primitive_action == 'push' and change_detected and self.push_rewards:
                current_onetime_reward = 0.5
            elif primitive_action == 'grasp' and grasp_success:
                current_onetime_reward = 1.0

            # Compute future reward
            if not change_detected and not grasp_success: # speed up stepping out of dead loop of no grasp and no-change
                future_reward = 0
            else:
                next_push_predictions, next_grasp_predictions, _ = self.forward(color_heightmap=next_color_heightmap, 
                                                                                depth_heightmap=next_depth_heightmap, 
                                                                                object_mass=prev_object_mass, 
                                                                                is_volatile=True,
                                                                                specific_rotation=-1)
                future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))

            expected_reward = current_onetime_reward + self.future_reward_discount * future_reward
            print('Current reward: %f' % (current_onetime_reward))
            print('Future reward: %f' % (future_reward))
            print('Expected reward: %f + %f x %f = %f' % (current_onetime_reward, self.future_reward_discount, future_reward, expected_reward))
            return expected_reward, current_onetime_reward

    # Compute labels and backpropagate

    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, expected_reward, prev_object_mass):

        if self.method == 'reactive':

            # Compute fill value
            fill_value = 2

            # Compute labels
            label = np.zeros((1, 320, 320)) + fill_value
            action_area = np.zeros((224, 224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((224, 224)) + fill_value
            tmp_label[action_area > 0] = expected_reward
            label[0, 48:(320-48), 48:(320-48)] = tmp_label

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'push':
                # loss = self.push_criterion(self.model.output_prob[best_pix_ind[0]][0], Variable(torch.from_numpy(label).long().cuda()))

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap, prev_object_mass, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.push_criterion(self.model.output_prob[0][0], Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.push_criterion(self.model.output_prob[0][0], Variable(torch.from_numpy(label).long()))
                loss.backward()
                loss_value = loss.cpu().data.numpy()

            elif primitive_action == 'grasp':
                # loss = self.grasp_criterion(self.model.output_prob[best_pix_ind[0]][1], Variable(torch.from_numpy(label).long().cuda()))
                # loss += self.grasp_criterion(self.model.output_prob[(best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations][1], Variable(torch.from_numpy(label).long().cuda()))

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap, prev_object_mass, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.grasp_criterion(self.model.output_prob[0][1], Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.grasp_criterion(self.model.output_prob[0][1], Variable(torch.from_numpy(label).long()))
                loss.backward()
                loss_value += loss.cpu().data.numpy()

                # Since grasping is symmetric, train with another forward pass of opposite rotation angle
                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

                push_predictions, grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap, prev_object_mass, is_volatile=False, specific_rotation=opposite_rotate_idx)

                if self.use_cuda:
                    loss = self.grasp_criterion(self.model.output_prob[0][1], Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.grasp_criterion(self.model.output_prob[0][1], Variable(torch.from_numpy(label).long()))
                loss.backward()
                loss_value += loss.cpu().data.numpy()

                loss_value = loss_value/2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()

        elif self.method == 'reinforcement':

            # Compute labels
            label = np.zeros((1, 320, 320))
            action_area = np.zeros((224, 224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((224, 224))
            tmp_label[action_area > 0] = expected_reward
            label[0, 48:(320-48), 48:(320-48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224, 224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0, 48:(320-48), 48:(320-48)] = tmp_label_weights

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'push':

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap, prev_object_mass, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320), Variable(torch.from_numpy(
                        label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320), Variable(torch.from_numpy(
                        label).float())) * Variable(torch.from_numpy(label_weights).float(), requires_grad=False)
                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

            elif primitive_action == 'grasp':

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap, prev_object_mass, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), Variable(torch.from_numpy(
                        label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), Variable(torch.from_numpy(
                        label).float())) * Variable(torch.from_numpy(label_weights).float(), requires_grad=False)
                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

                push_predictions, grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap, prev_object_mass, is_volatile=False, specific_rotation=opposite_rotate_idx)

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), Variable(torch.from_numpy(
                        label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), Variable(torch.from_numpy(
                        label).float())) * Variable(torch.from_numpy(label_weights).float(), requires_grad=False)

                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                loss_value = loss_value/2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx, :, :].copy()
                # prediction_vis[prediction_vis < 0] = 0 # assume probability
                # prediction_vis[prediction_vis > 1] = 1 # assume probability
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0, 0, 255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

        return canvas

    def push_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[ndimage.interpolation.shift(rotated_heightmap, [0, -25], order=0) - rotated_heightmap > 0.02] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25, 25), np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_push_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_push_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                push_predictions = tmp_push_predictions
            else:
                push_predictions = np.concatenate((push_predictions, tmp_push_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        return best_pix_ind

    def grasp_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0, -25], order=0)
                                       > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0, 25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25, 25), np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind
