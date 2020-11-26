from threading import Semaphore
primitive_action = None
best_pix_ind = None
push_success = False
grasp_success = False

push_predictions = None
grasp_predictions = None
no_change_count = None
valid_depth_heightmap = None
color_heightmap = None
depth_heightmap = None
object_mass = 1.0

training_semaphore = Semaphore(0)
action_semaphore = Semaphore(0)
restarted = None