import sys
import cv2
import numpy as np

import argparse

NEW_SIZE_FACTOR = 0.4
OBJ_DIM = (15, 15)
CROSSCHAIR_DIM = (15, 15)
RED = (0,0,255)
GREEN = (0,255,0)

NUM_PARTICLES = 200
PARTICLE_SIGMA = np.min([OBJ_DIM]) // 4 # particle filter shift per generation
DIST_SIGMA = 0.5

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_video")
	return parser.parse_args()


def make_crosshairs(img, top_left_xy, bot_right_xy, ch_color, captured):
	obj_h, obj_w = CROSSCHAIR_DIM
	center_y, center_x = img.shape[0]//2, img.shape[1]//2

	img = cv2.rectangle(img, top_left_xy, bot_right_xy, ch_color, 1)
	img = cv2.line(img, (center_x, img.shape[0]*1//3), (center_x, center_y-obj_h//2), ch_color, 1)
	img = cv2.line(img, (center_x, center_y+obj_h//2), (center_x, img.shape[0]*2//3), ch_color, 1)
	img = cv2.line(img, (img.shape[0]*1//3, center_y), (center_x-obj_w//2, center_y), ch_color, 1)
	img = cv2.line(img, (center_x+obj_w//2, center_y), (img.shape[1]*2//3, center_y), ch_color, 1)
	return img

def mark_target(img, center_xy, ch_color, captured):
	obj_h, obj_w = OBJ_DIM
	center_x, center_y = int(center_xy[0]), int(center_xy[1])

	t1_x = int(center_xy[0] - OBJ_DIM[1]//2)
	t1_y = int(center_xy[1] - OBJ_DIM[0]//2)
	br_x = int(center_xy[0] + OBJ_DIM[1]//2)
	br_y = int(center_xy[1] + OBJ_DIM[0]//2)

	img = cv2.rectangle(img, (t1_x, t1_y), (br_x, br_y), ch_color, 1)
	img = cv2.line(img, (center_x, 0), (center_x, center_y-obj_h//2), ch_color, 1)
	img = cv2.line(img, (center_x, center_y+obj_h//2), (center_x, img.shape[0]), ch_color, 1)
	img = cv2.line(img, (0, center_y), (center_x-obj_w//2, center_y), ch_color, 1)
	img = cv2.line(img, (center_x+obj_w//2, center_y), (img.shape[1], center_y), ch_color, 1)

	return img

def object_rect(x, y):
	top_left_x = x - OBJ_DIM[1]//2
	top_left_y = y - OBJ_DIM[0]//2
	bot_right_x = x + OBJ_DIM[1]//2
	bot_right_y = y + OBJ_DIM[0]//2
	return top_left_x, top_left_y, bot_right_x, bot_right_y


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
	global clicked_x, clicked_y, is_click

	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		clicked_x = x
		clicked_y = y
		is_click = True
	else:
		is_click = False

if __name__ == "__main__":
	args = get_args()

	cam = cv2.VideoCapture(args.input_video)
	captured = False
	img_patch = np.zeros(OBJ_DIM)
	sift = cv2.xfeatures2d.SIFT_create()
	FLANN_INDEX_KDTREE = 0 #for sift, surf etc
	MIN_MATCH_COUNT = 4
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) #for sift, surf etc
	search_params = dict(checks = 50)
	# Object of Fast Approximate Nearest Neighbor Search Library
	flann = cv2.FlannBasedMatcher(index_params, search_params)

	particles_xy, particles_scores, particles_patches = [], [], []

	ret, img = cam.read()
	if img is None:
		cam.release()
		sys.exit(0)

	img_color = cv2.resize(img, (int(img.shape[1]*NEW_SIZE_FACTOR), int(img.shape[0]*NEW_SIZE_FACTOR)))
	img_h, img_w, _ = img_color.shape

	fps = cam.get(cv2.CAP_PROP_FPS)
	out_writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (img_w, img_h))

	clicked_x, clicked_y = img_w//2, img_h//2
	is_click = False

	# old:
	# top_left_x = img_w//2 - OBJ_DIM[1]//2
	# top_left_y = img_h//2 - OBJ_DIM[0]//2
	# bot_right_x = img_w//2 + OBJ_DIM[1]//2
	# bot_right_y = img_h//2 + OBJ_DIM[0]//2

	# object corners for mapping
	# h, w = key_img_patch.shape
	pts = np.float32([ [0,0],[0,OBJ_DIM[1]-1],[OBJ_DIM[0]-1,OBJ_DIM[1]-1],[OBJ_DIM[0]-1,0] ]).reshape(-1,1,2)
	
	i = 0
	start_track_frame = -1

	while True:
		_, img = cam.read()
		i += 1
		if img is None or (i - start_track_frame == 100 and start_track_frame != -1):
			cam.release()
			sys.exit(0)

		
		img_color = cv2.resize(img, (int(img.shape[1]*NEW_SIZE_FACTOR), int(img.shape[0]*NEW_SIZE_FACTOR)))
		img_color_clean = img_color.copy()
		img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

		# object rect
		top_left_x, top_left_y, bot_right_x, bot_right_y = object_rect(clicked_x, clicked_y)
		cv2.namedWindow('Object Tracker')	
		cv2.setMouseCallback('Object Tracker', click_event)

		if captured:
			key = cv2.waitKey(30) & 0xFF
		else:
			key = cv2.waitKey(20) & 0xFF
		
		if key == 27: # ESC
			break

		if (key == ord('c')) or (key == ord('C')) or is_click:
			start_track_frame = i
			captured = True
			img_patch = img_color_clean[top_left_y:bot_right_y, top_left_x:bot_right_x]
			# particles_xy = np.zeros((NUM_PARTICLES, 2))
			# # particles_xy[:, :] = [img_w//2, img_h//2]
			# particles_xy[:, :] = [int(clicked_x), int(clicked_y)]
			print(i, clicked_x, clicked_y)
			# find sift feature on key image patch
			key_img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
			kp_roi, des_roi = sift.detectAndCompute(key_img_patch, None)
			print(len(kp_roi))
			if len(kp_roi) < NUM_PARTICLES:
				particles_xy = np.array([[int(clicked_x), int(clicked_y)] for _ in range(NUM_PARTICLES-len(kp_roi))] + [p.pt for p in kp_roi])
			else:
				particles_xy = np.array([p.pt for p in kp_roi])

		elif (key == ord('d')) or (key == ord('D')):
			captured = False
			img_patch = np.zeros(img_patch.shape)
		
		if captured:
			# randomly sample particles
			for i, p in enumerate(particles_xy):
				if i == 0: 
					continue
				p[0] += np.random.normal(0, PARTICLE_SIGMA)
				p[1] += np.random.normal(0, PARTICLE_SIGMA)

				# adjust for out of frame particles
				p[0] = OBJ_DIM[1]//2 if p[0] < OBJ_DIM[1]//2 else p[0]
				p[0] = img_w - OBJ_DIM[1]//2 if p[0] > img_w - OBJ_DIM[1]//2 else p[0]
				p[1] = OBJ_DIM[0]//2 if p[1] < OBJ_DIM[0]//2 else p[1]
				p[1] = img_h - OBJ_DIM[0]//2 if p[1] > img_h - OBJ_DIM[0]//2 else p[1]

			# display particles
			for p in particles_xy:
				img_color = cv2.circle(img_color, (int(p[0]), int(p[1])), 1, GREEN, -1)

			# get patches for each particle
			particles_patches = []
			for p in particles_xy:
				patch_top_left_x = int(p[0] - OBJ_DIM[1]//2)
				patch_top_left_y = int(p[1] - OBJ_DIM[0]//2)
				patch_bot_right_x = int(p[0] + OBJ_DIM[1]//2)
				patch_bot_right_y = int(p[1] + OBJ_DIM[0]//2)
				temp_patch = img_color_clean[patch_top_left_y:patch_bot_right_y, patch_top_left_x:patch_bot_right_x]
				particles_patches.append(temp_patch)
				
			# compare each patch with the model patch
			model_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
			model_patch = cv2.GaussianBlur(model_patch, (3,3), 0)
			particles_scores = []
			for p in particles_patches:
				temp_patch = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
				temp_patch = cv2.GaussianBlur(temp_patch, (3,3), 0)
				mse = np.mean((model_patch - temp_patch)**2)
				particles_scores.append(mse)
				
			# convert to Gaussian dist
			particles_scores = np.array(particles_scores)
			# missing np.sqrt() is intentional as NaN
			particles_scores = 1. / (2. * np.pi * DIST_SIGMA) * np.exp(-particles_scores/(2.*DIST_SIGMA**2))
			# particles_scores = 1. / (np.sqrt(2. * np.pi) * DIST_SIGMA) * np.exp(-particles_scores**2/(2.*DIST_SIGMA**2))
			particles_scores = particles_scores/np.sum(particles_scores)

			# resample
			new_pxy_idx = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES-1, p=particles_scores, replace=True)
			best_idx = np.where(particles_scores == np.max(particles_scores))[0][0]
			best_xy = particles_xy[best_idx]
			new_set = particles_xy[new_pxy_idx]
			particles_xy = np.vstack((best_xy, new_set))

			# kp_frame, des_frame = sift.detectAndCompute(img, None)	

			# # Getting nearest matches using FLANN library
			# matches = flann.knnMatch(des_roi, des_frame, k=2)

			# good = []
			# # Outlier Filtering by nearest neighbour distance ratio
			# if len(matches) > 1:
			# 	for m,n in matches:
			# 		if m.distance < 0.7*n.distance:
			# 			good.append(m)
			# # print(len(good))

			# #On getting enough good matches
			# if len(good) >= MIN_MATCH_COUNT:
			# 	#Getting matched keypoint's location in ROI
			# 	src_pts = np.float32([ kp_roi[m.queryIdx].pt for m in good ]).reshape(-1,1,2)

			# 	#Getting matched keypoint's location in current frame
			# 	dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			# 	# find mapping matrix from src to dst
			# 	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
				
			# 	# Getting perspective transform
			# 	dst = cv2.perspectiveTransform(pts, M) # [4, 1, 2]

			# 	# Drawing rotating rectangle around object
			# 	cv2.polylines(img_color, [np.int32(dst)], True, (255,255,255), 3, cv2.LINE_AA)

			# 	# display best_xy/ mark target
			# 	best_xy = np.mean(dst, axis=0).astype(np.int32)[0]
			# 	print(best_xy)
			# 	img_color = mark_target(img_color, best_xy, RED, 1)
			
			# else:
			# 	print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))


			# display best_xy/ mark target
			img_color = mark_target(img_color, best_xy, RED, 1)

			# update model patch
			img_patch = particles_patches[best_idx]
			# xleft, xright, yleft, yright = 0, 0, 0, 0
			# xleft = int(np.min(dst[..., 0]))
			# xright = int(np.max(dst[..., 0]))
			# yleft = int(np.min(dst[..., 1]))
			# yright = int(np.min(dst[..., 1]))
			# img_patch = img_color_clean[yleft:yright, xleft:xright]

			
		img_color = make_crosshairs(img_color, (top_left_x, top_left_y), (bot_right_x, bot_right_y), GREEN, 1)
		cv2.imshow("Object Tracker", img_color)
		out_writer.write(img_color)

	cam.release()
	out_writer.release()
	cv2.destroyAllWindows()















