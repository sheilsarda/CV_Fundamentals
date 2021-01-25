"""
Visualization script for part 2 (Invictus vs Harleysville)

Note: You don't have to change this script for the assignment, but you
can if you'd like to change the images or other parameters

"""

import os
import glob
import numpy as np 
import cv2

from compute_ref_line_segment import compute_ref_line_segment

def draw_line(img, p1, p2, color):
	if np.abs(p1[2]) > 1e-6:
		p1 = p1 / p1[2]
	if np.abs(p2[2]) > 1e-6:
		p2 = p2 / p2[2]

	p1 = tuple(p1[:2].astype(int))
	p2 = tuple(p2[:2].astype(int))
	cv2.line(img, p1, p2, color, 3)


def main():
	# Load all image paths, and the keypoints on images
	keypoints = np.load('../data/invictus/keypoints.npy')
	img_files = sorted(glob.glob('../data/invictus/images/*.png'))

	# Process all images
	processed_imgs = []
	for i in range(len(keypoints)):
		img = cv2.imread(img_files[i])
		ref, ll, lr, ur, ul = keypoints[i]

		vanishing_pt, top_pt, bottom_pt = compute_ref_line_segment(ref, ll, lr, ur, ul)
		draw_line(img, ll, ul, (0,0,255))
		draw_line(img, lr, ur, (0,0,255))
		draw_line(img, ll, lr, (51,255,51))
		draw_line(img, ul, ur, (51,255,51))
		draw_line(img, top_pt, bottom_pt, (255,0,0))
		draw_line(img, ul, vanishing_pt, (255,255,0))
		draw_line(img, ur, vanishing_pt, (255,255,0))
		draw_line(img, top_pt, vanishing_pt, (255,255,0))

		processed_imgs.append(img)

	# Save some examples
	save_ind = [0, 50, 100, 150, 200]
	if not os.path.exists('part_2_results'):
	    os.mkdir('part_2_results')

	for ind in save_ind:
	    cv2.imwrite('part_2_results/frame_'+str(ind)+'.png', processed_imgs[ind])

	# Visualize the sequence of processed images
	for im in processed_imgs:
		cv2.imshow('display',im)
		cv2.waitKey(3)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()

