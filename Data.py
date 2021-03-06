import time
import os
import numpy as np
import torch
import torch.utils.data as data
from utils import *

class DataLayer(data.Dataset):
	def __init__(self, data_path, current_fold, organ_number, low_range, high_range, \
				slice_threshold, slice_thickness, organ_ID, plane, filter_slices = False):
		self.low_range = low_range
		self.high_range = high_range
		self.slice_thickness = slice_thickness
		self.organ_ID = organ_ID

		image_list = open(training_set_filename(current_fold), 'r').read().splitlines()
		self.training_image_set = np.zeros((len(image_list)), dtype = np.int)
		for i in range(len(image_list)):
			s = image_list[i].split(' ')
			self.training_image_set[i] = int(s[0])
		slice_list = open(list_training[plane], 'r').read().splitlines()
		self.slices = len(slice_list)
		self.image_ID = np.zeros((self.slices), dtype = np.int)
		self.slice_ID = np.zeros((self.slices), dtype = np.int)
		self.image_filename = ['' for l in range(self.slices)]
		self.label_filename = ['' for l in range(self.slices)]
		self.average = np.zeros((self.slices))
		self.pixels = np.zeros((self.slices), dtype = np.int)
		self.filter_slices = filter_slices

		for l in range(self.slices):
			s = slice_list[l].split(' ')
			self.image_ID[l] = s[0]
			self.slice_ID[l] = s[1]
			self.image_filename[l] = s[2] # important
			self.label_filename[l] = s[3] # important
			self.average[l] = float(s[4]) # pixel value avg
			self.pixels[l] = int(s[organ_ID * 5]) # sum of label
		if slice_threshold <= 1: # 0.98
			pixels_index = sorted(range(self.slices), key = lambda l: self.pixels[l])
			last_index = int(math.floor((self.pixels > 0).sum() * slice_threshold))
			min_pixels = self.pixels[pixels_index[-last_index]]
		else: # or set up directly
			min_pixels = slice_threshold
		self.active_index = [l for l, p in enumerate(self.pixels)
							if p >= min_pixels and self.image_ID[l] in self.training_image_set] # true active

	def __getitem__(self, index):
		self.index1 = self.active_index[index]

		if self.filter_slices is True:
			#should filter index1 so that the data could offer enought different active slices?
			while self.slice_ID[self.index1 + self.slice_thickness/2] != self.slice_ID[self.index1] +  self.slice_thickness/2:
				self.index1 = self.index1 - 1
			
			while self.slice_ID[self.index1 - self.slice_thickness/2 - 1] != self.slice_ID[self.index1] - self.slice_thickness/2 - 1:
				self.index1 = self.index1 + 1


		self.index0 = self.index1 - 1
		if self.index1 == 0 or self.slice_ID[self.index0] != self.slice_ID[self.index1] - 1:
			self.index0 = self.index1
		self.index2 = self.index1 + 1
		if self.index1 == self.slices - 1 or self.slice_ID[self.index2] != self.slice_ID[self.index1] + 1:
			self.index2 = self.index1
	
		index_after = []
		index_before = []
		last_valid = self.index1
		for i in range(int(self.slice_thickness / 2)):
			index_tmp = self.index1 + i + 1
			if self.index1 == self.slices - 1:
				index_tmp = self.index1
			if self.slice_ID[index_tmp] != self.slice_ID[self.index1] + i + 1:
				index_tmp = last_valid
			else:
				last_valid = index_tmp

			index_after.append(index_tmp)

		last_valid = self.index1
		for i in range(int(self.slice_thickness) - int(self.slice_thickness/2) - 1):
			index_tmp = self.index1 - i - 1
			if self.index1 == 0:
				index_tmp = self.index1
			if self.slice_ID[index_tmp] != self.slice_ID[self.index1] - i - 1:
				index_tmp = last_valid
			else:
				last_valid = index_tmp
			index_before.append(index_tmp)

		self.index_all = index_before + [self.index1] + index_after
		self.data, self.label = self.load_data()
		return torch.from_numpy(self.data), torch.from_numpy(self.label)

	def __len__(self):
		return len(self.active_index)
	
	def load_data(self):
		if self.slice_thickness == 1:
			image1 = np.load(self.image_filename[self.index1]).astype(np.float32)
			label1 = np.load(self.label_filename[self.index1])
			width = label1.shape[0]
			height = label1.shape[1]
			image = np.repeat(image1.reshape(1, width, height), 3, axis = 0)
			label = label1.reshape(1, width, height)
		elif self.slice_thickness == 3:
			image0 = np.load(self.image_filename[self.index0])
			width = image0.shape[0]
			height = image0.shape[1]
			image = np.zeros((3, width, height), dtype = np.float32)
			image[0, ...] = image0
			image[1, ...] = np.load(self.image_filename[self.index1])
			image[2, ...] = np.load(self.image_filename[self.index2])
			label = np.zeros((3, width, height), dtype = np.uint8)
			label[0, ...] = np.load(self.label_filename[self.index0])
			label[1, ...] = np.load(self.label_filename[self.index1])
			label[2, ...] = np.load(self.label_filename[self.index2])

		else:
			image0 = np.load(self.image_filename[self.index_all[0]])
			width = image0.shape[0]
			height = image0.shape[1]
			image = np.zeros((self.slice_thickness,width,height),dtype = np.float32)
			label = np.zeros((self.slice_thickness,width,height),dtype = np.uint8)
			label[0, ...] = np.load(self.label_filename[self.index_all[0]])
			image[0, ...] = image0
			for i in range(1,int(self.slice_thickness)):
				image[i, ...] = np.load(self.image_filename[self.index_all[i]])
				label[i, ...] = np.load(self.label_filename[self.index_all[i]])

		#np.minimum(np.maximum(image, self.low_range, image), self.high_range, image)
		#image.clip()
		np.clip(image, self.low_range, self.high_range, out = image)
		image -= self.low_range
		image /= (self.high_range - self.low_range)
		label = is_organ(label, self.organ_ID).astype(np.uint8)
		return image, label
