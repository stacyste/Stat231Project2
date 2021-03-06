import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import copy

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:

	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d imags.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
	
	#select weak classifiers to form a strong classifier.
	#after training, by calling self.sc_function(), a prediction can be made.
	#self.chosen_wcs should be assigned a value after self.train() finishes.
	#call Weak_Classifier.calc_error() in this function.
	#cache training results to self.visualizer for visualization.
	#
	#
	#detailed implementation is up to you
	#consider caching partial results and using parallel computing

	def train(self, save_dir = None):
		self.chosen_wcs =[]
		chosenwcIDList = []
		weights = [1/self.data.shape[0]]*self.data.shape[0]

		indxSCtoSave = [1, 10, 20, 50, 100]
		#indxSCtoSave = [0, 1, 2, 3]

		#for T in self.num_chosen_wc
		for t in tqdm(range(self.num_chosen_wc)):
			if self.num_cores == 1:
				wcErrorList = [wc.calc_error(weights, self.labels) for wc in self.weak_classifiers]
			else:
				wcErrorList = Parallel(n_jobs = self.num_cores)(delayed(wc.calc_error)(weights, self.labels) for wc in self.weak_classifiers)
			#find all errors and choose the best classifer
			#wcErrorList = [wc.calc_error(weights, self.labels) for wc in self.weak_classifiers]
			#print("Weak classifier error List: ", wcErrorList)
			minError, pol, thresh  = min(wcErrorList, key = lambda t: t[0])
			
			#print("min error: ", minError, " Polarity: ", pol, " Thresholds: ", thresh)
			bestWcIndx = wcErrorList.index((minError, pol, thresh))
			self.weak_classifiers[bestWcIndx].threshold = thresh
			self.weak_classifiers[bestWcIndx].polarity = pol

			bestWeakClassifier = copy.deepcopy(self.weak_classifiers[bestWcIndx])
			chosenwcIDList.append(bestWeakClassifier.id)

			print("Threshold: ", bestWeakClassifier.threshold)
			print("polarity: ", bestWeakClassifier.polarity)
			#calculate alpha and update classifiers
			alph = .5*(np.log(1-minError) - np.log(minError))
			#print("alpha: ", alph)

			self.chosen_wcs.append([alph, bestWeakClassifier])
			#print("Updated weak classifier attribute: ", len(self.chosen_wcs))

			self.visualizer.strong_classifier_scores[t] = [self.sc_function(img) for img in self.data]


			#print("strong classifier scores: ", self.visualizer.strong_classifier_scores[t])
			
			#create visualizer objects
			if t in indxSCtoSave:
				weakClassifierIndices = list(set(range(len(self.weak_classifiers))) - set(chosenwcIDList))
				weakClassifierErrorList = [wcErrorList[i][0] for i in weakClassifierIndices]
				numberwcToSave = min([len(weakClassifierErrorList), 1000])
				self.visualizer.weak_classifier_accuracies[t] = np.partition(weakClassifierErrorList, numberwcToSave-1)[:numberwcToSave]
				
			weights = self.update_weights(bestWeakClassifier, weights, alph)

		if save_dir is not None:
			pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))

		return self.chosen_wcs


	def update_weights(self, wc, weights, alpha):
		preds = np.sign(np.subtract.outer(wc.threshold, wc.activations))*wc.polarity
		newWeights = weights*np.exp(-alpha*preds*self.labels)
		#weights[labels != preds*wc.polarity] = weights[labels != preds*wc.polarity]*np.exp(alpha)
		#classificationIndicator = [int(label != prediction*wc.polarity) for label,prediction in zip(self.labels, preds)]
		#newWeights = [weight*np.exp(alpha*indicator) for weight, indicator in zip(currentWeights, classificationIndicator)]
		return newWeights/sum(newWeights)

	def calculate_thresholds(self, save_dir = None, load_dir = None):
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached thresholds, %s loading...]' % load_dir)
			wc_thresholds = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_thresholds = [np.linspace(min(wc.activations), max(wc.activations), self.num_bins) for wc in self.weak_classifiers]
			else:
				wc_thresholds = Parallel(n_jobs = self.num_cores)(delayed(np.linspace)(min(wc.activations), max(wc.activations), self.num_bins) for wc in self.weak_classifiers)
			wc_thresholds = np.array(wc_thresholds)
			
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_thresholds)
				print('[Saved calculated thresholds to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.interpolatedThresholds = wc_thresholds[wc.id, :]
		return wc_thresholds

	#########################################################################
	#Unused
	def calculate_alpha(self, selectedClassifierError):
		return .5*(np.log(1-selectedClassifierError) - np.log(selectedClassifierError))

	def weak_classifier_errors(self, weights):
		if self.num_cores == 1:
			wc_errors = [wc.calc_error(weights, self.labels) for wc in self.weak_classifiers]
		else:
			wc_errors = Parallel(n_jobs = self.num_cores, backend = "threading")(delayed(wc.calc_error)(weights, self.labels) for wc in self.weak_classifiers) 
		return wc_errors

	def set_strong_classifier_scores(self):
		scores = [int(self.sc_function(img)) for img in self.data]
		self.visualizer.strong_classifier_scores = scores
		return scores
	############################################################################

	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])			

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, scale_step = 20):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################

		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		
		print('after nms:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #gree rectangular with line width 3

		return img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]

		wrong_patches = patches[np.where(predicts > 0), ...]

		return wrong_patches

	def visualize(self):
		self.visualizer.labels = self.labels
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		self.visualizer.draw_wc_accuracies()
