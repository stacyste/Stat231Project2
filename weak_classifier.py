from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed


class Weak_Classifier(ABC):
	#initialize a harr filter with the positive and negative rects
	#rects are in the form of [x1, y1, x2, y2] 0-index
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		self.id = id
		self.plus_rects = plus_rects
		self.minus_rects = minus_rects
		self.num_bins = num_bins
		self.activations = None
		self.interpolatedThresholds = None

	#take in one integrated image and return the value after applying the image
	#integrated_image is a 2D np array
	#return value is the number BEFORE polarity is applied
	def apply_filter2image(self, integrated_image):
		pos = 0
		for rect in self.plus_rects:
			rect = [int(n) for n in rect]
			pos += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		neg = 0
		for rect in self.minus_rects:
			rect = [int(n) for n in rect]
			neg += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		return pos - neg
	
		
	#take in a list of integrated images and calculate values for each image
	#integrated images are passed in as a 3-D np-array
	#calculate activations for all images BEFORE polarity is applied
	#only need to be called once
	def apply_filter(self, integrated_images):
		values = []
		for idx in range(integrated_images.shape[0]):
			values.append(self.apply_filter2image(integrated_images[idx, ...]))
		if (self.id + 1) % 100 == 0:
			print('Weak Classifier No. %d has finished applying' % (self.id + 1))
		return values
	
	#using this function to compute the error of
	#applying this weak classifier to the dataset given current weights
	#return the error and potentially other identifier of this weak classifier
	#detailed implementation is up you and depends
	#your implementation of Boosting_Classifier.train()
	@abstractmethod
	def calc_error(self, weights, labels):
		pass
	
	@abstractmethod
	def predict_image(self, integrated_image):
		pass

class Ada_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.polarity = None
		self.threshold = None

	def calc_error(self, weights, labels):

		#print("weightsum: ", sum(weights))
		predictedClassifications  = np.sign(np.subtract.outer(self.interpolatedThresholds, self.activations))
		#print("predicted classifications: ", predictedClassifications)
		errList = ((weights*np.array(labels != predictedClassifications)).sum(axis=1))
		#print("errors: ", errList)

		polarities = np.full(len(errList), 1) #initialize all polarities to 1
		polarities[errList > .5] = -1 
		errList[errList > .5] = 1-errList[errList > .5]
		#print("Polarities: ", polarities)
		#print("polarized error list: ", errList)

		minimumErrorIndex = np.argmin(errList) 
		#print("min Index: ", minimumErrorIndex)
		minError = errList[minimumErrorIndex]
		minPolarity = polarities[minimumErrorIndex]
		minThreshold = self.interpolatedThresholds[minimumErrorIndex]
		##for threshold in thresholds:
		#	predictedClassifications = [1 if activation > threshold else -1 for activation in self.activations]
			
			#currentError = self.weighted_error_calc(weights, labels, predictedClassifications)
		#	normalizer = sum(weights)
		#	incorrectClassifications = [int(label != prediction) for label,prediction in zip(labels, predictedClassifications)]
		#	errList = [weight*indicator for weight, indicator in zip(weights, incorrectClassifications)]



		#	currentPolarity = 1

			#flip polarity if necessary
		#	if currentError > .5:
		#		currentError = 1 - currentError
		#		currentPolarity = -1

			#update minimum error if it is smaller than before
		#	if currentError < minError:
		#		minError = currentError
		#		minThreshold = threshold
		#		minPolarity = currentPolarity

		#self.polarity = minPolarity
		#self.threshold = minThreshold
		return minError, minPolarity, minThreshold


	######################################################
	#Unused
	def make_classification_predictions(self, threshold, num_cores=1):
		return [1 if activation > threshold else -1 for activation in self.activations]

	def weighted_error_calc(self, weights, labels, classificationPredictions, num_cores=1):
		normalizer = sum(weights)
		incorrectClassifications = self.classification_indicator_function(labels, classificationPredictions)
		errList = [weight*indicator for weight, indicator in zip(weights, incorrectClassifications)]

		return sum(errList)/normalizer

	#Indicator for errors, 1 if error, 0 if correct
	def classification_indicator_function(self, labels, classificationPredictions):
		return [int(label != prediction) for label,prediction in zip(labels, classificationPredictions)]


	def calculate_error_with_polarity(self, er):
		if er > .5:
			error = 1-er
			polarity = -1
		else:
			error = er
			polarity = 1

		return error, polarity

	def errors_with_correct_polarities(self, errorList):
		errors = []
		polarities= []
		for er in errorList:
			if er > .5:
				errors.append(1-er)
				polarities.append(-1)
			else:
				errors.append(er)
				polarities.append(1)

		return errors, polarities
		######################################################

	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		return self.polarity * np.sign(value - self.threshold)

class Real_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.thresholds = None #this is different from threshold in ada_weak_classifier, think about it
		self.bin_pqs = None
		self.train_assignment = Noneactivations

	def calc_error(self, weights, labels):
		######################
		######## TODO ########
		######################
		return
	
	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		bin_idx = np.sum(self.thresholds < value)
		return 0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx])

def main():
	plus_rects = [(1, 2, 3, 4)]
	minus_rects = [(4, 5, 6, 7)]
	num_bins = 50
	id_num = 1
	ada_hf = Ada_Weak_Classifier(id_num, plus_rects, minus_rects, num_bins)
	real_hf = Real_Weak_Classifier(id_num, plus_rects, minus_rects, num_bins)

if __name__ == '__main__':
	main()
