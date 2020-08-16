# import the necessary packages
from imutils import paths
import numpy as np
import cv2

def quantify_image(image, bins=(4, 6, 3)):
	# compute a 3D color histogram over the image and normalize it
	#image OpenCV加载的图像
	# bins 绘制直方图时，x轴用作我们的“箱”。在这种情况下，默认  指定 4  色相箱， 6  饱和箱，以及 3  价值箱。
	# 这是一个简短的示例—如果我们仅使用2个（等距）的bin，则我们正在计算像素在[0，128]  或  [128，255 ]范围内的次数。
	# 然后，将与x轴值合并的像素数绘制在y轴上。
	#计算图像上的3D颜色直方图并将其标准化
	hist = cv2.calcHist([image], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()

	# return the histogram
	#返回直方图
	return hist

def load_dataset(datasetPath, bins):
	# grab the paths to all images in our dataset directory, then
	# initialize our lists of images
	imagePaths = list(paths.list_images(datasetPath))
	data = []

	# loop over the image paths
	for imagePath in imagePaths:
		# load the image and convert it to the HSV color space
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# quantify the image and update the data list
		features = quantify_image(image, bins)
		data.append(features)

	# return our data list as a NumPy array
	return np.array(data)