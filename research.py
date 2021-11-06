import matplotlib.pyplot as plt
import numpy as np
import cv2

arr = np.array([[64,17,153,21,151,54,62,50],[478,20,530,20,530,52,478,52],[64,133,142,137,141,174,62,169],(70,75,157,95,148,132,61,111)])
arr1 = arr.tolist()

def sort(arr):
	n = len(arr)
	# Traverse through all array elements
	for i in range(n-1):
		# Last i elements are already in place
		for j in range(0, n-i-1):
			# traverse the array from 0 to n-i-1
			# Swap if the element found is greater
			# than the next element
			if (arr[j][0] > arr[j + 1][0] and arr[j][7] > arr[j+1][1]) :
				temp = arr[j]
				arr[j] = arr[j + 1]
				arr[j + 1] = temp

sort(arr1)

print(arr1)

