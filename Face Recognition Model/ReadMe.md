EIGENFACES FACE RECOGNITION
---------------------------
1) import all the 10 images of Subject 1 (S1)
2) downsample by a factor of 4
3) Reshape ( vectorize the data matrix)
4) The dimension of the S1 is ~(10,4800)
5) take the mean (M) of all the 10 images. ~(1,4800).
6) Subtract the mean from all the 10 images. ie. centering X1=S1-M
7) Perform SVD or find the covariance. ( You may need to transpose for SVD)
if using SVD:
	* Find the eigenvectors corresponding to the 6 largest eigenvalues
	* These are the 6 eigenfaces. Reshape to get the images.
if using covariance:
	* Find the eigenvectors corresponding to the 6 largest eigenvalues
	* Find Z for the 6 eigenvectors( as described in lectures). These are the 6 eigenfaces
	* Reshape to get the images
8) Repeat the steps for Subject 2
9) You will get 6 eigenfaces for Subject 1 and 6 eigenfaces for Subject 2



ISOMAP
------
1) Find the nearest neighbors of each datapoint within a specific distance (epsilon). So if a datapoint is within epsilon distance, it is considered a nearest neighbor,otherwise it is not
2) Find the shortest path distance matrix D between each pair of points. It will find the shortest path along the data cloud, which is a little different than finding just the shortest Euclidean distance, depending on the shape of the data.
3) Find low dimensional representation which preserves the distance information. This helps us unfold the data cloud and find the direct distances.
