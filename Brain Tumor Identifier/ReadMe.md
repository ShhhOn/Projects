OBJECTIVE
---------
Creating a one-size-fits-all AI model for MRI diagnosis can be challenging because MRI images can come in assorted sizes, different positioning of the head, with varying levels of contrast and background noise. If all images fed into the model are regularized, the accuracy of the model could potentially increase when seeing a fully new data set, thus expanding the scope of the model’s usability. We can utilize suggestions given by high performers of the Large Scale Visual Recognition Challenge to preprocess all images and possibly increase model performance

PROBLEM STATEMENT
-----------------
Identifying patients with brain tumors can be complicated, and with increasing demands on medical providers, there may be missed opportunities to diagnose patients. If diagnosed early enough, managing tumors through lifestyle changes or earlier invasive surgery can lead to a longer, healthier life for patients. Our project is to develop a process for regularizing images and detecting tumors, thus expediting the diagnosis process for a quicker medical response and relieving the resource burden in developing countries.

DATASETS
--------
Our project uses 2 datasets: the first dataset will be used for training and validation. The second dataset will serve as a testing dataset to ensure that our model fits appropriately and is not overfitted to the first dataset.

*https://www.kaggle.com/preetviradiya/brian-tumor-dataset
	*2513 brain tumor images, 2087 healthy brain images. Split the dataset into 80% training and 20% validation images. Here is an example of the data of both healthy brains and brains with tumors.

*https://www.kaggle.com/ahmedhamada0/brain-tumor-detection
	*3501 train images, 161 test images, 202 validation images. This will be the test dataset that we evaluate our models on.