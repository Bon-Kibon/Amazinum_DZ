Overview:

I used model developed in lesson 26, which decides whether person is happy or not.

Deployment info:

docker pull yourdockerhubusername/simple-cv-app:latest
docker run -p 5000:5000 yourdockerhubusername/simple-cv-app:latest
Go to this address: http://127.0.0.1:5000/
Modeling info:

It's a simple cnn with one convolutional layer for binary classification.
	
	Dataset info:

&nbsp;	    X\_train shape: (600, 64, 64, 3)

&nbsp;	    Y\_train shape: (600, 1)

&nbsp;	    X\_test shape: (150, 64, 64, 3)

&nbsp;	    Y\_test shape: (150, 1)

&nbsp;	Scores:

&nbsp;	  Loss = 0.299

&nbsp;	  Test Accuracy = 0.946



Interface description:

'/' -- GET -- renders index.html -- main page with enter field for 

&nbsp;				    image index which prediction 

&nbsp;				    you want to receive


'/results' -- GET -- requests <photo\_id> -- renders results.html -- shows ground\_truth and 

&nbsp;								    predict for <photo\_id>

'/image\_<photo\_id>' -- GET -- returns .jpg -- returns original image by the index

