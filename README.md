# Disparity-Maps
Given stereo image pairs (three stereo pairs are provided in   https://drive.google.com/drive/folders/1pdDkFpa59m4A02pLkUqI1H2Zb_GerLkK?usp=sharing  ),  you are required to write a program with C++ or Python or Matlab to compute the the Disparity Maps of each left (view1) image.    

# How to run 
1. Check rectification of stereo image: ".\PSNR_Assignment2\PSNR_Python\preprocess.py"
2. OpenCV SGBM algorithm + Weighted Least Squares filter: ".\PSNR_Assignment2\PSNR_Python\opencv.py"
3. PSNR calculation: ".\PSNR_Assignment2\PSNR_Python\psnr_cal.py"

#Peak-SNR(PSNR) Result
psnr_cal.py::test PASSED

The Peak-SNR value of Art is %0.4f 
 20.330918632895468

The Peak-SNR value of Dolls is %0.4f 
 27.358401273490426

The Peak-SNR value of Reindeer is %0.4f 
 16.011420214174002
