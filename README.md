Depth Estimation using MiDaS Model
This project utilizes the MiDaS model to perform depth estimation from a single camera input. MiDaS is a deep learning-based model capable of generating depth maps from images.

Required Libraries
The project requires the following Python libraries:

torch
cv2 (OpenCV)
numpy
matplotlib
You can install the required libraries using the following command:
   pip install torch opencv-python numpy matplotlib

Usage
Update the model_type variable at the beginning of the code with the model type you wish to use.
Set the path for the image you want to analyze in the img variable
Run the Python script.

Output
When executed, the code will display the depth map predicted from the input image using the plasma colormap.

Example Output
When the code is run, a depth map similar to the following will be displayed:

 (An example image can be added)

Notes
The code checks for GPU availability and utilizes it if available.
The image is resized according to the dimensions returned from the model prediction.

License
Please check the MiDaS license when using this project.
