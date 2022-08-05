# Course_ComputerVision
電腦視覺(英文授課) Computer Vision, in NCTU

There are 3 homeworks(projects)
- [Course_ComputerVision](#course_computervision)
  - [Environment](#environment)
- [HW1: Photometric Stereo](#hw1-photometric-stereo)
- [HW2: Image Stitching](#hw2-image-stitching)
- [Final Project: Depth Estimation from Stereo Images](#final-project-depth-estimation-from-stereo-images)

## Environment
dependency:  
`python==3.6.13`
```python
matplotlib==3.3.4
numpy==1.19.5
open3d==0.15.1
opencv-python==4.5.5.62
scipy==1.5.4
tqdm==4.63.0
```
maybe that's all(?

# HW1: Photometric Stereo
Introduction:  
Given many image taken(rendered) from **same** viewpoint
with **different** light directions(LightSource.txt).  
Try to estimate the depth(height).

| pic1 | pic2 | pic3 | ... |
|------|------|------|-----|
|![](HW1_Photometric_Stereo\test\bunny\pic1.bmp)|![](HW1_Photometric_Stereo\test\bunny\pic2.bmp)|![](HW1_Photometric_Stereo\test\bunny\pic3.bmp)|...|

Result:  
![](HW1_Photometric_Stereo\test\bunny_result_example.jpg)

Detail:  
Use `Diffuse reflection` method to simulate the image color.  
Calculate it backward to get the height.
$$
I_{diffuse} = consant * (n \cdot l) \\
n: \text{normal vector} \\
l: \text{light direction vector}
$$
1. Estimate every normal vector in `image[i, j]`.
   $$l = norm(I_{diffuse} \times l_{inv}),\ l_{inv}: inverse(l)$$
2. Estimate every Gradient vector in (x, y) directions in `image[i, j]`  
   by normal vectors.
3. Estimate the surface height by each Gradient vector.

# HW2: Image Stitching
Introduction: 
Result: 
Detail: 

# Final Project: Depth Estimation from Stereo Images
Introduction: 
Result: 
Detail: 
