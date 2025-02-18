# torchRIR
A [PyTorch](https://pytorch.org/) based library for computing [Room Impulse Responses](https://paperswithcode.com/task/room-impulse-response). This would allow backpropagating through RIR data ([RT60](https://en.wikipedia.org/wiki/Reverberation#Sabine_equation), [room modes](https://en.wikipedia.org/wiki/Room_modes)) on fast simulations performed in [CUDA](https://developer.nvidia.com/cuda-toolkit). 

## v0.1.0 Roadmap
-[x] Base Classes
  -[x] Source
  -[x] Ray
  -[x] Patch
  -[x] Room / ConvexRoom
-[ ] Image Source Method
  -[x] Batched version of Mirror Points on Walls
  -[ ] RIR For Convex Room
-[ ] Ray Tracer
  -[ ] 


References:
-  Savioja, L., & Svensson, U. P. (2015). Overview of geometrical room acoustic modeling techniques. The Journal of the Acoustical Society of America, 138(2), 708–730. doi:10.1121/1.4926438  https://doi.org/10.1121/1.4926438 