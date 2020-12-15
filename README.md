# real-state-10k

This is the real state 10k dataset from https://google.github.io/realestate10k, just in an easier way to download.

The camera trajectories for RealEstate10K are on the folder [RealEstate10K](https://github.com/Findeton/real-state-10k/main/RealEstate10K) 

The data consists of a set of .txt files, one for each video clip, specifying timestamps and poses for frames in that clip. For a learning application, frames can be sampled from the training clips in order for learning, for instance, a view synthesis model. In Google's 2018 SIGGRAPH paper [Stereo Magnification: Learning view synthesis using multiplane images](https://ai.google/research/pubs/pub46965), for example, triplets of frames were sampled from each clip during training, two for predicting a model, and a third held out as ground truth for computing a view synthesis loss used to train the network.

This data is licensed by Google LLC under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

# Dataset Design

The data is split into train and test subdirectories, each with a set of .txt files, one .txt file for each video clip (about 90% of the clips are in train, and the remaining 10% in test). The format of each .txt file is as follows:

  <Video URL>
  <frame1>
  <frame2>
  <...>

where each frame line has the following 19 columns:

     1. timestamp (int: microseconds since start of video)

   2-6. camera intrinsics (float: focal_length_x, focal_length_y,
                                  principal_point_x, principal_point_y)

  7-19. camera pose (floats forming 3x4 matrix in row-major order)

The camera intrinsics can be organized into a 3x3 matrix K and the camera pose parameters into a 3x4 matrix P = [ R | t ], such that the matrix KP maps a (homogeneous) 3D point p in a world coordinate frame to a (homogeneous) 2D point in the image.

The camera intrinsics are expressed in resolution-independent normalized image coordinates, where the top left corner of the image is (0,0), and the bottom right corner of the image is (1,1). This allows for the intrinsic parameters to be applied to frames at whatever resolution they are represented on disk (or resized to prior to training), by scaling them according to the image size in pixels. For an image of resolution width x height pixels, the intrinsics matrix at the actual scale of the image is

K = [
      [ width * focal_length_x, 0,                       width * principal_point_x ],
      [ 0,                      height * focal_length_y, height * principal_point_y],
      [ 0,                      0,                       1                         ]
]

