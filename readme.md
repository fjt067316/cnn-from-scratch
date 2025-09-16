# CNN From Scratch

This is a cnn written in c++ single threaded cpu

## Data

MNIST Dataset loaded as csv file with each row being a image
Load data into vector size 28x28
reshape to 1x28x28

## Convolutional Layer

Input shape: 3d
output shape: 3d

feature map shape: (input_depth, filter_len, filter_len)
bias shape: (num_feature_maps)

multiple feature maps
1 feature map == 1 filter
1 bias per filter
each layer has a list of feature maps thus it has a 4d filter variable

each feature map is the same depth as the input
each feature map gets its own copy of the input, performs a convolution and creates a 2d output
the 2d outputs of each feature map are stacked ontop of eachother to create a 3d output

## Fully Connected Layer

Input shape: 1d
Output shape: 1d

each fcl has a weight matrix of shape (n_outputs , n_inputs)
each fcl has bias matrix of shape (n_outputs)

each row gets 1 copy if input and performs dot product 
each row has a bias added to it after dot product
