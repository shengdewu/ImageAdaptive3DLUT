## trilinear

### build cpp trilinear

By default, we use pytorch 1.x:

    cd trilinear/cpp/torch_1_x
    sh setup.sh
    
For pytorch 0.4.1:

    cd trilinear/cpp/torch_0_4_1
    sh make.sh

### Use cpp trilinear 
    from .cpp.TrilinearInterpolationFunction import TrilinearInterpolationFunction
      
### Use python trilinear 
    from .python.TrilinearInterpolationFunction import TrilinearInterpolationFunction