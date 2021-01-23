# Style Transfer by Rigid Alignment

This is a pytorch implementation of the paper [Style Transfer by Rigid Alignment in Neural Net Feature Space](https://arxiv.org/abs/1909.13690)


## How To Use

Download the repository into your system

`python styletransfer.py --content path/to/content/image --style path/to/style/image --output path/to/output/image`

Here is an example

`python styletransfer.py --content 'inputs/contents/cold.jpg' --style 'inputs/styles/sketch.png' --output 'outputs/cold_sketch.jpg'

### Additional Flags

`--alpha` Choose the value of alpha

`--pretrained_path` Path to pretrained models (The names should be as it is in the pretrained_models directory)

### To Add Additional Style

`--additional_style_flag` Set this to True

`--beta` Choose the value of beta

`--style1` Set the path of the additional style image


## Results

![result1](examples/result1.png)

![result2](examples/result2.png)

![result3](examples/result3.png)

![result5](examples/result5.png)

![result7](examples/result7.png)

![result8](examples/result8.png)

## Variation of alpha

![result4](examples/result4.png)

![result6](examples/result6.png)

![sample_alpha](examples/sample_alpha.png)

![result_alpha](examples/result_alpha.png)

## Variation of beta

![sample_beta](examples/sample_beta.png)

![result_beta](examples/result_beta.png)
