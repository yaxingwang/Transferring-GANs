# Transferring GANs generating images from limited data
# Abstract: 
Transferring the knowledge of pretrained networks to new domains by means of finetuning is a widely used practice for applications based on discriminative models. To the best of our knowledge this practice has not been studied within the context of generative deep networks. Therefore, we study domain adaptation applied to image generation with generative adversarial networks. We evaluate several aspects of domain adaptation, including the impact of target domain size, the relative distance between source and target domain, and the initialization of conditional GANs. Our results show that using knowledge from pretrained networks can shorten the convergence time and can significantly improve the quality of the generated images, especially when the target data is limited. We show that these conclusions can also be drawn for conditional GANs even when the pretrained model was trained without conditioning. Our results also suggest that density may be more important than diversity and a dataset with one or few densely sampled classes may be a better source model than more diverse datasets such as ImageNet or Places.

# Overview 
- [Dependences](#dependences)
- [Installation](#installtion)
- [Instructions](#instructions)
- [Results](#results)
- [References](#references)
- [Contact](#contact)
# Dependences 
- Python2.7, NumPy, SciPy, NVIDIA GPU
- **Tensorflow:** the version should be more 1.0(https://www.tensorflow.org/)
- **Dataset:** lsun-bedroom(http://lsun.cs.princeton.edu/2017/) or your dataset 

# Installation 
- Install tensorflow
- Opencv 
# Instructions
- Using 'git clone https://github.com/yaxingwang/Transferring-GANs'

    You will get new folder whose name is 'Transferring-GANs' in your current path, then  use 'cd Transferring-GANs' to enter the downloaded new folder
    
- Download pretrain models[Google driver](https://drive.google.com/file/d/1e7Pw-m-DgAiB_aQnNUUwBRVFc2izRiRw/view?usp=sharing); [Tencent qcloud](https://share.weiyun.com/5mBsISh)

    Uncompressing downloaded folder to current folder, then you have new folder 'transfer_model'  which contains two folders: 'conditional', 'unconditional', each of which has four folders: 'imagenet', 'places', 'celebA', 'bedroom'

- Download dataset or use your dataset.

    I have shown one example and you could make it with same same form.

- Run 'python transfer_gan.py'

   Runing code with default setting. The pretrained model can be seleted by changing the parameter 'TARGET_DOMAIN'
 
- Conditional GAN 
  If you are interested in using conditional model, just setting parameter 'ACGAN = True'
# Results 
Using pretrained models not only get high performance, but fastly attach convergence. In following figure, we show conditional and unconditional settings.
<br>
<p align="center"><img width="100%" height='60%'src="results/FID.png" /></p>



# References 
- \[1\] 'Improved Training of Wasserstein GANs' by Ishaan Gulrajani et. al, https://arxiv.org/abs/1704.00028, (https://github.com/igul222/improved_wgan_training)[code] 
- \[2\] 'GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium' by Martin Heusel  et. al, https://arxiv.org/abs/1706.08500, (https://github.com/bioinf-jku/TTUR)[code]
# Contact

If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: yaxing@cvc.uab.es
