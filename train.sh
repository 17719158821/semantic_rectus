#python3 train.py --exp=exp_withoutGamma --num_epochs=300 --do_GammaTransform=False
#python3 train.py --exp=exp_withGamma --num_epochs=300 --do_GammaTransform=True
#python3 train.py --exp=exp_withGause --num_epochs=300 --do_GaussianNoise=True
#python3 train.py --exp=exp_withRC_Gamma --num_epochs=300 --do_random_crop=True --do_GammaTransform=True
python3 train.py --exp=exp_withRC_RT_Gause_E_S_1200 --num_epochs=1200 --do_random_crop=True --do_rotation=True --do_GaussianNoise=True --do_elastic_deform=True --do_scale=True
#python3 train.py --exp=exp_withRC_RT_Gause_E_S_P2 --num_epochs=200 --do_random_crop=True --do_rotation=True --do_GaussianNoise=True --do_elastic_deform=True --do_scale=True --augmentation_prob=0.2
