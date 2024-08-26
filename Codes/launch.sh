# seeds=$1
seeds=0

python generate_3D_landmarks.py --seeds=${seeds}

python gen_warped_images.py --outdir=out --network=ffhqrebalanced512-128.pkl --seeds=${seeds}

# python gen_inversion_wraped_images.py --outdir=out --network=ffhqrebalanced512-128.pkl --seeds=${seeds}
python gen_inversion_wraped_images.py --outdir=out --network=ffhqrebalanced512-128.pkl --edit_exp=True --directions=./editing_direction/fat.npy --seeds=${seeds}
# python gen_inversion_wraped_images.py --outdir=out --network=ffhqrebalanced512-128.pkl --edit_shape=True --directions=./editing_direction/fat.npy --seeds=${seeds}

