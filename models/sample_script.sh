python3 main.py -dataset reddit -model stacked_lstm --eval-every 5 --num-rounds 100 --clients-per-round 10 --batch-size 5 -lr 2.0
python3 main.py -dataset femnist -model cnn -lr 0.01 --batch-size 10 --num-epoch 5 --num-round 100 --eval-every 5 --clients-per-round 10
python3 main.py -dataset celeba -model cnn -lr 0.01 --batch-size 10 --num-epoch 5 --num-round 100 --eval-every 5 --client-per-round 10
