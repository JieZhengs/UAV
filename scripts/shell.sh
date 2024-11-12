python main.py --config ./config/train.yaml --work-dir ./work_dir/train -model_saved_name ./runs/train --device 0 --batch-size 1 --test-batch-size 1 --warm_up_epoch 0 --only_train_epoch 100
python main.py --config ./config/test.yaml --work-dir ./work_dir/test -model_saved_name ./runs/test --device 0 --batch-size 16 --test-batch-size 16 --weights ./runs/train_td_joint247.pt
