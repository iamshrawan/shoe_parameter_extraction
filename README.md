# Shoe Parameter Extraction

# Starter Command
Use the following command to train a model
```bash
   python train.py -a resnet18 -d dataset_folder_path --epochs num_epochs -b batch_size --finetune --seed 2022 --exp exp_name --entity wandb_username --sample_interval sample_interval  --cuda --mv
```
- Replace *dataset_folder_path* with path to dataset folder
- Replace *num_epochs* (integer)
- Replace *batch_size* with number of examples to load for each gradient descent step (integer)
- Replace *exp_name* with experiment name. This option is used to track the experiment in Weights and Biases https://wandb.ai
- Replace *wandb_username* with your Weights & Biases username
- Replace *sample_interval* with number of intervals to sample predictions on validation set. This will be uploaded to Weights and Biases.

--finetune is used to load pretrained model. Its certain layers are frozen, and rest of the network is trained on the shoe dataset. --mv is used to train in the multi-view mode. --cuda option is used to train on gpu. If you want to train on different gpu, change the gpu id in 'cuda:0' of line 69 to the required gpu id as 'cuda:gpu_id'.
Explore the code and other parameters as well.

After training, the predictions on test set are stored in test_{exp_name}.json file in the format of 'idx':[predictions, ground_truth]. 'idx' refers to an index of the dataset. To view predictions, follow the code in [test.ipynb](test.ipynb)
Try generating plots like the right hand side plots in slide number 16.


Dataset can be found in: https://drive.google.com/drive/folders/1VQZc9kWp-_MYTv8p62Db4JjXElV4v3xh?usp=sharing

