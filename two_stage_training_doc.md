# Two Stage Training

### Create feature for discriminator w/o training Generator
Create feature from raw text file , split them in training and valid set for training Discriminator. By default this script uses BertTokenizer from Hugging face of model `bert-base-uncased`, So it expect vocab file of `bert-base-uncased` model which can be downloaded from [s3 storage here](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) .  run following command for feature creation required to train Discriminator.

```python
python create_OD_feature.py \
--vocab_file "/media/data_dump/Amardeep/bert-base-uncased-vocab.txt"  \ #path to vocab file as mentioned above
--out_dir "/media/data_dump/Amardeep/test_fol/od_bert" \ #output director to store feature file created
--in_dir "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/" \ # input director containg raw text files to create feature
--max_seq_len 512 \
--workers 10 \
--chunk_size 100000 \
--valid_split 0.3
```

## Training Discriminator only
To train discriminator only create feature from above step. store training arguments in a `my_config.json` file. we can inintialize embedding with a pretrained weight, to do so pleases keep configuration parameter in mind, and initialize weight only with suitable weight, for example by default hidden_size =768 and vocab size = 30522, so we can initialize out weight with that of `bert-base-uncased` embedding.

```python
python train_disc_only.py \
--config_file "/home/amardeep/spanElectra/keyword-language-modeling/configs/od_bert_base_config.json" \ #path to config.json file containig model related arguments,`configs/my_config.json`
--train_file "/media/data_dump/Amardeep/test_fol/od_bert/train.bin" \ #path to feature file of training data, created above
--valid_file "/media/data_dump/Amardeep/test_fol/od_bert/valid.bin" \ #path to feature file of valid data, created above
--out_dir "/media/data_dump/Amardeep/test_fol/od_bert/" \ #output directory to store model outputs
--train_batch_size 8 \
--valid_batch_size 8\
--workers 0 \ # keep this as 0 only
--epochs 1 \
--lr 3e-5 \
--device_ids 1 \#list ids of GPU device to use, mention multiple ids for multi GPU. example "--device ids 0 1 2" for using 3 gpu with these ids
-- embedding_path "bert_base_uncased_embedding_weight.pt" #file path to weight to a pretrained embedding
```



