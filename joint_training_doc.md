# Joint Training 
A tutorial for Training SpanElectra by joint method. In this method we train generator and Discriminator all together. According to value of parameters like embedding size, hidden size, attention layer, etc they will be sharing as much layer as possible.
steps would include 1) create vocab files by tokenizing all the text 2) setting training argument in argument.py file 3) training. For easy training create dir as suggested below

```
--wikitext
  |  
  |-1- data_dir
  |     
  |-2- vocab
  |     |-2.1- 10k_wikitext
  |           
  |-3- models
        |-3.1- joint
 ```

`data_dir` should contain only text files (train.txt, valid.txt, etc.). make sure there is no hidden folder.

## 1. vocab building
for vocabulary creation run following
```python
python create_vocab_SE.py \
--data_dir "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/" \ #path to dir containing text(.txt) files to build vocab **(1)**
--out_dir '/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k' \  #dir to save vocab.json and merges.txt **(2.1)**
--tokenizer_type "BPE" \ # type of tokenizer ByteLevel ('BPE') or word piece Tokenizer ('BERT).
--min_fre 2 \ # min frequency of a word to be included in vocab
--vocab_size 10000 \  #size of vocab
--model_name "trial"   # name of model
```
this will create Voabulary files in `out_dir(2.1)` based on type of the tokenizer we choose on given text Data. "BPE" will create vocab.json (vocab_file) and merges.txt (merge_file)

We have to create these vocab files only once unless we want to change vocab size. It's not necessary to perform this step every time you run same model.

Total **training process flow** like this:
1. *feature creation from training dataset and validation dataset* : from training raw text data, sentences are tokenized and converted into chunks of seqence length b/w 1 and max_seq_size randomly(biased towards max_seq_len). These tokenized line are saved in a file. span Masking are done on these sentences with help of a geometric distribution and `mask_ratio` factors. This step is done by `crete_feature_SE.py` script.

2. *LM model training*: after feature creation LM models are trained, argument and configuration of same are handeled by `config.json` which contain configuration reated to LM model.

## 2. feature creation
do following step for train and vaildation data to create feature.
```python
python create_feature_SE.py \
--tokenizer_type "BPE" \  # type of tokenizer ByteLevel ('BPE') or word piece Tokenizer ('BERT).
--vocab_file "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-vocab.json" \ # file path to vocab.json created during vocab building
--merges_file "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-merges.txt" \ # file path to merges.txt created during vocab building
--in_file "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/wiki.train.tokens" \ #file path to train/valid text data *NOT* to directory for which train/valid features will be created
--out_file "/media/data_dump/Amardeep/spanElectra/out/jfeat/train.txt" \  #file path to store features created out of in_file
--max_seq_len 512 \ # max sequence length of features
--workers 30 \  #num of worker in multi processing, set according to your machine capabilities
--chunk_size 1000000  # num of sentences passed to single process in multi processing, set according to your machine capabilities
```
## 3. setting LM model config
`configs/default.json` represent default configuration of the model. you can copy this file and paste in this same directory by renaming this, lets name this config file as `my_config.json`. Do suitable changes in `my_config.json` as per model specification you want to train.

## 4. training
```python
python joint_train_span_electra.py \
--config_file "/home/amardeep/spanElectra/keyword-language-modeling/configs/default.json" \ #path to config.json file containig model related arguments,`configs/my_config.json` created in step 3
--train_file "/media/data_dump/Amardeep/spanElectra/out/jfeat/train.txt" \ #path to feature file of training data, created using step 2
--valid_file "/media/data_dump/Amardeep/spanElectra/out/jfeat/valid.txt" \ #path to feature file of validation data, created using step 2
--out_dir "/media/data_dump/Amardeep/spanElectra/out/jfeat/" \ #output directory to store model outputs
--train_batch_size 8 \
--valid_batch_size 8\
--workers 8 \
--epochs 2 \
--lr 3e-5 \ # learning rate of model
--device_ids 0 \ #list ids of GPU device to use, mention multiple ids for multi GPU. example "--device ids 0 1 2" for using 3 gpu with these ids

```
**optional arguments**
in *step 4* provide function argument as per reuirements only, otherwise don't. 
```python
--train_occur 10000 \ #only if you don't want to use whole trainig data then specify number of datapoints/features you want to use 
--valid_occur 1000 \ #only if you don't want to use whole validation data then specify number of datapoints/features you want to use for validation
--checkpoint_path "/media/checkpoint.pt" #if you want resume a training from a check point then specify this, it will load a model from this check point
```

## Old training Method

follow above vocab building step then follow below steps

## config and arguments for joint training

after creating vocab do following changes in `argument.py` file. only do following recommended changes. you can leave other to default.

1. Firstly set in general argument as per model size and requirement. Unlike BERT we have *different hidden size and embedding size*, so that discriminator and generator could share embedding layers between them and at the same time they could have different size. Remember we need genrator size (i.e generator hidden size) 1/2 or 1/4 that of size of dicriminator (i.e discriminator hidden size). 
```python
hidden_size= 256  # hidden size of encoders, for joint training we have discriminator hidden size as well generator hidden size
embedding_size= 512 # keep hidden and embedding size same to avoid projection layer
max_span_len= 20  # max span len to be masked
mask_ratio= 0.2 # mask ration of total text
vocab_size= 10000 # vocab size of tokenizer you would be using
max_seq_len = 512 
```

2. set training data args (`class Joint_trainDataArgs`):
  
  2.1. set tokenizer arguments, specify tokenizer type and path to vocab.json and merges.txt(if using BPE). These files were created in *vocab builiding* step.

``` python
    self.tokenizer_type ='BPE' #choose tokenizer type ('BPE' or 'BERT')
    self.vocab_file= '/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-vocab.json' # file path to vocab.json
    self.merges_file= '/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-merges.txt' #file path to merges.txt   
```
    
  2.2. specify location of training text data, output directory to store output.

```python
    self.raw_text_dir = '/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/wiki.train.tokens' #file path to train text data *NOT* to directory
    self.out_dir = '/media/data_dump/Amardeep/spanElectra/out/joint/' # location of directory where you want to store outputs by model.
```
    
  2.3.(optional) set argument to save tokenized/masked feature. by setting these argument you can specify to save features in a pickle file at each step of the feature creation.
  
  ```python
  self.save_tokenized_text= True # if true save all the features just after toeknization (recommended)
  self.save_features = False #if true save features after masking i.e final features (this will create huge pickle file, recommended to save tokenized feature only)
  ```
  
  we can load these saved feature directly next time.
 
3. set validation data args(`class Joint_validDataArgs`)
```python
self.raw_text_dir = '/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/wiki.valid.tokens' ##file path to raw text for validation
```
4. set training model arguments and config(`class JointTrainingConfig`)

4.1 set generator and descriminator config by setting 
```python
    gen_hidden_size = 128 #genrator hidden size
    embedding_size = 256 # embedding size
    disc_hidden_size = 512 #discriminator hidden size
```
4.2 set save directory(`save_dir`) to save stats and model realted output, if you have multiple gpu and want to use a specific gpu than set `device_id`(if you have single gpu make sure to set it as 0) as per numbering of gpu. set `num_workers` for multiple worker(recommended value is 0).

4.3 *model training specific config*
```python
    epochs = 1 #number of epoch you want to run
    learningRate = 4e-5 # learning rate
    train_batch_size = 8 #training batch size
    valid_batch_size = 8 #validation batch size
```
finally run `python jointTrainSpanElectra.py` to start training
