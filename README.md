# SpanElectra
SpanElectra is a compute optimized span level Language model for pre training and fine tuning tasks. Instead of training a model that generate tokens, we train a discriminative model that predicts whether each token in the corrupted input was replaced by a generator sample or not. It has two parts: 

![alt text](https://github.com/ad6398/SpanElectra/blob/main/assets/se-arch.png?raw=true)

## Generator:
* It is a small span level MLM model like spanBERT.
* Span Boundary Generative Objective (SBGO):
   * Jack		[mask_1]		[mask_2] 		[mask_3] 		hill
     * token1  =  function( rep(Jack),  rep( hill), pos(mask_1) )
     * token2  =  function( rep(Jack),  rep( hill), pos(mask_2) )

     rep = representation function of tokens in MLM
     
     pos= position of token wrt start of span or positional embedding
     
## Discriminator:
* Span Boundary Predictive Objective(SBPO):
  * Jack 		go 		at 		the 	hill
    
    is_replaced(go) = function( rep(Jack), rep(hill), pos(go) )

* Electra Objective/ all token objective (ATO):
  * predict whether each token present in sentence is replaced/corrupted or not.
  * Jack 		go 		at 		the 	hill
    
    is_replaced(Jack) = function( rep(whole_sentence ))


### Why SpanElectra?
* **More accurate:** Shifting from token level to span level produced better result as evident from the spanBERT paper. For both generative and predictive part,  we have span level objective (SBGO and SBPO) as well as token level objective (ATO), clearly promising a better result.
* **Needs less resources:** Generative models take more time and computing resources than predictive model (easily evident from Electra paper).
Generative part of spanElectra( i.e. generator) is small in size in comparison to predictive part (Discriminator) and other MLMs like BERT, hence making spanElectra compute and time optimized.



## Tokenization and Vocab Creation
`create_vocab_SE.py` will create a Vocabulary file based on the type of tokenizer we choose on given text Data. `the following class contains all arguments needed to develop vocabulary.
  
  create_vocab_SE.py >> class tok_args:
    
    tokenizer_type = 'BPE'  # type of tokenizer ByteLevel ('BPE') or word piece Tokenizer ('BERT).
    
    data_dir = "../input/wikitextv1/wikitext-2/"   # dir containing text(.txt) files to build vocab
    
    model_name= 'trial' # name of model
    
    min_fre =2 # min frequency of a word so as to be included in vocab
    
    vocab_size= 10000 #size of vocab
    
    out_dir = '/kaggle/working/tok' #dir to save vocab.json or merges.txt
    
 
 Make changes in the above class and run `create_vocab_SE.py` to build vocab.
    
    
### Arguments
`argument.py` contains all argument for training MLM_panElectra(spanBert), spanElectra, creating training and valid datasets.

### Masking
`masking.py` contains a method to get masked spans and fake spans from a span-level Language model

### Feature Creations
`process.py` creates MLM feature(i.e to train MLM spanElectra or generator) from raw text and feature for SpanElectra(discriminator). It also contains classes for creating Datasets and saving features.

### Models
`modelling_span_electra.py` has all MLM, generator, and discriminator model definitions.

### Training
`trainMLM_SpanElectra.py` will train a generator model and save it. Later this saved weight will be used to initialize the discriminator. it will use MLM_trainDataArgs (to create training features), MLM_trainingConfig(arguments to train MLM model/generator), MLM_validDataArgs(to create validation features) arguments from argument.py. 

`trainSpanElectra.py` will train discrimnator model. it will use SE_trainDataArgs (to create training features), SE_validDataArgs (to create validation features), SE_trainingConfig (args to train spanElectra model).

