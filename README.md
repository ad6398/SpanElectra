# SpanElectra
New and Public repo to train spanELectra. SpanElectra uses 

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

### matrices and loss
loss-> cross entropy
accu -> flat accu
