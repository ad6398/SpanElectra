from configuration_span_electra import SpanElectraConfig

hidden_size = 256
embedding_size = 512  # keep hidden and embedding size same for less layers
max_span_len = 20
mask_ratio = 0.2
vocab_size = 10000
max_seq_len = 512

pad_token = "[PAD]"
pad_token_id = 2
mask_token = "[MASK]"
mask_token_id = 3
lowercase = True  # CR
dummy_id = 0
ignore_label = 2

##################### Joint training args ##################


class Joint_trainDataArgs(object):
    def __init__(self):
        self.tokenizer_type = "BPE"
        self.vocab_file = "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-vocab.json"  # file path to vocab.json
        self.merges_file = "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-merges.txt"  # fp to merges.txt
        self.lowercase = lowercase
        self.max_seq_len = max_seq_len
        self.mask_feature_file = None
        self.tokenized_file = None
        self.raw_text_dir = "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/wiki.train.tokens"  # CR
        self.save_tokenized_text = True
        self.out_dir = "/media/data_dump/Amardeep/spanElectra/out/joint/"  # CR
        self.save_features = False
        self.mask_id = mask_token_id
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.mask_ratio = mask_ratio
        self.max_span_len = max_span_len
        self.save_name = "jointtrainData"
        self.occu = None


class Joint_validDataArgs(Joint_trainDataArgs):
    def __init__(self):
        super().__init__()
        self.raw_text_dir = "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/wiki.valid.tokens"  ##dir for raw text for validation
        self.save_name = "jointvalidData"
        self.occu = None
        self.tokenized_file = None


class JointTrainingConfig:
    gen_hidden_size = 128
    embedding_size = 256
    disc_hidden_size = 512

    use_SBO = True
    all_token_clf = True
    gen_config = SpanElectraConfig(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=gen_hidden_size,
        max_span_len=max_span_len,
        max_position_embeddings=max_seq_len,
        pad_token_id=pad_token_id,
        use_SBO=use_SBO,
        all_token_clf=all_token_clf,
    )
    disc_config = SpanElectraConfig(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=disc_hidden_size,
        max_span_len=max_span_len,
        max_position_embeddings=max_seq_len,
        pad_token_id=pad_token_id,
        use_SBO=use_SBO,
        all_token_clf=all_token_clf,
    )

    device_ids = [0, 1, 2]  # list of devices you want to use
    num_workers = 0
    save_dir = "/media/data_dump/Amardeep/spanElectra/out/joint/"  # CR
    checkpoint_path = None

    epochs = 2
    learningRate = 4e-5
    train_batch_size = 9
    valid_batch_size = 9

    ignore_label = pad_token_id
    pad_token_id = pad_token_id


#############################################################

#################################Generator (2stage train) model config###################


class MLM_trainDataArgs(object):
    def __init__(self):
        self.tokenizer_type = "BPE"
        self.vocab_file = "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-vocab.json"  # file path to vocab.json
        self.merges_file = "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-merges.txt"  # fp to merges.txt
        self.lowercase = lowercase
        self.max_seq_len = max_seq_len
        self.mask_feature_file = None
        self.tokenized_file = None
        self.raw_text_dir = "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/wiki.train.tokens"  # CR
        self.save_tokenized_text = True
        self.out_dir = "/media/data_dump/Amardeep/spanElectra/out/2s_tri/"  # CR
        self.save_features = True
        self.mask_id = mask_token_id
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.mask_ratio = mask_ratio
        self.max_span_len = max_span_len
        self.save_name = "MLMtrainData"
        self.occu = 1000


class MLM_validDataArgs(MLM_trainDataArgs):
    def __init__(self):
        super().__init__()
        self.raw_text_dir = "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/wiki.valid.tokens"  ##dir for raw text for validation
        self.save_name = "MLMvalidData"
        self.tokenizer_type = "BPE"
        self.occu = 1000
        self.tokenized_file = None


class MLM_trainingConfig:
    modelConfig = SpanElectraConfig(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        pad_token_id=pad_token_id,
        max_span_length=max_span_len,
        max_position_embeddings=max_seq_len,
    )
    device_id = 0
    num_workers = 0
    epochs = 3
    learningRate = 4e-5
    train_batch_size = 8
    valid_batch_size = 8
    ignore_label = pad_token_id
    save_dir = "/media/data_dump/Amardeep/spanElectra/out/2s_tri/"  # CR


##################################################################

#################################discrimnator(2stage train) args################
class SE_trainDataArgs:
    def __init__(self):
        self.generator_path = "/media/data_dump/Amardeep/spanElectra/out/2s_tri/MLMmodel_2"  # fp of genrattor model

        self.tokenizer_type = "BPE"
        self.vocab_file = "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-vocab.json"  # file path to vocab.json
        self.merges_file = "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-merges.txt"  # fp to merges.txt
        self.lowercase = lowercase
        self.max_seq_len = max_seq_len
        self.mask_feature_file = None
        self.tokenized_file = "/media/data_dump/Amardeep/spanElectra/out/2s_tri/tokenized_textMLMtrainData.p"
        self.raw_text_dir = None  # CR
        self.save_tokenized_text = True
        self.out_dir = "/media/data_dump/Amardeep/spanElectra/out/2s_tri/"  # CR
        self.save_features = True
        self.mask_id = mask_token_id
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.mask_ratio = mask_ratio
        self.max_span_len = max_span_len
        self.save_name = "SEtrainData"
        self.occu = 1000


class SE_validDataArgs(SE_trainDataArgs):
    def __init__(self):
        super().__init__()
        self.raw_text_dir = None  ##dir for raw text for validation
        self.save_name = "SEvalidData"
        self.occu = 1000
        self.tokenized_file = "/media/data_dump/Amardeep/spanElectra/out/2s_tri/tokenized_textMLMvalidData.p"


class SE_trainingConfig:
    modelConfig = SpanElectraConfig(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        pad_token_id=pad_token_id,
        max_span_length=max_span_len,
        max_position_embeddings=max_seq_len,
        ignore_label=ignore_label,
        use_SBO=True,
    )
    device_id = 1
    num_workers = 0
    epochs = 1
    learningRate = 4e-5
    train_batch_size = 8
    valid_batch_size = 8
    ignore_label = pad_token_id
    save_dir = "/media/data_dump/Amardeep/spanElectra/out/2s_tri/"  # CR
    worker = 0
    generator_weight_path = (
        "/media/data_dump/Amardeep/spanElectra/out/2s_tri/MLMmodel_wight2.pt"
    )


################################################################
