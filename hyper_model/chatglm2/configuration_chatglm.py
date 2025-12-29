from transformers import PretrainedConfig


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)



class HLChatGLMConfig(ChatGLMConfig):
    def __init__(
            self,
            roles_num: int = 5,
            roles_emb_dim: int = 64,
            layers_num: int = 28,
            layers_emb_dim: int = 64,
            residual_blocks_num: int = 2,
            hyper_hidden_dim: int = 512,
            rank_dim: int = 8,
            alpha: int = 32,
            dropout=0.0,
            layernorm_input=False,
            layernorm_output=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.roles_num = roles_num
        self.roles_emb_dim = roles_emb_dim
        self.layers_num = layers_num
        self.layers_emb_dim = layers_emb_dim
        self.residual_blocks_num = residual_blocks_num
        self.hyper_hidden_dim = hyper_hidden_dim
        self.rank_dim = rank_dim
        self.alpha = alpha
        self.dropout = dropout
        self.layernorm_input = layernorm_input
        self.layernorm_output = layernorm_output
        

