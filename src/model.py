import jittor as jt
from jittor import nn
import math

class BertConfig:
    def __init__(
        self,
        architectures=["BertForMaskedLM"],
        attention_probs_dropout_prob=0.1,
        gradient_checkpointing=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        model_type="bert",
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=0,
        position_embedding_type="absolute",
        transformers_version="4.6.0.dev0",
        type_vocab_size=2,
        use_cache=True,
        vocab_size=30522,
    ):
        self.architectures = architectures
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.gradient_checkpointing = gradient_checkpointing
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.transformers_version = transformers_version
        self.type_vocab_size = type_vocab_size
        self.use_cache = use_cache
        self.vocab_size = vocab_size

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", jt.arange(config.max_position_embeddings).unsqueeze(0))
        
    def execute(self, input_ids, token_type_ids=None):
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = jt.zeros_like(input_ids)
        
        word_embed = self.word_embeddings(input_ids)
        position_embed = self.position_embeddings(position_ids)
        token_type_embed = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embed + position_embed + token_type_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    # 分头
    def transpose_for_scores(self, x):
        # 输入 (batch_size, seq_len, all_head_size)
        new_shape = (x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        x = x.view(new_shape)
        # 输出 (batch_size, num_heads, seq_len, head_dim)
        return x.permute(0, 2, 1, 3)

    def execute(self, hidden_states, attention_mask=None):
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        attn_scores = jt.matmul(Q, K.transpose(0,1,3,2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_scores = attn_scores + (attention_mask * -1e4)

        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        context = jt.matmul(attn_probs, V)
        context = context.permute(0, 2, 1, 3).view(hidden_states.shape[0], hidden_states.shape[1], -1)
        return context

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def execute(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attn_output = self.output(self_output, hidden_states)
        return attn_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def execute(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        
    def execute(self, hidden_states, attention_mask=None):
        attn_output = self.attention(hidden_states, attention_mask)
        inter_output = self.intermediate(attn_output)
        layer_output = self.output(inter_output, attn_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def execute(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

# 原始论文中bert的pooler是取[cls] with MLP
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def execute(self, hidden_states):
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        pooled = self.activation(pooled)
        return pooled

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        
    def execute(self, input_ids, token_type_ids=None, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoder_output)
        
        return encoder_output, pooled_output

class BertForCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
    
    def execute(self, input_ids, token_type_ids=None, attention_mask=None):
        encoder_output, pooled_output = self.bert(input_ids)
        return encoder_output, pooled_output


config = BertConfig()
if __name__ == "__main__":
    model = BertForCL(config)
    params_dict = jt.load('ckpt/pytorch_model.bin')
    # params_dict['bert.embeddings.LayerNorm.weight'] = params_dict.pop("bert.embeddings.LayerNorm.gamma")
    params_dict = {k.replace("LayerNorm.gamma", "LayerNorm.weight"): v for k, v in params_dict.items()}
    params_dict = {k.replace("LayerNorm.beta", "LayerNorm.bias"): v for k, v in params_dict.items()}
    model.load_state_dict(params_dict)
    # print(model)