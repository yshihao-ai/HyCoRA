import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List
from loguru import logger
class Role_ids:
    def __init__(self, 
        value: Optional[torch.Tensor] = None
    ):
        self.value = value
    def set_role_ids(self, value):
        self.value = value

class Dropout(torch.nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def extra_repr(self) -> str:
        return "p={}".format(self.p)

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    # nn.init.constant_(m.weight, 0.0)
    # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
    # nn.init.norm_(m.weight, mean=0, std=1e-4)

    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):

    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Role_emb_tensor:
    def __init__(self):
        self.role_emb_tensor = None
        self.all_role_embs = list()
        self.all_lora_tensor = dict()

    def set_role_emb_tensor(self, tensor):
        self.role_emb_tensor = tensor
    
    def __call__(self):
        return self.role_emb_tensor
        

class HyperLora(nn.Module):
    def __init__(self,
                 linear: nn.Linear,
                 roles_emb: nn.Embedding,
                 layers_emb: nn.Embedding,
                 hypernet_encoder: nn.Module,
                 down_linear: nn.Linear,
                 input_dim: int,
                 output_dim: int,
                 rank: int = 8,
                 alpha: int = 32,
                 idx: int = 0,
                 role_ids: Role_ids = None,
                 role_emb_tensor: Role_emb_tensor = None
        ):
        super().__init__()

        self.linear = linear
        self.roles_emb = roles_emb
        self.layers_emb = layers_emb
        self.hypernet_encoder = hypernet_encoder
        self.down_linear = down_linear
        self.rank = rank
        self.alpha = alpha
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=0.1)
        self.role_emb_tensor = role_emb_tensor
        # Layer idx and role id
        self.idx = idx
        self.role_ids = role_ids
        # LoRA
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.output_dim))
        nn.init.zeros_(self.lora_B)


    def embed(self, role: torch.Tensor, layer: torch.Tensor, input_mask: list):
        r = self.roles_emb(role)
        self.role_emb_tensor.set_role_emb_tensor(r)
        l = self.layers_emb(layer)
        # (zero)masking of input embeddings
        x = torch.cat([r * input_mask[0], l * input_mask[1]], dim=-1)

        return x

    def forward(self, x):

        role_ids = self.role_ids.value.to(x.device)
        layer_ids = torch.full(role_ids.size(), self.idx).to(x.device)
        
        vr = self.embed(role_ids, layer_ids, [1, 1])
        hr = self.hypernet_encoder(vr)
        dh =  torch.sqrt(hr.new([hr.shape[1]])).squeeze()

        hyper_A = self.down_linear(hr).reshape(-1, self.input_dim, self.rank) / dh
        scale = self.alpha / self.rank

        out = self.linear(x)
        result_dtype = out.dtype
        x = x.to(self.lora_B.dtype)
        delta = self.dropout(x).bmm(hyper_A)
        delta = delta @ self.lora_B * scale
        out.add_(delta)
        out = out.to(result_dtype)
        
        return out


    

class ResidualBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            dropout: float,
    ):
        super().__init__()

        self.activation_fn = nn.ReLU()
        self.dropout_module = Dropout(dropout)
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout_module(x)
        x = self.fc2(x)
        return x + shortcut

class HyperNetworkEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            layernorm_input: bool,
            layernorm_output: bool,
            dropout: float,
    ):
        super().__init__()

        self.activation_fn = nn.ReLU()
        self.num_layers = num_layers
        self.dropout_module = Dropout(dropout)
        self.input_dim = input_dim
        self.fc_input = Linear(input_dim, hidden_dim)
        layers = []
        for _ in range(self.num_layers):
            layers.append(ResidualBlock(hidden_dim, dropout))
        self.layers = nn.ModuleList(layers)

        if layernorm_input:
            self.norm_input = LayerNorm(input_dim)
            print(self.norm_input.weight)
        else:
            self.norm_input = None

        if layernorm_output:
            self.norm_output = LayerNorm(hidden_dim)
            print(self.norm_output.weight)
        else:
            self.norm_output = None

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor):

        if self.norm_input is not None:
            x = self.norm_input(x)

        x = self.fc_input(x)
        x = self.activation_fn(x)
        x = self.dropout_module(x)

        for layers in self.layers:
            x = layers(x)

        if self.norm_output is not None:
            x = self.norm_output(x)

        return x

class LoRAHead(nn.Module):
    def __init__(self, 
                in_features: int,
                out_feaures: int,
                head_rank: int = 32
        ):
        super().__init__()
        print("head_rank: ", head_rank)
        self.down = Linear(in_features=in_features, out_features=head_rank)
        self.up = Linear(in_features=head_rank, out_features=out_feaures)
    
    def forward(self, x: torch.Tensor):
        return self.up(self.down(x))

class Llama2HyperNetwork(nn.Module):
    def __init__(self,config):
        super(Llama2HyperNetwork, self).__init__()
        self.config = config
        self.roles_num = config.roles_num
        self.roles_emb_dim = config.roles_emb_dim
        self.layers_num = config.layers_num
        self.layers_emb_dim = config.layers_emb_dim
        
        self.residual_blocks_num = config.residual_blocks_num
        self.hyper_hidden_dim = config.hyper_hidden_dim
        
        self.rank_dim = config.rank_dim
        self.alpha = config.alpha
        
        self.layernorm_input = config.layernorm_input
        self.layernorm_output = config.layernorm_output
        self.dropout = config.dropout
        
        self.roles_emb = nn.Embedding(config.roles_num, config.roles_emb_dim)
        self.layers_emb = nn.Embedding(config.layers_num, config.layers_emb_dim)
        
        self.encoding_dim = config.roles_emb_dim + config.layers_emb_dim

        logger.info('Hyper Network Args: {}'.format(dict(
            residual_blocks_num=self.residual_blocks_num,
            roles_num=self.roles_num,
            roles_emb_dim=self.roles_emb_dim,
            layers_num=self.layers_num,
            layers_emb_dim=self.layers_emb_dim,
            hyper_hidden_dim=self.hyper_hidden_dim
        )))

        self.hypernet_encoder = HyperNetworkEncoder(
            input_dim = self.encoding_dim,
            hidden_dim = self.hyper_hidden_dim,
            num_layers = self.residual_blocks_num,
            layernorm_input = self.layernorm_input,
            layernorm_output = self.layernorm_output,
            dropout = self.dropout,
        )
        # q_proj
        self.q_down_linear = LoRAHead(self.hyper_hidden_dim, config.hidden_size * config.rank_dim)
        # k_proj
        self.k_down_linear = LoRAHead(self.hyper_hidden_dim, config.hidden_size * config.rank_dim)
        # v_proj
        self.v_down_linear = LoRAHead(self.hyper_hidden_dim, config.hidden_size * config.rank_dim)
        # o_proj
        self.o_down_linear = LoRAHead(self.hyper_hidden_dim, config.hidden_size * config.rank_dim)
        # up_proj
        self.up_down_linear = LoRAHead(self.hyper_hidden_dim, config.hidden_size * config.rank_dim)
        # down_proj
        self.down_down_linear = LoRAHead(self.hyper_hidden_dim, config.intermediate_size * config.rank_dim)
        # gate_proj
        self.gate_down_linear = LoRAHead(self.hyper_hidden_dim, config.hidden_size * config.rank_dim)

        self.role_ids = Role_ids()
        self.role_emb_tensor = Role_emb_tensor()
        self.init_weights()
        
    def set_role_ids(self, role_ids):
        self.role_ids.set_role_ids(role_ids)
        
    def embed(self, role: torch.Tensor, layer: torch.Tensor, input_mask: list):

        r = self.roles_emb(role)
        l = self.layers_emb(layer)

        # (zero)masking of input embeddings
        x = torch.cat([r * input_mask[0], l * input_mask[1]], dim=-1)

        return x

    def hyper_init(self, layer, target_in, target_out):
        with torch.no_grad():
            # feed random embedding into the hyper-network
            # and generate the weights for the given layer
            input_i = torch.randint(1, (1,)).squeeze()
            x = self.embed(input_i, input_i, [1, 1])

            # feed x to the hyper-network's layer and obtain the hyper-weight
            h = self.hypernet_encoder(x)
            dh_sqrt = torch.sqrt(h.new([h.shape[0]])).squeeze()
            hyper_weights = layer(h)

            hyper_weights = hyper_weights / dh_sqrt
            hyper_std = hyper_weights.std()

            # create regular (adapter) Linear layer(s)
            # to estimate the target STD for the hyper-layers
            target_std = Linear(target_in, target_out).weight.std()

            # scale down the weights of the layer,
            # to produce hyper-layer with same std as the regular layer
            factor = target_std / hyper_std
            # layer.weight.data *= factor
            layer.down.weight.data *= torch.sqrt(factor)
            layer.up.weight.data *= torch.sqrt(factor)
            return target_std

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                if m.weight.requires_grad:
                    nn.init.trunc_normal_(m.weight)
                    
        self.hyper_init(self.q_down_linear, self.config.hidden_size, self.rank_dim)
        self.hyper_init(self.k_down_linear, self.config.hidden_size, self.rank_dim)
        self.hyper_init(self.v_down_linear, self.config.hidden_size, self.rank_dim)
        self.hyper_init(self.o_down_linear, self.config.hidden_size, self.rank_dim)
        self.hyper_init(self.up_down_linear, self.config.hidden_size, self.rank_dim)
        self.hyper_init(self.down_down_linear, self.config.intermediate_size, self.rank_dim)
        self.hyper_init(self.gate_down_linear, self.config.hidden_size, self.rank_dim)









