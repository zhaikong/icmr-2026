import torch
import torch.nn as nn
import torch.nn.functional as F

class BiShareAdapter(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(BiShareAdapter, self).__init__()
        self.hidden_dim = hidden_dim
        self.l1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.l2 = nn.Linear(hidden_dim//2, hidden_dim)
        self.multihead_attention1 = nn.MultiheadAttention(hidden_dim//2, num_heads)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.init_weights()
        
    def init_weights(self):
        self.l2.weight.data.zero_()
        self.l2.bias.data.zero_()

    def forward(self, x):
        xinit = x
        x = self.l1(x)
        x2 = x
        attn_output, _ = self.multihead_attention1(x, x, x)
        x = F.gelu(x)
        alpha = torch.sigmoid(self.gate1)
        attn = alpha * attn_output + (1 - alpha) * x2
        x = self.l2(attn)
        return x + xinit

class MMadapter(nn.Module):
    def __init__(self, share_adapter, hidden_size, layer_id=0):
        super(MMadapter, self).__init__()
        # 默认中间层维度为 128
        self.img_proj_down = nn.Linear(hidden_size, 128)
        self.img_proj_up = nn.Linear(128, hidden_size)
        self.BiShareAdapterxx = share_adapter
        self.multihead_attention = nn.MultiheadAttention(128, 8)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        self.img_proj_up.weight.data.zero_()
        self.img_proj_up.bias.data.zero_()

    def forward(self, x):
        x_init = x
        x = self.img_proj_down(x)
        x = F.gelu(x)
        xmid = x
        x, _ = self.multihead_attention(x, x, x)
        if self.BiShareAdapterxx is not None:
            x = self.BiShareAdapterxx(x)
        x, _ = self.multihead_attention(x, x, x)
        alpha = torch.sigmoid(self.gate1)
        x = alpha * xmid + (1 - alpha) * x
        x = self.img_proj_up(x)
        return x_init + x


def create_adapters(num_layers, hidden_dim, adapter_type='bishare', num_heads=8, 
                    share_across_layers=False, bottleneck_dim=128):
    """
    创建 Adapter 模块列表的工厂函数
    
    Args:
        num_layers (int): Transformer 层数
        hidden_dim (int): 隐藏层维度
        adapter_type (str): Adapter 类型，'bishare' 或 'mm'
        num_heads (int): 注意力头数
        share_across_layers (bool): 是否在层之间共享 Adapter 参数
        bottleneck_dim (int): MMAdapter 的瓶颈维度
        
    Returns:
        nn.ModuleList: Adapter 模块列表
    """
    if adapter_type == 'bishare':
        if share_across_layers:
            # 所有层共享同一个 Adapter
            shared_adapter = BiShareAdapter(hidden_dim, num_heads)
            adapters = nn.ModuleList([shared_adapter for _ in range(num_layers)])
        else:
            # 每层独立的 Adapter
            adapters = nn.ModuleList([
                BiShareAdapter(hidden_dim, num_heads) for _ in range(num_layers)
            ])
    elif adapter_type == 'mm':
        if share_across_layers:
            # 创建一个共享的 BiShareAdapter
            shared_bishare = BiShareAdapter(bottleneck_dim, num_heads)
            adapters = nn.ModuleList([
                MMAdapter(shared_bishare, hidden_dim, i, bottleneck_dim) 
                for i in range(num_layers)
            ])
        else:
            # 每层独立的 MMAdapter
            adapters = nn.ModuleList([
                MMAdapter(None, hidden_dim, i, bottleneck_dim) 
                for i in range(num_layers)
            ])
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")
    
    return adapters


def freeze_model_except_adapters(model):
    """
    冻结模型的所有参数，除了 Adapter 相关的参数
    这是参数高效微调的关键步骤
    
    Args:
        model (nn.Module): 要冻结的模型
        
    Returns:
        tuple: (总参数量, 可训练参数量, 冻结参数量)
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # 只训练包含 'adapter' 或 'gate' 的参数
        if 'adapter' in name.lower() or 'gate' in name.lower():
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"\n{'='*60}")
    print(f"参数高效微调统计:")
    print(f"{'='*60}")
    print(f"总参数量:        {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量:    {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"冻结参数量:      {frozen_params:,} ({frozen_params/1e6:.2f}M)")
    print(f"可训练比例:      {100*trainable_params/total_params:.2f}%")
    print(f"{'='*60}\n")
    
    return total_params, trainable_params, frozen_params


def get_adapter_state_dict(model):
    """
    只获取 Adapter 相关的参数字典
    用于保存轻量级的 checkpoint
    
    Args:
        model (nn.Module): 模型
        
    Returns:
        dict: 只包含 Adapter 参数的状态字典
    """
    adapter_state_dict = {}
    for name, param in model.named_parameters():
        if 'adapter' in name.lower() or 'gate' in name.lower():
            adapter_state_dict[name] = param.data
    
    return adapter_state_dict


def load_adapter_state_dict(model, adapter_state_dict, strict=False):
    """
    只加载 Adapter 相关的参数
    
    Args:
        model (nn.Module): 模型
        adapter_state_dict (dict): Adapter 参数字典
        strict (bool): 是否严格匹配
        
    Returns:
        tuple: (missing_keys, unexpected_keys)
    """
    model_state_dict = model.state_dict()
    
    missing_keys = []
    unexpected_keys = []
    
    # 更新 Adapter 参数
    for name, param in adapter_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param
        else:
            unexpected_keys.append(name)
    
    # 检查缺失的 Adapter 参数
    for name in model_state_dict.keys():
        if ('adapter' in name.lower() or 'gate' in name.lower()) and name not in adapter_state_dict:
            missing_keys.append(name)
    
    model.load_state_dict(model_state_dict, strict=strict)
    
    return missing_keys, unexpected_keys
