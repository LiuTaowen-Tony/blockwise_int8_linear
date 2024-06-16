from functools import partial

import torch
import torch.nn as nn

import torch.optim as optim

from triton_functions import (
    int8_matmul_block64_rowwise_dequantize,
    quantize_block_rowwise,
    dequantize_block_rowwise
)

QUANTISE_WEIGHT=True

def replace_linear_with_blockwise_int8(model):
    to_be_replaced = []
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            to_be_replaced.append(name)
    
    for name in to_be_replaced:
        module = model._modules[name]
        model._modules[name] = BlockInt8Linear(module.in_features, 
                                               module.out_features, 
                                               module.bias is not None, 
                                               module.weight.device, 
                                               module.weight.dtype)
        model._modules[name].weight.data = module.weight.data
        if module.bias is not None:
            model._modules[name].bias.data = module.bias.data
    return model

def fast_matmul(a, b):
    b = b.t().contiguous()
    a_int8, a_state = quantize_block_rowwise(a)
    print((a_int8.abs()<3).sum()/ a_int8.numel())
    b_int8, b_state = quantize_block_rowwise(b)
    print((b_int8.abs()<3).sum()/ b_int8.numel())
    return int8_matmul_block64_rowwise_dequantize(a_int8, b_int8.t(), a_state, b_state)

class blockwise_int8_linear_Wint8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_3D, W_int8, state_W, bias):
        X = X_3D.view(-1, X_3D.size(-1))
        ctx.save_for_backward = X, W_int8, state_W
        return int8_matmul_block64_rowwise_dequantize(X, W_int8.t(), state_W, bias).view(*X_3D.size()[:-1], -1)

    @staticmethod
    def backward(ctx: torch.Any, G_3D) -> torch.Any:
        G = G_3D.reshape(-1, G_3D.size(-1))
        grad_X = grad_W = grad_bias = None

        X, W_int8, state_W = ctx.save_for_backward
        if ctx.needs_input_grad[0]:
            g_int8, g_state = quantize_block_rowwise(G)
            grad_X = int8_matmul_block64_rowwise_dequantize(g_int8, W_int8, g_state, state_W).view(*G_3D.size()[:-1], -1)
        if ctx.needs_input_grad[1]:
            Xt = X.t().contiguous()
            Gt = G.t().contiguous()
            x_int8, x_state = quantize_block_rowwise(Xt)
            g_int8, g_state = quantize_block_rowwise(Gt)
            grad_W = int8_matmul_block64_rowwise_dequantize(g_int8, x_int8.t(), g_state, x_state)
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias

class blockwise_int8_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_3D, W, bias):
        X = X_3D.view(-1, X_3D.size(-1))
        ctx.save_for_backward = X, W
        # return (fast_matmul(X, W.t()) + bias).view(*X_3D.size()[:-1], -1)
        # if bias is None:
        #     return X.mm(W.t()).view(*X_3D.size()[:-1], -1)
        # return X.mm(W.t()).add(bias).view(*X_3D.size()[:-1], -1)
        X_int8, state_X = quantize_block_rowwise(X)
        W_int8, state_W = quantize_block_rowwise(W)
        return int8_matmul_block64_rowwise_dequantize(X_int8, W_int8.t(), state_X, state_W, bias).view(*X_3D.size()[:-1], -1)

    @staticmethod
    def backward(ctx: torch.Any, G_3D) -> torch.Any:
        G = G_3D.reshape(-1, G_3D.size(-1))
        grad_X = grad_W = grad_bias = None

        X, W = ctx.save_for_backward
        if ctx.needs_input_grad[0]:
            # grad_X = fast_matmulT(G, W.t()).view(*G_3D.size()[:-1], -1)
            # grad_X = G.mm(W.to(G.dtype)).view(*G_3D.size()[:-1], -1)
            # grad_X = fast_matmul(G, W).view(*G_3D.size()[:-1], -1)
            g_int8, g_state = quantize_block_rowwise(G)
            W_int8, w_state = quantize_block_rowwise(W)
            grad_X = int8_matmul_block64_rowwise_dequantize(g_int8, W_int8, g_state, w_state).view(*G_3D.size()[:-1], -1)
        if ctx.needs_input_grad[1]:
            # grad_W = fast_matmulT(G.t(), X.t())
            # grad_W = G.t().mm(X.to(G.dtype))
            # ((Gt X)t)t = (Xt G)t
            Xt = X.t().contiguous()
            Gt = G.t().contiguous()
            x_int8, x_state = quantize_block_rowwise(Xt)
            g_int8, g_state = quantize_block_rowwise(Gt)
            grad_W = int8_matmul_block64_rowwise_dequantize(g_int8, x_int8.t(), g_state, x_state)
            # X_int8, x_state = quantize_block_rowwise(X)
            # g_int8, g_state = quantize_block_rowwise(G)
            # grad_W = int8_matmul_block64_rowwise_dequantize(X_int8, g_int8.t(), g_state, x_state)
            # grad_W = fast_matmul(G.t(), X)
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias

class BlockInt8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

    def prepare_for_eval(self):
        W_int8, state_W = quantize_block_rowwise(self.weight)
        self.register_buffer("W_int8", W_int8)
        self.register_buffer("state_W", state_W)

    def weight_quant_training(self, x):
        W_int8, state_W = quantize_block_rowwise(self.weight)
        del self.weight


        self.weight = dequantize_block_rowwise(W_int8, state_W)

    def infer(self, x):
        if not hasattr(self, "W_int8"):
            self.prepare_for_eval()
        X = x.view(-1, x.size(-1))
        X_int8, state_X = quantize_block_rowwise(X)
        return int8_matmul_block64_rowwise_dequantize(
            X_int8, self.W_int8.t(), state_X, self.state_W, self.bias
        ).view(*x.size()[:-1], -1)

    def forward(self, x):
        if not self.training:
            return self.infer(x)
        # if QUANTISE_WEIGHT:
        #     return self.weight_quant_training(x)

        return blockwise_int8_linear.apply(x, self.weight, self.bias)


#class BlockInt8Linear()

if __name__ == "__main__":
    # x = torch.randn(128, 128, device="cuda")
    # y = x.t()
    # y_int8, state_y = quantize_block_rowwise(y)

    # z = x.t().contiguous()
    # z_int8, state_z = quantize_block_rowwise(z)

    # print(y_int8)
    # print(z_int8)
    # print(state_y)
    # print(state_z)


    from microxcaling.mx import finalize_mx_specs
    from microxcaling import mx
    import block_linear_config
    block_linear_config.INPUT_OUTPUT_TORCH_TYPE = torch.float32
    block_linear_config.GROUP_SIZE = 128
    

    mx_specs = {
        'w_elem_format': 'int8',
        'a_elem_format': 'int8',
        'block_size': 128,
        'bfloat': 16,
        'custom_cuda': True,
        'quantize_backprop': True,
    }
    mx_specs = finalize_mx_specs(mx_specs)

    dtype = torch.float32
    mxlinear = mx.Linear(128, 128, mx_specs=mx_specs, bias=False).cuda()
    blinear = BlockInt8Linear(128, 128, dtype=dtype, bias=False).cuda()
    linear = nn.Linear(128, 128, dtype=dtype, bias=False).cuda()
    linear.weight.data.copy_(blinear.weight.data)
    mxlinear.weight.data.copy_(blinear.weight.data)

    x = torch.randn(128, 128, device="cuda", requires_grad=True, dtype=dtype)
    y = torch.randn(128, 128, device="cuda", dtype=dtype)

    by_hat = blinear(x)
    by_hat.retain_grad()
    y_hat = linear(x)
    mx_y_hat = mxlinear(x)
    # print("-------")
    # print(y_hat)
    # print(by_hat)
    # print(mx_y_hat)

    bloss = (by_hat - y).pow(2).sum()
    bloss.backward()
    b_grad = blinear.weight.grad
    b_x_grad = x.grad.clone()

    x.grad.zero_()

    loss = (y_hat - y).pow(2).sum()
    loss.backward()
    grad = linear.weight.grad
    x_grad = x.grad.clone()

    x.grad.zero_()

    mx_loss = (mx_y_hat - y).pow(2).sum()
    mx_loss.backward()
    mx_grad = mxlinear.weight.grad
    mx_x_grad = x.grad.clone()


    # print("-----------")
    # print((b_grad - grad).abs().max())
    # print((mx_grad - grad).abs().max())
    # print(b_grad)
    # print(grad)
    # print(mx_grad)

    
    # print("-----------")
    # print((b_x_grad - x_grad).abs().max())
    # print((mx_x_grad - x_grad).abs().max())
    # print(b_x_grad)
    # print(x_grad)
    # print(mx_x_grad)

    print("-----------")
    print(by_hat.grad)
    W = blinear.weight
    G = by_hat.grad

    g_int8, g_state = quantize_block_rowwise(G)
    W_int8, w_state = quantize_block_rowwise(W)
    grad_X_a = int8_matmul_block64_rowwise_dequantize(g_int8, W_int8, g_state, w_state)
    true_x_grad = by_hat.grad.mm(blinear.weight)

    Wt = W.t().contiguous()
    Wt_int8, wt_state = quantize_block_rowwise(Wt)
    grad_X_b = int8_matmul_block64_rowwise_dequantize(g_int8, Wt_int8.t(), g_state, wt_state)

    

    print((grad_X_a - grad_X_b).abs().max())
    print(grad_X_a)
    print(grad_X_b)
    print(true_x_grad)


    print(g_state)
    # print(w_state)
    # print(wt_state)
    # direct_grad2 = by_hat.grad.t().mm(blinear.weight)
    # print((direct_grad - direct_grad2).abs().max())
    # print(direct_grad)
    # print(direct_grad2)






