import torchattacks
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
import numpy as np

from scipy.optimize import linear_sum_assignment
import math

import heapq

# def set_PDSG(model: nn.Module):
#     for name, m in model.named_children():
#         if isinstance(m, neuron.LIFNode):
#             setattr(model, name, PDSG_LIFNode(m.tau, m.decay_input, m.v_threshold, m.v_reset, m.detach_reset))
#         else:
#             set_PDSG(m)
#     return model

def set_PDSG(model: nn.Module):
    for name, m in model.named_children():
        if isinstance(m, neuron.LIFNode):
            setattr(model, name, PDSG_LIFNode(m.tau, m.decay_input, m.v_threshold, m.v_reset, m.detach_reset))
        elif isinstance(m, neuron.ParametricLIFNode):  # ← 新增分支
            # 反算 tau：优先 w（tau = 1 + exp(-w)），备选 log_tau（tau = exp(log_tau)）
            with torch.no_grad():
                if hasattr(m, 'w'):
                    tau = 1.0 + torch.exp(-m.w)
                elif hasattr(m, 'log_tau'):
                    tau = torch.exp(m.log_tau)
                else:
                    tau = torch.as_tensor(getattr(m, 'tau', 2.0))
                if tau.numel() > 1:
                    tau = tau.mean()
                tau = float(tau.item())
            setattr(model, name, PDSG_LIFNode(tau, m.decay_input, m.v_threshold, m.v_reset, m.detach_reset))
        else:
            set_PDSG(m)
    return model


class FGSM(torchattacks.FGSM):
    def __init__(self, model, encoder, eps=8/255, min_val=0, max_val=1):
        super().__init__(model, eps)
        self.model = SNNContainer(model, encoder)
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, inputs, labels=None, *args, **kwargs):
        self.model.training = self.model.model.training
        return super().__call__(inputs, labels, *args, **kwargs)
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=self.min_val, max=self.max_val).detach()

        return adv_images
        
class PGD(torchattacks.PGD):
    def __init__(self, model, encoder, eps=8/255, steps=10, random_start=True, min_val=0, max_val=1):
        super().__init__(model, eps, eps/4, steps, random_start)
                
        self.model = SNNContainer(model, encoder)
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, inputs, labels=None, *args, **kwargs):
        self.model.training = self.model.model.training
        return super().__call__(inputs, labels, *args, **kwargs)
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.min_val, max=self.max_val).detach()

        return adv_images

class CWLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, labels):
        num_classes = inputs.shape[1]
        loss = 0
        labels = nn.functional.one_hot(labels, num_classes=num_classes).float()
        for b in range(inputs.shape[0]):
            t = inputs[b][labels[b] == 1] - torch.max(inputs[b][labels[b] != 1])
            loss += torch.max(t, torch.zeros_like(t))
        return loss / inputs.shape[0]

@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

# PDSG backward function
def piecewise_quadratic_backward(grad_output: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor):
    sigma = sigma.expand_as(x)
    grad = torch.exp(- ((x - 0.5 * sigma)/(np.sqrt(2)*sigma))**2) / (np.sqrt(2*np.pi)*sigma)
    grad_input = grad_output * grad
    return grad_input, None, None, None


class surrogate_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, sigma: torch.Tensor):
        if x.requires_grad:
            ctx.save_for_backward(x, sigma)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.saved_tensors[1])

class PDSG(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, v: torch.Tensor):
        with torch.no_grad():
            if len(x.shape) == 4: # for conv layer
                if x.shape[2] == x.shape[3]: # conventional conv layer
                    dim = (0,1,3,4)
                else: # transformer layer
                    dim = (0,1,2,4)
            else: # for linear layer
                dim = tuple(range(len(x.shape)))
            sigma = v.std(dim=dim, keepdim=True).squeeze(0)
            sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
        return surrogate_func.apply(x, sigma)

class SDA(nn.Module):
    def __init__(self, model, k_init=10, N=500, batch_limit=64, scale = False):
        super().__init__()
        self.k_init = k_init
        self.N = N
        self.model = model
        self.batch_limit = batch_limit # prevent GPU out of memory when k is large
        self.scale = scale

    # def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
    #     for l in self.model.modules():
    #         if isinstance(l, neuron.BaseNode):
    #             l.train() # enable back-propagation
    #         else:
    #             l.eval()   
    #     assert inputs.shape[1] == 1
    #     criterion = CWLoss()

    #     selected_mask = torch.zeros_like(inputs)
    #     FDs = torch.zeros_like(inputs).fill_(torch.inf)
    #     adv_inputs = inputs.clone().detach().to(inputs)
    #     success_flag = False
        
    #     # Generation process
    #     for n in range(self.N):
    #         # calculate gradients
    #         adv_inputs = adv_inputs.detach()
    #         adv_inputs.requires_grad_(True)
    #         outputs = self.model(adv_inputs).mean(0)
    #         loss = criterion(outputs, labels)
            
    #         grad = torch.autograd.grad(loss, adv_inputs,
    #                             retain_graph=False, create_graph=False)[0]
            
    #         # ---------------------------------------------------------------------------
    #         # step 1: select contributing gradients. (1-2x)*g<=0 equals to 0<=x-sgn(g)<=1
    #         grad_mask = adv_inputs - grad.sign()
    #         grad_mask = torch.bitwise_and(grad_mask >= 0.0, grad_mask <= 1.0)
    #         grad = grad * grad_mask
            
    #         # prevent gradient vanishing (seldom occurs)
    #         if (grad != 0).any():
    #             random_value = torch.rand_like(grad) * grad[grad != 0].abs().min()
    #         else:
    #             random_value = torch.rand_like(grad)
    #         grad[grad == 0] = random_value[grad == 0]
            
    #         # exclude selected pixels
    #         grad = grad * (1 - selected_mask)
            
    #         # ---------------------------------------------------------------------------
    #         # step 2: select topk gradients
            
    #         # update k
    #         k = (n+1) * self.k_init
            
    #         # indices of topk gradients
    #         indices = list(np.unravel_index(torch.topk(grad.abs().flatten(), k=k)[1].cpu(), shape=inputs.shape))
            
    #         # parallel perturb and forward
    #         indices = [list(i) for i in indices]
    #         indices[1] = list(np.linspace(0, k-1, k, dtype=int) % self.batch_limit)
    #         with torch.no_grad():
    #             if k <= self.batch_limit:
    #                 parallel_adv_inputs = adv_inputs.repeat(1,k,1,1,1)
    #                 parallel_adv_inputs[indices] = 1 - parallel_adv_inputs[indices]
    #                 outputs = self.model(parallel_adv_inputs).mean(0)
    #                 loss_each_batch = []
    #                 for b in range(k):
    #                     loss_each_batch.append(criterion(outputs[b].unsqueeze(0), labels))
    #                 loss_each_batch = torch.stack(loss_each_batch).squeeze(1)
    #             else: # When k is large, GPU out of memory may occur. Hence divide k by groups.
    #                 loss_each_batch = []
    #                 for kb in range(int(np.ceil(k / self.batch_limit))):
    #                     start = kb * self.batch_limit
    #                     end = np.min([(kb+1)*self.batch_limit, k])
    #                     parallel_adv_inputs = adv_inputs.repeat(1,end-start,1,1,1)
    #                     index = list(np.array(indices)[:,start:end])
    #                     parallel_adv_inputs[index] = 1 - parallel_adv_inputs[index]
    #                     outputs = self.model(parallel_adv_inputs).mean(0)
    #                     for b in range(outputs.shape[0]):
    #                         loss_each_batch.append(criterion(outputs[b].unsqueeze(0), labels))
    #                 loss_each_batch = torch.stack(loss_each_batch).squeeze(1)           
    #         indices[1] = np.zeros(k, dtype=int)
            
    #         # ---------------------------------------------------------------------------
    #         # step 3: calculate FDs, FDs=[loss(adv)-loss(ori)]/delta. We expect loss(adv) < loss(ori)
    #         FDs[indices] = loss_each_batch - loss.repeat(k)
    #         selected_mask[indices] = 1
    #         adv_inputs.requires_grad_(False)
    #         selected_mask[FDs > 0] = 0
    #         FDs[FDs > 0] = torch.inf
            
    #         # update adversarial examples
    #         adv_inputs = inputs * (1 - selected_mask) + (1 - inputs) * selected_mask
            
    #         # test if attack is successful
    #         with torch.no_grad():
    #             outputs = self.model(adv_inputs).mean(0)
    #             _, predicted = outputs.max(1)
    #             if predicted.eq(labels).cpu().sum().item() == 0:
    #                 success_flag = True
    #                 break
        
    #     if not success_flag:
    #         return inputs
        
    #     # Reduction process
    #     final_adv_inputs = adv_inputs
    #     n_count = (selected_mask == 1).sum().item()
    #     FDs_sort = torch.topk(FDs.abs().flatten(), k=n_count, largest=False)[1].cpu()
    #     left_ptr = 0
    #     right_ptr = n_count - 1
    #     while left_ptr <= right_ptr: # binary search
    #         ptr = (left_ptr + right_ptr) // 2
    #         FDs_indices = np.unravel_index(FDs_sort[0:ptr+1], shape=inputs.shape)
    #         temp_adv_inputs = adv_inputs.clone().detach()
    #         temp_adv_inputs[FDs_indices] = 1 - adv_inputs[FDs_indices]
    #         with torch.no_grad():
    #             outputs = self.model(temp_adv_inputs).mean(0)
    #             _, predicted = outputs.max(1)
    #             if predicted.eq(labels).cpu().sum().item() == 0: # still adversarial
    #                 left_ptr = ptr + 1
    #                 final_adv_inputs = temp_adv_inputs
    #             else: # not adversarial
    #                 right_ptr = ptr - 1
 
    #     return final_adv_inputs 

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, target_label: int = -1):
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # 让 SNN BaseNode 参与反传，其余模块 eval（沿用原逻辑）
        for l in self.model.modules():
            if isinstance(l, neuron.BaseNode):
                l.train()
            else:
                l.eval()

        assert inputs.shape[1] == 1

        # -------- targeted / untargeted 设置 --------
        is_targeted = (target_label is not None) and (target_label >= 0)
        if is_targeted:
            ce_criterion = nn.CrossEntropyLoss()
            tgt_tensor = torch.full_like(labels, target_label, dtype=labels.dtype, device=labels.device)
            def compute_loss(logits):
                return ce_criterion(logits, tgt_tensor)
            def success_cond(pred):
                return (pred.eq(tgt_tensor).cpu().sum().item() == pred.numel())
        else:
            criterion = CWLoss()
            def compute_loss(logits):
                return criterion(logits, labels)
            def success_cond(pred):
                return (pred.eq(labels).cpu().sum().item() == 0)

        selected_mask = torch.zeros_like(inputs)
        FDs = torch.zeros_like(inputs).fill_(torch.inf)
        adv_inputs = inputs.clone().detach().to(inputs)
        success_flag = False

        # ===================== Generation process =====================
        for n in range(self.N):
            # print (n)
            if is_targeted and n >= 50:
                break
            # 计算梯度
            adv_inputs = adv_inputs.detach()
            adv_inputs.requires_grad_(True)
            outputs = self.model(adv_inputs).mean(0)  # [B, C]
            loss = compute_loss(outputs)              # 标量
            if loss.dim() > 0:
                loss = loss.mean()                                # -> 0-d 标量

            grad = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)[0]

            # step 1: 选择贡献梯度 (1-2x)*g<=0 等价于 0<=x-sgn(g)<=1
            grad_mask = adv_inputs - grad.sign()
            grad_mask = torch.bitwise_and(grad_mask >= 0.0, grad_mask <= 1.0)
            grad = grad * grad_mask

            # 防止梯度完全为 0
            if (grad != 0).any():
                random_value = torch.rand_like(grad) * grad[grad != 0].abs().min()
            else:
                random_value = torch.rand_like(grad)
            grad[grad == 0] = random_value[grad == 0]

            # 排除已选像素
            grad = grad * (1 - selected_mask)

            # step 2: 选 top-k 梯度
            k = (n + 1) * self.k_init
            flat_idx = torch.topk(grad.abs().flatten(), k=k)[1].cpu()
            indices = list(np.unravel_index(flat_idx, shape=inputs.shape))  # [dim lists]

            # 并行扰动 + 前向
            indices = [list(i) for i in indices]
            indices[1] = list(np.linspace(0, k - 1, k, dtype=int) % self.batch_limit)

            with torch.no_grad():
                if k <= self.batch_limit:
                    parallel_adv_inputs = adv_inputs.repeat(1, k, 1, 1, 1)
                    parallel_adv_inputs[indices] = 1 - parallel_adv_inputs[indices]
                    outputs = self.model(parallel_adv_inputs).mean(0)  # [k, B, C]→mean(0)→[B,C]? 原代码为 mean(0)
                    # 按 batch 逐一取 loss（保持原有语义）
                    loss_each_batch = []
                    for b in range(k):
                        loss_each_batch.append(compute_loss(outputs[b].unsqueeze(0)))
                    loss_each_batch = torch.stack(loss_each_batch)  # [k]
                else:
                    loss_each_batch = []
                    n_groups = int(np.ceil(k / self.batch_limit))
                    for kb in range(n_groups):
                        start = kb * self.batch_limit
                        end = np.minimum((kb + 1) * self.batch_limit, k)
                        parallel_adv_inputs = adv_inputs.repeat(1, end - start, 1, 1, 1)
                        index = list(np.array(indices)[:, start:end])
                        parallel_adv_inputs[index] = 1 - parallel_adv_inputs[index]
                        outputs = self.model(parallel_adv_inputs).mean(0)
                        for b in range(outputs.shape[0]):
                            loss_each_batch.append(compute_loss(outputs[b].unsqueeze(0)))
                    loss_each_batch = torch.stack(loss_each_batch)  # [k]
                loss_each_batch = loss_each_batch.view(-1)            # 强制变成 [k]

            indices[1] = np.zeros(k, dtype=int)

            # print (FDs[indices].shape)
            # print (loss_each_batch.shape)
            # print (loss.shape)
            # step 3: 计算 FDs（期望 loss(adv) < loss(ori)）
            FDs[indices] = loss_each_batch - loss  # 标量广播
            selected_mask[indices] = 1
            adv_inputs.requires_grad_(False)

            # 过滤无益像素
            bad = (FDs > 0)
            selected_mask[bad] = 0
            FDs[bad] = torch.inf

            # 更新对抗样本（二值翻转）
            adv_inputs = inputs * (1 - selected_mask) + (1 - inputs) * selected_mask

            # 成功判定
            with torch.no_grad():
                outputs = self.model(adv_inputs).mean(0)
                _, predicted = outputs.max(1)
                if success_cond(predicted):
                    success_flag = True
                    break

        if not success_flag:
            return inputs

        # ===================== Reduction process =====================
        final_adv_inputs = adv_inputs
        n_count = (selected_mask == 1).sum().item()
        FDs_sort = torch.topk(FDs.abs().flatten(), k=n_count, largest=False)[1].cpu()
        left_ptr, right_ptr = 0, n_count - 1

        while left_ptr <= right_ptr:  # 二分删点
            ptr = (left_ptr + right_ptr) // 2
            FDs_indices = np.unravel_index(FDs_sort[0:ptr + 1], shape=inputs.shape)
            temp_adv_inputs = adv_inputs.clone().detach()
            temp_adv_inputs[FDs_indices] = 1 - adv_inputs[FDs_indices]
            with torch.no_grad():
                outputs = self.model(temp_adv_inputs).mean(0)
                _, predicted = outputs.max(1)
                if success_cond(predicted):  # 仍对抗
                    left_ptr = ptr + 1
                    final_adv_inputs = temp_adv_inputs
                else:
                    right_ptr = ptr - 1

        if self.scale:
            nonzero_mask = (inputs != 0)
            if nonzero_mask.any():
                nonzero_vals = inputs[nonzero_mask]
                mean_val = nonzero_vals.float().mean()          # 均值
                upper = 2.0 * mean_val - 1                          # 均值的两倍
                # randint 上界是开区间，先转成整数，至少要 > 1
                high_int = int(upper.item())
                if high_int <= 1:
                    high_int = 2

            # 2) 构造 mask：final_adv_inputs != inputs 且 inputs == 0
            mask = (final_adv_inputs != inputs) & (inputs == 0)
            num = mask.sum()

            if num > 0:
                # 3) 在这些位置填充 [1, high_int) 的随机整数
                final_adv_inputs[mask] = torch.randint(
                    low=1,
                    high=high_int,                 # 上界来自非零均值*2
                    size=(num,),
                    device=final_adv_inputs.device,
                    dtype=final_adv_inputs.dtype,  # 如果是 float，这里会自动转成 float
                )

        return final_adv_inputs




import time

class SpikeFool(nn.Module):
    """
    单类版稀疏对抗攻击（SpikeFool），接口与 SDA 一致：
      forward(inputs, labels) -> adversarial_inputs
    约定：
      - inputs: [T, 1, C, H, W]
      - labels: [1]（可无用）
      - 返回: 与 inputs 同形状
    建议：攻击前先启用 surrogate（如 set_PDSG(net)），并确保 BaseNode.train() 其余 eval().
    """
    def __init__(self,
                 model: nn.Module,
                 max_hamming_distance: int = 1000,
                 lambda_: float = 2.0,
                 lb: float = 0.0,
                 ub: float = 1.0,
                 max_outer_iter: int = 4,
                 max_inner_deepfool: int = 10,
                 overshoot: float = 0.02,
                 step_size: float = 0.1,
                 verbose: bool = False):
        super().__init__()
        self.model = model
        self.max_hamming_distance = int(max_hamming_distance)
        self.lambda_ = float(lambda_)
        self.lb = float(lb)
        self.ub = float(ub)
        self.max_outer_iter = int(max_outer_iter)
        self.max_inner_deepfool = int(max_inner_deepfool)
        self.overshoot = float(overshoot)
        self.step_size = float(step_size)
        self.verbose = bool(verbose)

    # -------------------- 公共 API --------------------
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        self._ensure_T1CHW(inputs)
        # SNN 节点 train（启用替代梯度），其他 eval（稳定 BN/Dropout/DropPath）
        for m in self.model.modules():
            if isinstance(m, neuron.BaseNode):
                m.train()
            else:
                m.eval()

        x0 = inputs.detach().clone()
        device = x0.device
        x = x0.clone()

        # 初始预测
        with torch.no_grad():
            logits0 = self._logits_eval_T1C(x)
            pred0 = self._top1(logits0)
        if self.verbose:
            print(f"[SpikeFool] init pred={pred0}, logits_shape={tuple(logits0.shape)}")

        # 累计重要性（|normal| 的累计，用于 L0 投影排序）
        score = torch.zeros_like(x0, dtype=torch.float32, device=device)

        # 外层循环：逐步逼近边界 + 稀疏化
        for it in range(self.max_outer_iter):
            if self.verbose:
                l0_now = int((x != x0).sum().item())
                cur = self._logits_eval_T1C(x)
                pred = self._top1(cur)
                margin = (cur[0, pred] - (cur[0].topk(2).values[0,1] if cur.size(1)>1 else cur[0, pred])).item()
                print(f"[SpikeFool] iter={it} pred={pred} margin={margin:.6f} L0={l0_now}")

            # 1) 在当前点做 1 次 DeepFool 步（返回法向 normal 以及一个边界点近似）
            normal, boundary_pt = self._deepfool_step(x)
            if normal is None:
                if self.verbose:
                    print("[SpikeFool] deepfool step failed: normal=None, stop.")
                break

            # 2) 线性稀疏更新（贪心沿最大坐标方向推进），不在此处二值化
            x_next = self._linear_sparse_update(x, normal, boundary_pt)

            # 3) 只对“新翻转”的位置累加重要性分数
            with torch.no_grad():
                flip_new = (x_next != x)
                score.add_(flip_new.float() * normal.abs())

            x = x_next

            # 4) L0 投影到预算内（并在此处统一二值化）
            with torch.no_grad():
                x, kept_mask = self._project_L0_binary(x0, x, score, self.max_hamming_distance)
                l0_after = int(kept_mask.sum().item())

            # 5) 早停条件：已改变预测且满足 L0 预算
            with torch.no_grad():
                logits = self._logits_eval_T1C(x)
                pred = self._top1(logits)
            if pred != pred0 and l0_after <= self.max_hamming_distance:
                if self.verbose:
                    print(f"[SpikeFool] success at iter={it}, L0={l0_after}, pred: {pred0}->{pred}")
                return x

        # 未成功则返回原样
        if self.verbose:
            print("[SpikeFool] attack failed: return original inputs.")
        return x0

    # -------------------- 形状/工具 --------------------
    @staticmethod
    def _ensure_T1CHW(x: torch.Tensor):
        if x.dim() != 5 or x.size(1) != 1:
            raise RuntimeError(f"Expected [T,1,C,H,W], got {tuple(x.shape)}")

    @torch.no_grad()
    def _logits_eval_T1C(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_T1CHW(x)
        self._reset_states()
        out = self.model(x)           # 典型: [T,1,C]
        if out.dim() != 3 or out.size(1) != 1:
            raise RuntimeError(f"Model output expected [T,1,C], got {tuple(out.shape)}")
        out = out.mean(dim=0)         # [1,C]
        return out

    def _logits_train_T1C(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_T1CHW(x)
        self._reset_states()
        out = self.model(x)           # [T,1,C]
        if out.dim() != 3 or out.size(1) != 1:
            raise RuntimeError(f"Model output expected [T,1,C], got {tuple(out.shape)}")
        out = out.mean(dim=0)         # [1,C]
        return out

    @staticmethod
    def _top1(logits_1C: torch.Tensor) -> int:
        if logits_1C.dim() != 2 or logits_1C.size(0) != 1:
            raise RuntimeError(f"logits must be [1,C], got {tuple(logits_1C.shape)}")
        return int(torch.argmax(logits_1C, dim=1).item())

    def _reset_states(self):
        try:
            self.model.reset_states()
        except AttributeError:
            pass

    # -------------------- DeepFool 单步（在当前 x 上） --------------------
    def _deepfool_step(self, x: torch.Tensor):
        """
        返回:
          normal: 与 x 同形状的单位法向（若失败则 None）
          boundary_pt: DeepFool 累积后的连续点 x0 + r_tot
        """
        x0 = x.detach().clone()
        device = x0.device
        lambda_fac = self.lambda_
        overshoot = self.overshoot
        step_size = self.step_size
        max_iter = self.max_inner_deepfool

        # 初始标签
        with torch.no_grad():
            f0 = self._logits_eval_T1C(x0)     # [1,C]
            label = self._top1(f0)

        X_adv = x0.clone()
        r_tot = torch.zeros_like(x0, dtype=x0.dtype, device=device)

        k_i = label
        loop_i = 0

        while k_i == label and loop_i < max_iter:
            x_cur = X_adv.clone().detach().requires_grad_(True)
            fs = self._logits_train_T1C(x_cur)            # [1,C]
            num_classes = fs.size(1)

            # 每轮动态排序
            with torch.no_grad():
                I = torch.argsort(fs[0], descending=True)

            # 原类梯度
            self.model.zero_grad(set_to_none=True)
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x_cur.grad.detach().clone()

            # 逐类找最小 pert
            pert = torch.tensor(float("inf"), device=device, dtype=fs.dtype)
            w = torch.zeros_like(x_cur)
            for k in range(1, num_classes):
                x_cur.grad = None
                self.model.zero_grad(set_to_none=True)
                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x_cur.grad.detach().clone()

                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).detach()

                w_k_norm = w_k.norm().clamp_min(1e-12)
                pert_k = torch.abs(f_k) / w_k_norm
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # 稳健步长
            w_norm = w.norm().clamp_min(1e-12)
            r_i = torch.clamp(pert, min=step_size) * w / w_norm
            r_tot = r_tot + r_i
            X_adv = (x0 + r_tot).detach()

            # 重新评估标签
            with torch.no_grad():
                probe = x0 + (1 + overshoot) * r_tot
                logits_probe = self._logits_eval_T1C(probe)
                k_i = self._top1(logits_probe)

            # with torch.no_grad():
            #     top_vals, top_idx = torch.topk(fs[0], k=min(2, fs.size(1)), largest=True)
            #     I0 = int(top_idx[0].item())                  # 原类
            #     if top_idx.numel() == 1:
            #         return None, X_adv                       # 只有1类，无法DeepFool
            #     I1 = int(top_idx[1].item())                  # 对手类（第二大）

            # # 原类梯度已在外面算过：grad_orig = ∂fs[:,I0]/∂x
            # x_cur.grad = None
            # self.model.zero_grad(set_to_none=True)
            # fs[0, I1].backward(retain_graph=True)
            # cur_grad = x_cur.grad.detach().clone()

            # w = cur_grad - grad_orig                         # 法向差
            # f_k = (fs[0, I1] - fs[0, I0]).detach()           # logit 差

            # w_norm = w.norm().clamp_min(1e-12)
            # pert  = torch.abs(f_k) / w_norm                  # 只对 I1 计算一次

            # # 稳健步长
            # r_i   = torch.clamp(pert, min=self.step_size) * w / w_norm
            # r_tot = r_tot + r_i
            # X_adv = (x0 + r_tot).detach()

            # # 越界探针
            # with torch.no_grad():
            #     probe = x0 + (1.0 + self.overshoot) * r_tot
            #     logits_probe = self._logits_eval_T1C(probe)
            #     k_i = self._top1(logits_probe)

            loop_i += 1

        # 末尾取法向
        x_final = X_adv.clone().detach().requires_grad_(True)
        fs = self._logits_train_T1C(x_final)              # [1,C]
        self.model.zero_grad(set_to_none=True)
        (fs[0, k_i] - fs[0, label]).backward(retain_graph=False)
        normal = x_final.grad.detach()
        nrm = normal.norm()
        if nrm <= 0:
            return None, X_adv
        normal = normal / nrm.clamp_min(1e-10)

        # 放大一次（DeepFool 原则）
        r_tot = lambda_fac * r_tot
        boundary_pt = (x0 + r_tot).detach()
        return normal, boundary_pt

    # -------------------- 线性稀疏更新（贪心最大坐标） --------------------
    def _linear_sparse_update(self, x_i: torch.Tensor, normal: torch.Tensor, boundary_point: torch.Tensor):
        """
        不在这里二值化；只负责把 x_i 朝边界“按最大坐标”推进，直到越界。
        """
        lb, ub = self.lb, self.ub
        coord_vec = normal.detach().clone()               # 同形状
        plane_normal = coord_vec.reshape(-1)
        plane_point  = boundary_point.detach().reshape(-1)

        x = x_i.detach().clone()

        f_k = torch.dot(plane_normal, x_i.reshape(-1) - plane_point)
        sign_true = float(torch.sign(f_k).item() or 1.0)
        beta = 1e-3 * sign_true
        current_sign = sign_true

        # 贪心地一次只修改“幅值最大的坐标”
        while current_sign == sign_true and coord_vec.nonzero().numel() > 0:
            f_k = torch.dot(plane_normal, x.reshape(-1) - plane_point) + beta
            denom = coord_vec.abs().max().clamp_min(1e-12)
            pert = (f_k.abs() / denom).to(x.dtype)

            # 只在最大坐标上更新
            max_idx = torch.argmax(coord_vec.abs()).item()
            r = torch.zeros_like(coord_vec).reshape(-1)
            r[max_idx] = (pert * coord_vec.reshape(-1)[max_idx].sign()).item()
            r = r.view_as(coord_vec)

            x = (x + r).clamp(lb, ub)

            f_k = torch.dot(plane_normal, x.reshape(-1) - plane_point)
            current_sign = float(torch.sign(f_k).item() or -sign_true)

            # 用过的坐标置 0，避免重复
            coord_vec.reshape(-1)[max_idx] = 0.0

        return x

    # -------------------- L0 投影 + 二值化 --------------------
    @staticmethod
    def _project_L0_binary(x0: torch.Tensor,
                           x: torch.Tensor,
                           score: torch.Tensor,
                           k: int):
        """
        将 x 投影到与 x0 的 Hamming 距离 ≤ k，并在投影后**统一二值化**到 {0,1}.
        score：重要性（|normal| 的累计）；维度同 x。
        """
        flip_mask = (x != x0)
        cur_l0 = int(flip_mask.sum().item())
        if cur_l0 <= k:
            x_bin = torch.round(x.clamp(0.0, 1.0))
            return x_bin, flip_mask

        # 选择分数最高的 k 个翻转位置保留，其余丢弃
        flat_mask  = flip_mask.flatten()
        flat_score = score.flatten()

        flip_scores = flat_score[flat_mask]
        topk_vals, topk_idx = torch.topk(flip_scores, k, largest=True, sorted=False)

        keep = torch.zeros_like(flat_mask, dtype=torch.bool)
        flip_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        keep[flip_indices[topk_idx]] = True
        keep = keep.view_as(flip_mask)

        x_proj = x0.clone()
        x_proj[keep] = 1.0 - x0[keep]
        x_proj = torch.round(x_proj.clamp(0.0, 1.0))
        return x_proj, keep

    # @staticmethod
    # def _project_L0_binary(x0: torch.Tensor,
    #                       x: torch.Tensor,
    #                       score: torch.Tensor,
    #                       k: int):
    #     """
    #     仅当有效翻转数 > k 时投影；使用 kthvalue 找阈值，
    #     保留分数 >= 阈值 的翻转（如因并列超出 k，再在子集 topk 精修）。
    #     不强行凑满 k；未超预算直接原样返回。
    #     """
    #     with torch.no_grad():
    #         flip = _effective_flip_mask(x0, x)                  # bool
    #         num = int(flip.sum().item())
    #         if num <= k:
    #             return x.clamp(0.0, 1.0), flip

    #         flat_flip  = flip.flatten()
    #         flat_score = score.flatten()

    #         flip_scores = flat_score[flat_flip]                 # [num]
    #         # 目标：保留“分数最大”的前 k 个
    #         # kthvalue 返回第 m 小；要 top-k 最大 ⇒ 取第 (num-k+1) 小 作为阈值
    #         m = max(num - k + 1, 1)
    #         thresh = torch.kthvalue(flip_scores, k=m).values

    #         # 先用阈值筛一遍（大多数情况就刚好 ≤ k）
    #         keep_mask_flat = torch.zeros_like(flat_flip, dtype=torch.bool)
    #         flip_idx_flat  = torch.nonzero(flat_flip, as_tuple=False).squeeze(1)
    #         prelim_keep_in_flip = flip_scores >= thresh
    #         prelim_keep_idx = flip_idx_flat[prelim_keep_in_flip]
    #         keep_mask_flat[prelim_keep_idx] = True

    #         # 若因并列导致 kept > k，再在 kept 子集里做一次小范围 topk 精修
    #         kept = int(keep_mask_flat.sum().item())
    #         if kept > k:
    #             # 只对 kept 子集排序，规模远小于全量
    #             sub_scores = flat_score[keep_mask_flat]
    #             sub_vals, sub_idx = torch.topk(sub_scores, k=k, largest=True, sorted=False)
    #             new_keep = torch.zeros_like(keep_mask_flat, dtype=torch.bool)
    #             kept_positions = torch.nonzero(keep_mask_flat, as_tuple=False).squeeze(1)
    #             new_keep[kept_positions[sub_idx]] = True
    #             keep_mask_flat = new_keep

    #         keep = keep_mask_flat.view_as(flip)

    #         # 投影：丢弃的翻转还原成 x0
    #         x_out = x.clone()
    #         drop = flip & (~keep)
    #         x_out[drop] = x0[drop]
    #         return x_out.clamp(0.0, 1.0), keep


def _effective_flip_mask(x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """跨 0.5 阈值才算有效翻转"""
    return ((x0 < 0.5) & (x > 0.5)) | ((x0 >= 0.5) & (x < 0.5))

class SNNContainer(nn.Module):
    def __init__(self, model, encoder):
        super().__init__()
        self.model = model
        self.encoder = encoder
    
    def forward(self, x):
        for l in self.model.modules():
            if isinstance(l, neuron.BaseNode):
                l.train() # enable back-propagation
            else:
                l.eval()
        return torch.mean(self.model(self.encoder(x)), dim=0)

class PDSG_LIFNode(nn.Module): # modified spikingjelly's LIFNode
    def __init__(self, tau: float=2.0, decay_input=True, v_threshold: float=1.0, v_reset: float=0.0, detach_reset: bool=False):
        super().__init__()
        self.tau = tau
        self.decay_input = decay_input
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = PDSG()
        
    def forward(self, x: torch.Tensor):
        mem = 0.
        mem_pot = []
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            if t == 0:
                if self.decay_input:
                    mem = (x[t, ...] + self.v_reset) / self.tau
                else:
                    mem = x[t, ...] + self.v_reset / self.tau
            else:
                if self.decay_input:
                    mem = mem + (x[t, ...] - (mem - self.v_reset)) / self.tau
                else:
                    mem = mem + (-(mem - self.v_reset)) / self.tau + x[t, ...]
            # record membrane potential[1:t] to calculate standard deviation (following Time-Accumulated Batch Normalization (TAB))
            mem_pot.append(mem.clone().detach())
            spike = self.surrogate_function(mem - self.v_threshold, torch.stack(mem_pot, dim=0))
            if self.detach_reset:
                spike_d = spike.detach()
            else:
                spike_d = spike
            if self.v_reset is not None:
                mem = mem * (1 - spike_d) + self.v_reset * spike_d
            else:
                mem -= spike_d * self.v_threshold
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks

class SNNWrapTimeMajor(nn.Module):
    """
    包一层，兼容 SNN 输出两种形状：
      - [B, num_classes]：直接用
      - [T, B, num_classes]：按 reduction 聚合为 [B, num_classes]
    """
    def __init__(self, model: nn.Module, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("last", "mean")
        self.model = model
        self.reduction = reduction

    def forward(self, x_tbchw: torch.Tensor) -> torch.Tensor:
        out = self.model(x_tbchw)
        if out.dim() == 2:    # [B,C]
            return out
        if out.dim() == 3:    # [T,B,C]
            return out[-1] if self.reduction == "last" else out.mean(dim=0)
        raise ValueError(f"Unexpected logits shape {out.shape}")

class PGDTimeShiftAfterEncoder(nn.Module):
    """
    Spike Timing Attack（仅 after-encoder）：
      输入/输出: [T,B,C,H,W]
      在 Δ∈{-D...0...D} 的 logits 上做 PGD
      训练期: soft shift + 容量惩罚（cap_limit=1）
      终轮: 基于最大置信度的矢量化投影，保证不多占用
    """
    def __init__(
        self,
        device,
        model_without_encoder: nn.Module,   # SNN，吃 [T,B,C,H,W]
        reduction: str = "mean",            # 若 SNN 输出 [T,B,C]，如何聚合
        D: int = 1,
        steps: int = 20,
        alpha_phi: float = 1.0,
        lambda_cap: float = 5.0,
        temperature: float = 1.0,
        random_start: bool = False,
        cap_limit: float = 1.0, 
    ):
        super().__init__()
        self.device = device
        self.model = SNNWrapTimeMajor(model_without_encoder, reduction=reduction)
        self.D = int(D)
        self.K = 2 * self.D + 1
        self.steps = int(steps)
        self.alpha_phi = float(alpha_phi)
        self.lambda_cap = float(lambda_cap)
        self.temperature = float(temperature)
        self.random_start = bool(random_start)
        self.cap_limit = float(cap_limit)
        self.loss_fn = nn.CrossEntropyLoss()

    # ---- soft shift 前向（沿时间维=0）----
    def _soft_shift(self, x_tbchw, phi):
        T, B, C, H, W = x_tbchw.shape
        pi = F.softmax(phi / self.temperature, dim=-1)  # [T,B,C,H,W,K]
        x_soft = torch.zeros_like(x_tbchw)
        for k, d in enumerate(range(-self.D, self.D + 1)):
            mass = pi[..., k] * x_tbchw
            rolled = torch.roll(mass, shifts=d, dims=0)
            if d > 0: rolled[:d] = 0
            elif d < 0: rolled[d:] = 0
            x_soft += rolled
        return x_soft

    def _soft_shift_with_pi(self, x_tbchw, pi):
        # x_tbchw: [T,B,C,H,W]
        # pi:      [T,B,C,H,W,K] （已经是 ST 后的“前硬后软”权重）
        T, B, C, H, W, K = *x_tbchw.shape, pi.shape[-1]
        out = torch.zeros_like(x_tbchw)
        for k, d in enumerate(range(-self.D, self.D + 1)):
            mass = pi[..., k] * x_tbchw
            rolled = torch.roll(mass, shifts=d, dims=0)
            if d > 0: rolled[:d] = 0
            elif d < 0: rolled[d:] = 0
            out += rolled
        return out


    @torch.no_grad()
    def _final_projection(self, x_tbchw, pi):
        """
        不多占用投影（矢量化）：每个源点取 Δ* 的 pi_max 作为“得分”，
        按时间roll到目标后做 elementwise max，冲突保得分最高者。
        """
        T, B, C, H, W = x_tbchw.shape
        src = (x_tbchw > 0.5).float()       # 哪些源位置有事件
        pi_max, idx = pi.max(dim=-1)        # [T,B,C,H,W]

        planes = []
        for k, d in enumerate(range(-self.D, self.D + 1)):
            choose = (idx == k).float()
            score = src * choose * pi_max   # 只给选中Δ的源点打分
            rolled = torch.roll(score, shifts=d, dims=0)
            if d > 0: rolled[:d] = 0
            elif d < 0: rolled[d:] = 0
            planes.append(rolled)
        max_score = torch.stack(planes, dim=-1).max(dim=-1).values
        adv = (max_score > 0).float()       # winner-take-all -> 0/1
        return adv

    @torch.no_grad()
    def measure_bound_looseness(self, x_src, x_adv, D):
        """
        测量 bound 松的程度：
        - 返回违反 strict bound 的 spike 数量占比
        """
        T, B, C, H, W = x_src.shape
        src_mask = (x_src > 0.5)  # bool
        adv_mask = (x_adv > 0.5)  # bool

        violations = 0
        total_adv_spikes = adv_mask.sum().item()

        for t in range(T):
            adv_spikes_here = adv_mask[t]
            if not adv_spikes_here.any():
                continue

            src_window = torch.zeros_like(adv_spikes_here, dtype=torch.bool)
            for shift in range(-D, D + 1):
                src_t = t - shift
                if 0 <= src_t < T:
                    src_window |= src_mask[src_t]  # bool OR

            invalid_spikes = adv_spikes_here & (~src_window)  # 找到违反 bound 的 spike
            violations += invalid_spikes.sum().item()

        looseness_ratio = violations / (total_adv_spikes + 1e-8)
        return looseness_ratio



    @torch.no_grad()
    def _final_projection_strict(self, x_tbchw, pi, margin: float = 0.05):
        """
        改进版：
        1. 先按 pi_max 正常 winner-take-all 投影（不多占用）。
        2. 统计丢失的 spike 数。
        3. 从原始 src 中补回缺的 spike（优先原位补），直到 spike 数完全匹配。
        """
        T, B, C, H, W = x_tbchw.shape
        src = (x_tbchw > 0.5).float()       # 原始事件位置
        pi_max, idx = pi.max(dim=-1)        # [T,B,C,H,W]

        # Step 1: 正常 winner-take-all 投影
        planes = []
        for k, d in enumerate(range(-self.D, self.D + 1)):
            choose = (idx == k).float()
            score = src * choose * pi_max
            rolled = torch.roll(score, shifts=d, dims=0)
            if d > 0:
                rolled[:d] = 0
            elif d < 0:
                rolled[d:] = 0
            planes.append(rolled)
        max_score = torch.stack(planes, dim=-1).max(dim=-1).values
        adv = (max_score > 0).float()

        # Step 2: 统计丢失的 spike 数
        total_src_spikes = src.sum(dim=0)   # [B,C,H,W]
        total_adv_spikes = adv.sum(dim=0)   # [B,C,H,W]
        missing_mask = (total_adv_spikes < total_src_spikes)  # [B,C,H,W]
        num_missing = (total_src_spikes - total_adv_spikes)   # [B,C,H,W]

        # Step 3: 补点
        # 方案：优先在原位 src 上补（被覆盖掉的 spike），直到数量匹配
        for t in range(T):
            # 原来有 spike、现在没 spike、且该像素缺额 > 0
            to_fill = (src[t] > 0) & (adv[t] == 0) & missing_mask
            if to_fill.any():
                adv[t][to_fill] = 1.0
                # 更新缺额
                num_missing[to_fill] -= 1
                # 更新缺额 mask
                missing_mask = num_missing > 0

        return adv

    @torch.no_grad()
    def _final_projection_relaxed_packets(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        矢量化“包级宽松投影”（优化阶段用）：
          - 每个非零 time-bin 是不可分的“包”，值 v 原样搬运；
          - 仅允许 |Δ|≤D；
          - 每个源先选自己的 best-Δ（argmax_k π），把该候选 roll 到目标；
          - 目标端按分数取 elementwise max（冲突保分数最高的那一个）；
          - 被选中的源包按其 best-Δ 落到目标；其余源包直接回原位；
          - 为了保证回原位一定可行，禁止“移动落到别的源的原位”（即 d≠0 时目标不能是任何 src 原位）。
        返回 adv: [T,B,C,H,W]，值不变、不会重合、不会丢包；但相对严格版可能“少移动”（不再尝试次优 Δ）。
        """
        T, B, C, H, W = x_tbchw.shape
        src_mask = (x_tbchw > 0).float()   # 哪些位置有“包”
        src_val  = x_tbchw                 # 包的值 v（不可分）

        # 每个源的最佳 Δ 索引（沿 K 维）
        pi_max, idx = pi.max(dim=-1)       # [T,B,C,H,W], [T,B,C,H,W]

        score_planes = []
        val_planes   = []

        # 目标禁止“占用别人的原位”（只有 d=0 自己留在原位允许）
        target_reserved = src_mask         # [T,B,C,H,W]

        for k, d in enumerate(range(-self.D, self.D + 1)):
            choose = (idx == k).float()                        # 只保留每个源自己的 best-Δ
            # 分数=该源的 pi_max；值=该源的 v
            score_src = choose * pi_max * src_mask             # [T,B,C,H,W]
            val_src   = choose * src_val                       # [T,B,C,H,W]

            rolled_s = torch.roll(score_src, shifts=d, dims=0)
            rolled_v = torch.roll(val_src,   shifts=d, dims=0)

            # 边界清零
            if d > 0:
                rolled_s[:d] = 0; rolled_v[:d] = 0
            elif d < 0:
                rolled_s[d:] = 0; rolled_v[d:] = 0

            # 禁止移动落到“别的源”的原位（保证回原位总能成功）
            if d != 0:
                rolled_s = rolled_s * (1.0 - target_reserved)
                rolled_v = rolled_v * (1.0 - target_reserved)

            score_planes.append(rolled_s)
            val_planes.append(rolled_v)

        # 目标端：在 K 个候选中选分数最大的那个（每个目标最多接收1个包）
        S = torch.stack(score_planes, dim=-1)                  # [T,B,C,H,W,K]
        V = torch.stack(val_planes,   dim=-1)                  # [T,B,C,H,W,K]
        argK = S.argmax(dim=-1)                                # [T,B,C,H,W]
        adv_moved = torch.gather(V, dim=-1, index=argK.unsqueeze(-1)).squeeze(-1)

        # 反投影：标记哪些源已被选中（被选中的“包”不再回原位）
        onehot = torch.zeros_like(S)
        onehot.scatter_(-1, argK.unsqueeze(-1), 1.0)           # [T,B,C,H,W,K]
        used_src = torch.zeros_like(src_mask)
        for k, d in enumerate(range(-self.D, self.D + 1)):
            sel_tgt_k = onehot[..., k]                         # 该 k 在目标处被选中的位置
            back = torch.roll(sel_tgt_k, shifts=-d, dims=0)    # 拉回源时刻
            if d > 0:
                back[-d:] = 0
            elif d < 0:
                back[:-d] = 0
            # 只有“这个源的 best-Δ 恰为 k”的才算真正被用到
            used_src = used_src + back * (idx == k).float() * src_mask
        used_src = (used_src > 0.5).float()

        # 未被选中的源包回原位；因为禁止了落到别人的原位，所以这里不会与 adv_moved 冲突
        leftover = src_val * (1.0 - used_src)
        adv = adv_moved + leftover
        return adv

    @torch.no_grad()
    def _final_projection_packets_greedy(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        批量向量化包级贪心（严格 |Δ|≤D；不重合；值不变；不丢包）：
        - 把 (B,C,H,W) 展开成 N 维，所有时间线并行处理；
        - 每轮：为所有“未放置”的源并行计算“在当前可用目标下”的最佳 Δ 与分数；
          选择全局分数最大的那个源，提交放置；更新占用/保留，再进入下一轮；
        - 与你之前严格版行为一致，但去掉了 (b,c,h,w) 的 Python 循环。
        """
        T, B, C, H, W = x_tbchw.shape
        N = B * C * H * W
        device = x_tbchw.device
        dtype  = x_tbchw.dtype

        # 展开到 [T,N] / [T,N,K]
        x_vals = x_tbchw.reshape(T, N)              # 每个 time-bin 的“包值 v”，v>0 视为一个不可分包
        src_mask = (x_vals > 0)                     # [T,N] 哪些位置有包
        pi_flat = pi.reshape(T, N, -1)              # [T,N,K]
        K = pi_flat.shape[-1]
        assert K == 2 * self.D + 1, "pi 的最后一维应为 2D+1"

        # 输出与状态
        adv = torch.zeros_like(x_vals)              # [T,N]
        occupied = torch.zeros(T, N, dtype=torch.bool, device=device)   # 目标占用
        reserved = src_mask.clone()                 # 初始：所有源的原位都保留（禁止别人占）
        unplaced = src_mask.clone()                 # 还未落地的源（包）

        # 便捷工具：把目标侧可用 mask（在 t2）平移到源侧（在 t）
        def target2source(mask_tgt: torch.Tensor, d: int) -> torch.Tensor:
            # 输入/输出: [T,N] -> [T,N]
            out = torch.roll(mask_tgt, shifts=-d, dims=0)
            if d > 0:    # 往上滚，底部越界无效
                out[-d:, :] = False
            elif d < 0:  # 往下滚，顶部越界无效
                out[:(-d), :] = False
            return out

        # 常量：-inf（用来 mask 不可选项）
        NEG_INF = torch.finfo(dtype).min

        # 主循环：每次只提交 1 个全局最优包，直到所有源落地
        # 复杂度约 O(M * K * T)，但无 (B,C,H,W) 的 Python 循环；T/K 通常很小
        while unplaced.any():
            # 目标侧的“当前可用”：未占 & (非保留 或 Δ=0 特例)
            # 先构建每个 k 的 目标可用 mask，再回到源侧得到“该源用此 k 是否可用”
            best_score = torch.full((T, N), NEG_INF, device=device, dtype=dtype)
            best_k     = torch.zeros((T, N), dtype=torch.long, device=device)

            for k, d in enumerate(range(-self.D, self.D + 1)):
                # 目标侧：允许落点 = 未占 & (未保留 或 Δ=0)
                allow_tgt = (~occupied) & ( (~reserved) | (d == 0) )
                # 映射回源侧：只有源存在且其用该 Δ 的目标是“允许”的，才可考虑
                allow_src = target2source(allow_tgt, d) & unplaced

                # 候选分数（不允许的位置置 -inf）
                score_k = torch.where(allow_src, pi_flat[..., k], torch.as_tensor(NEG_INF, device=device, dtype=dtype))
                # 更新每个源的“当前最佳 k”
                take = score_k > best_score
                best_score = torch.where(take, score_k, best_score)
                best_k     = torch.where(take, torch.as_tensor(k, device=device), best_k)

            # 如果所有未放置源都找不到可用候选（理论上不该发生），统一回原位兜底
            if (best_score == NEG_INF).all():
                # 回原位：原位应始终保留，必成功
                adv = adv + (x_vals * unplaced.float())
                occupied = occupied | (unplaced)   # 原位也当作“占用”
                reserved = reserved & (~unplaced)  # 释放刚落地的原位
                unplaced.zero_()
                break

            # 选择“全局分数最大”的那个源 (t*, n*)
            # 仅在未放置源上取 argmax
            mask_score = torch.where(unplaced, best_score, torch.as_tensor(NEG_INF, device=device, dtype=dtype))
            flat_idx = mask_score.view(-1).argmax()
            t_sel = (flat_idx // N).item()
            n_sel = (flat_idx %  N).item()
            k_sel = int(best_k[t_sel, n_sel].item())
            d_sel = k_sel - self.D
            t2_sel = t_sel + d_sel

            v = x_vals[t_sel, n_sel]

            placed = False
            # 尝试落到最佳目标（严格检查：未占 & 非保留 或 Δ=0）
            if (0 <= t2_sel < T) and (not occupied[t2_sel, n_sel]) and ( (not reserved[t2_sel, n_sel]) or (d_sel == 0) ):
                adv[t2_sel, n_sel] = v
                occupied[t2_sel, n_sel] = True
                placed = True
                # 若成功落到 t2!=t，释放原位；t2==t 也算落地，照样释放
                reserved[t_sel, n_sel] = False
            else:
                # 备选：在窗口内找最近可用（未占 & 非保留）的目标；若仍不可用，回原位
                done = False
                for r in range(1, self.D + 1):
                    for t3 in (t_sel - r, t_sel + r):
                        if 0 <= t3 < T and (not occupied[t3, n_sel]) and (not reserved[t3, n_sel]):
                            adv[t3, n_sel] = v
                            occupied[t3, n_sel] = True
                            reserved[t_sel, n_sel] = False
                            done = True
                            placed = True
                            break
                    if done:
                        break
                if not placed:
                    # 回原位（始终可行，因为原位一直处于保留状态）
                    adv[t_sel, n_sel] = v
                    occupied[t_sel, n_sel] = True
                    reserved[t_sel, n_sel] = False
                    placed = True

            # 标记该源落地
            unplaced[t_sel, n_sel] = False

        # 还原形状
        return adv.view(T, B, C, H, W)


    def _capacity_penalty(self, x_soft):
        over = torch.clamp(x_soft - self.cap_limit, min=0.0)
        return (over * over).mean()

    # 新增：占用惩罚（与“包级、不重合”匹配）
    def _occupancy_penalty(self, x_src_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        x_src_tbchw: [T,B,C,H,W] 原始输入（包掩码来自它）
        pi:          [T,B,C,H,W,K] 每个包选择各 Δ 的概率
        return: 标量；惩罚“期望占用 > 1”
        """
        T, B, C, H, W = x_src_tbchw.shape
        src_mask = (x_src_tbchw > 0).float()  # 只关心“有包没有”，不看 v 大小
        occ = x_src_tbchw.new_zeros((T, B, C, H, W))
        for k, d in enumerate(range(-self.D, self.D + 1)):
            # 该 Δ 下，每个源包以概率 pi[...,k] 试图落到 t+d
            contrib = src_mask * pi[..., k]                # [T,B,C,H,W]
            rolled  = torch.roll(contrib, shifts=d, dims=0)
            if d > 0: rolled[:d] = 0
            elif d < 0: rolled[d:] = 0
            occ += rolled
        overflow = (occ - self.cap_limit).clamp_min(0.0)             # 期望“包数”超 1 的部分
        return (overflow * overflow).mean()


    def forward(self, x_enc_tbchw: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # x_enc_tbchw: [T,B,C,H,W]（已经过 DVSSignEncoder）
        x_enc_tbchw = x_enc_tbchw.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        assert x_enc_tbchw.dim() == 5, f"Expect [T,B,C,H,W], got {x_enc_tbchw.shape}"

        T, B, C, H, W = x_enc_tbchw.shape
        phi = torch.zeros(T, B, C, H, W, self.K, device=self.device,
                          dtype=x_enc_tbchw.dtype, requires_grad=True)
        if self.random_start:
            phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)

        # project_func = self._final_projection_strict
        project_func = self._final_projection_packets_greedy
        for _ in range(self.steps):
            tau = self.temperature        
            pi = F.softmax(phi / tau, dim=-1)

            x_soft = self._soft_shift_with_pi(x_enc_tbchw, pi)      # [T,B,C,H,W]
            # x_hard = project_func(x_enc_tbchw, pi)     # [T,B,C,H,W], 0/1
            with torch.no_grad():
                x_hard = self._final_projection_relaxed_packets(x_enc_tbchw, pi).detach()
            x_pil = x_hard + (x_soft - x_soft.detach())
            logits = self.model(x_pil)  
            # logits = self.model(x_soft)                      # [B,C] 或聚合后 [B,C]
            ce = self.loss_fn(logits, labels)
            # cap = self._capacity_penalty(x_soft)
            cap = self._occupancy_penalty(x_enc_tbchw, pi)
            loss = ce - self.lambda_cap * cap
            # print (ce)
            # print (cap)

            grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                phi += self.alpha_phi * grad.sign()
                phi.clamp_(-10.0, 10.0)
                phi.requires_grad_(True)

        with torch.no_grad():
            pi = F.softmax(phi / self.temperature, dim=-1)
            adv_tbchw = project_func(x_enc_tbchw, pi)  # [T,B,C,H,W]
        return adv_tbchw


class PGDTimeShiftAfterEncoder_Lowgpu(nn.Module):
    """
    Spike Timing Attack（仅 after-encoder）：
      输入/输出: [T,B,C,H,W]
      在 Δ∈{-D...0...D} 的 logits 上做 PGD
      训练期: soft shift + 容量惩罚（cap_limit=1）
      终轮: 基于最大置信度的矢量化投影，保证不多占用
    """
    def __init__(
        self,
        device,
        model_without_encoder: nn.Module,   # SNN，吃 [T,B,C,H,W]
        reduction: str = "mean",            # 若 SNN 输出 [T,B,C]，如何聚合
        D: int = 1,
        steps: int = 20,
        alpha_phi: float = 1.0,
        lambda_cap: float = 20.0,
        temperature: float = 1.0,
        random_start: bool = False,
        cap_limit: float = 1.0, 
    ):
        super().__init__()
        self.device = device
        self.model = SNNWrapTimeMajor(model_without_encoder, reduction=reduction)
        self.D = int(D)
        self.K = 2 * self.D + 1
        self.steps = int(steps)
        self.alpha_phi = float(alpha_phi)
        self.lambda_cap = float(lambda_cap)
        self.temperature = float(temperature)
        self.random_start = bool(random_start)
        self.cap_limit = float(cap_limit)
        self.loss_fn = nn.CrossEntropyLoss()

    # ========== 1) 活跃线工具 ==========
    @staticmethod
    def _flatten_tbchw(x):  # [T,B,C,H,W] -> (x_flat[T,N], B,C,H,W,N)
        T, B, C, H, W = x.shape
        N = B*C*H*W
        return x.view(T, N), (B, C, H, W, N)

    @staticmethod
    def _active_index(x_flat):  # x_flat: [T,N]
        # 整条时间轴上是否有包（或非零值）
        active = (x_flat != 0).any(dim=0)  # [N] bool
        idx = active.nonzero(as_tuple=False).squeeze(1)  # [N_active]
        return active, idx

    # ========== 2) soft shift（活跃线版） ==========
    def _soft_shift_with_pi_active(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        只在活跃时间线计算 soft shift；行为与 _soft_shift_with_pi 完全一致
        x_tbchw: [T,B,C,H,W], pi: [T,B,C,H,W,K]
        """
        T = x_tbchw.shape[0]
        K = pi.shape[-1]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)   # [T,N]
        pi_flat = pi.view(T, shape_info[-1], K)             # [T,N,K]

        active_mask, idx = self._active_index(x_flat)       # [N], [N_active]
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]                             # [T,n]
        pi_act = pi_flat[:, idx, :]                         # [T,n,K]

        out_act = torch.zeros_like(x_act)                   # [T,n]
        for k, d in enumerate(range(-self.D, self.D + 1)):
            mass = pi_act[..., k] * x_act                   # [T,n]
            rolled = torch.roll(mass, shifts=d, dims=0)
            if d > 0:  rolled[:d] = 0
            elif d < 0: rolled[d:] = 0
            out_act += rolled

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = out_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ========== 3) 宽松投影（活跃线 + 流式逐Δ，省内存） ==========
    @torch.no_grad()
    def _final_projection_relaxed_packets_active(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        宽松投影：best-Δ → 目标端取 max；未选中回原位。
        仅对活跃线计算；禁止 d!=0 占用他人原位；值守恒、不重合、|Δ|≤D。
        """
        T = x_tbchw.shape[0]
        K = pi.shape[-1]
        dev, dtype = x_tbchw.device, x_tbchw.dtype

        x_flat, shape_info = self._flatten_tbchw(x_tbchw)      # [T,N]
        pi_flat = pi.view(T, shape_info[-1], K)                # [T,N,K]
        active_mask, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]                                # [T,n]
        pi_act = pi_flat[:, idx, :]                            # [T,n,K]

        src_mask = (x_act > 0)                                 # [T,n] bool
        src_val  = x_act                                       # [T,n]
        pi_max, idx_k = pi_act.max(dim=-1)                     # [T,n]

        # 目标端“流式”最大化（不 stack）
        best_s = torch.full_like(x_act, torch.finfo(dtype).min)
        best_v = torch.zeros_like(x_act)
        best_k = torch.zeros_like(idx_k, dtype=torch.long)
        target_reserved = src_mask

        for k, d in enumerate(range(-self.D, self.D + 1)):
            choose  = (idx_k == k) & src_mask
            score_s = torch.where(choose, pi_max, torch.zeros(1, device=dev, dtype=dtype))
            value_s = torch.where(choose, src_val, torch.zeros(1, device=dev, dtype=dtype))

            rolled_s = torch.roll(score_s, shifts=d, dims=0)
            rolled_v = torch.roll(value_s, shifts=d, dims=0)
            if d > 0:  rolled_s[:d] = 0; rolled_v[:d] = 0
            elif d < 0: rolled_s[d:] = 0; rolled_v[d:] = 0

            if d != 0:
                allow = (~target_reserved)
                rolled_s = rolled_s * allow
                rolled_v = rolled_v * allow

            take   = rolled_s > best_s
            best_s = torch.where(take, rolled_s, best_s)
            best_v = torch.where(take, rolled_v, best_v)
            best_k = torch.where(take, torch.as_tensor(k, device=dev, dtype=torch.long), best_k)

        adv_moved = best_v

        # 标记被选中源；未选中的回原位
        used_src = torch.zeros_like(src_mask)
        for k, d in enumerate(range(-self.D, self.D + 1)):
            sel_tgt_k = (best_k == k) & (best_s > 0)
            back = torch.roll(sel_tgt_k, shifts=-d, dims=0)
            if d > 0:  back[-d:] = False
            elif d < 0: back[:-d] = False
            used_src |= back & (idx_k == k) & src_mask

        leftover = torch.where(~used_src, src_val, torch.zeros(1, device=dev, dtype=dtype))
        adv_act = adv_moved + leftover                          # [T,n]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ========== 4) 严格投影（活跃线版，tight + no-loss） ==========
    # @torch.no_grad()
    # def _final_projection_packets_greedy_active(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    #     """
    #     严格投影（tight bound、值守恒、不重合、不丢包），仅对活跃线计算。
    #     行为与你当前的批量严格版一致。
    #     """
    #     T = x_tbchw.shape[0]
    #     K = pi.shape[-1]
    #     dev, dtype = x_tbchw.device, x_tbchw.dtype

    #     x_flat, shape_info = self._flatten_tbchw(x_tbchw)      # [T,N]
    #     pi_flat = pi.view(T, shape_info[-1], K)                # [T,N,K]
    #     active_mask, idx = self._active_index(x_flat)
    #     if idx.numel() == 0:
    #         return torch.zeros_like(x_tbchw)

    #     x_act  = x_flat[:, idx]                                # [T,n]
    #     pi_act = pi_flat[:, idx, :]                            # [T,n,K]
    #     n = x_act.shape[1]

    #     adv_act  = torch.zeros_like(x_act)
    #     occupied = torch.zeros(T, n, dtype=torch.bool, device=dev)
    #     reserved = (x_act > 0)
    #     unplaced = reserved.clone()
    #     NEG_INF  = torch.finfo(dtype).min

    #     def tgt2src(mask_tgt, d):
    #         out = torch.roll(mask_tgt, shifts=-d, dims=0)
    #         if d > 0:  out[-d:] = False
    #         elif d < 0: out[:(-d)] = False
    #         return out

    #     while unplaced.any():
    #         best_score = torch.full((T, n), NEG_INF, device=dev, dtype=dtype)
    #         best_k     = torch.zeros((T, n), dtype=torch.long, device=dev)

    #         for k, d in enumerate(range(-self.D, self.D + 1)):
    #             allow_tgt = (~occupied) & ((~reserved) | (d == 0))
    #             allow_src = tgt2src(allow_tgt, d) & unplaced
    #             score_k   = torch.where(allow_src, pi_act[..., k], torch.as_tensor(NEG_INF, device=dev, dtype=dtype))
    #             take      = score_k > best_score
    #             best_score = torch.where(take, score_k, best_score)
    #             best_k     = torch.where(take, torch.as_tensor(k, device=dev), best_k)

    #         if (best_score == NEG_INF).all():
    #             adv_act += x_act * unplaced.float()
    #             occupied |= unplaced
    #             reserved &= (~unplaced)
    #             unplaced.zero_()
    #             break

    #         mask_score = torch.where(unplaced, best_score, torch.as_tensor(NEG_INF, device=dev, dtype=dtype))
    #         flat = mask_score.view(-1).argmax()
    #         t_sel = int(flat // n)
    #         i_sel = int(flat %  n)
    #         k_sel = int(best_k[t_sel, i_sel].item())
    #         d_sel = k_sel - self.D
    #         t2_sel = t_sel + d_sel
    #         v = x_act[t_sel, i_sel]

    #         placed = False
    #         if (0 <= t2_sel < T) and (not occupied[t2_sel, i_sel]) and ((not reserved[t2_sel, i_sel]) or (d_sel == 0)):
    #             adv_act[t2_sel, i_sel] = v
    #             occupied[t2_sel, i_sel] = True
    #             reserved[t_sel, i_sel] = False
    #             placed = True
    #         else:
    #             done = False
    #             for r in range(1, self.D + 1):
    #                 for t3 in (t_sel - r, t_sel + r):
    #                     if 0 <= t3 < T and (not occupied[t3, i_sel]) and (not reserved[t3, i_sel]):
    #                         adv_act[t3, i_sel] = v
    #                         occupied[t3, i_sel] = True
    #                         reserved[t_sel, i_sel] = False
    #                         done = True
    #                         placed = True
    #                         break
    #                 if done: break
    #             if not placed:
    #                 adv_act[t_sel, i_sel] = v
    #                 occupied[t_sel, i_sel] = True
    #                 reserved[t_sel, i_sel] = False

    #         unplaced[t_sel, i_sel] = False

    #     out_flat = torch.zeros_like(x_flat)
    #     out_flat[:, idx] = adv_act
    #     B, C, H, W, N = shape_info
    #     return out_flat.view(T, B, C, H, W)
    @torch.no_grad()
    def _final_projection_packets_greedy_active(
        self,
        x_tbchw: torch.Tensor,
        pi: torch.Tensor,
        return_disp: bool = False,
    ):
        """
        严格投影（tight bound、值守恒、不重合、不丢包），仅对活跃线计算。
        额外：返回每个“源点”的实际位移 Δ ∈ [-D, D]（源坐标系下）。
        """
        T = x_tbchw.shape[0]
        K = pi.shape[-1]
        dev, dtype = x_tbchw.device, x_tbchw.dtype

        x_flat, shape_info = self._flatten_tbchw(x_tbchw)      # [T,N]
        pi_flat = pi.view(T, shape_info[-1], K)                # [T,N,K]
        active_mask, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            out_zero = torch.zeros_like(x_tbchw)
            if return_disp:
                return out_zero, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_zero

        x_act  = x_flat[:, idx]                                # [T,n]
        pi_act = pi_flat[:, idx, :]                            # [T,n,K]
        n = x_act.shape[1]

        adv_act  = torch.zeros_like(x_act)                     # 目标上的值
        occupied = torch.zeros(T, n, dtype=torch.bool, device=dev)
        reserved = (x_act > 0)                                 # 源是否有包
        unplaced = reserved.clone()

        # 记录每个“源点”的位移 Δ（源坐标下），int16 足够覆盖常见 D
        disp_act = torch.zeros(T, n, device=dev, dtype=torch.int16)

        NEG_INF  = torch.finfo(dtype).min

        def tgt2src(mask_tgt, d):
            out = torch.roll(mask_tgt, shifts=-d, dims=0)
            if d > 0:
                out[-d:] = False
            elif d < 0:
                out[:(-d)] = False
            return out

        while unplaced.any():
            best_score = torch.full((T, n), NEG_INF, device=dev, dtype=dtype)
            best_k     = torch.zeros((T, n), dtype=torch.long, device=dev)

            for k, d in enumerate(range(-self.D, self.D + 1)):
                allow_tgt = (~occupied) & ((~reserved) | (d == 0))
                allow_src = tgt2src(allow_tgt, d) & unplaced
                score_k   = torch.where(allow_src, pi_act[..., k], torch.as_tensor(NEG_INF, device=dev, dtype=dtype))
                take      = score_k > best_score
                best_score = torch.where(take, score_k, best_score)
                best_k     = torch.where(take, torch.as_tensor(k, device=dev), best_k)

            # 若没有可放置位置，未放置的回原位，Δ=0
            if (best_score == NEG_INF).all():
                adv_act += x_act * unplaced.float()
                # disp_act 对未放置的仍为 0
                occupied |= unplaced
                reserved &= (~unplaced)
                unplaced.zero_()
                break

            # 选取全局最优的一个源点来放置
            mask_score = torch.where(unplaced, best_score, torch.as_tensor(NEG_INF, device=dev, dtype=dtype))
            flat = mask_score.view(-1).argmax()
            t_sel = int(flat // n)
            i_sel = int(flat %  n)
            k_sel = int(best_k[t_sel, i_sel].item())
            d_sel = k_sel - self.D
            t2_sel = t_sel + d_sel
            v = x_act[t_sel, i_sel]

            placed = False
            # 优先放在期望位移 d_sel 的目标时间
            if (0 <= t2_sel < T) and (not occupied[t2_sel, i_sel]) and ((not reserved[t2_sel, i_sel]) or (d_sel == 0)):
                adv_act[t2_sel, i_sel] = v
                occupied[t2_sel, i_sel] = True
                reserved[t_sel, i_sel] = False
                disp_act[t_sel, i_sel] = int(t2_sel - t_sel)   # 记录 Δ ∈ [-D, D]
                placed = True
            else:
                # 备选：在半径 r 内寻找最近可用目标时间
                done = False
                for r in range(1, self.D + 1):
                    for t3 in (t_sel - r, t_sel + r):
                        if 0 <= t3 < T and (not occupied[t3, i_sel]) and (not reserved[t3, i_sel]):
                            adv_act[t3, i_sel] = v
                            occupied[t3, i_sel] = True
                            reserved[t_sel, i_sel] = False
                            disp_act[t_sel, i_sel] = int(t3 - t_sel)  # 记录实际 Δ
                            done = True
                            placed = True
                            break
                    if done:
                        break
                # 若仍未放置，回原位，Δ=0
                if not placed:
                    adv_act[t_sel, i_sel] = v
                    occupied[t_sel, i_sel] = True
                    reserved[t_sel, i_sel] = False
                    disp_act[t_sel, i_sel] = 0

            unplaced[t_sel, i_sel] = False

        # 写回原形状
        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        adv_tbchw = out_flat.view(T, B, C, H, W)

        # 位移还原到 [T,B,C,H,W]
        out_disp_flat = torch.zeros_like(x_flat, dtype=torch.int16)
        out_disp_flat[:, idx] = disp_act
        disp_tbchw = out_disp_flat.view(T, B, C, H, W)

        if return_disp:
            return adv_tbchw, disp_tbchw
        return adv_tbchw


    # ========== 5) 期望占用惩罚（活跃线版） ==========
    def _occupancy_penalty_active(self, x_src_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        与原 _occupancy_penalty 等价，但只在活跃线计算，显存更小
        """
        T = x_src_tbchw.shape[0]
        K = pi.shape[-1]
        x_flat, shape_info = self._flatten_tbchw(x_src_tbchw)  # [T,N]
        pi_flat = pi.view(T, shape_info[-1], K)                # [T,N,K]
        active_mask, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros((), device=x_src_tbchw.device, dtype=x_src_tbchw.dtype)

        x_act  = x_flat[:, idx]                                # [T,n]
        pi_act = pi_flat[:, idx, :]                            # [T,n,K]

        src_mask = (x_act > 0).float()
        occ = torch.zeros_like(x_act)
        for k, d in enumerate(range(-self.D, self.D + 1)):
            contrib = src_mask * pi_act[..., k]                # [T,n]
            rolled  = torch.roll(contrib, shifts=d, dims=0)
            if d > 0:  rolled[:d] = 0
            elif d < 0: rolled[d:] = 0
            occ += rolled
        # 在 _occupancy_penalty_active 的最后：
        overflow = (occ - self.cap_limit).clamp_min(0.0)
        num_all     = x_src_tbchw.shape[0] * x_src_tbchw.shape[1] * x_src_tbchw.shape[2] * x_src_tbchw.shape[3] * x_src_tbchw.shape[4]  # T*B*C*H*W
        num_active  = overflow.numel()   # 实际是 T * N_active
        scale = (num_active / num_all)   # 把“活跃均值”缩回“全量均值”的量级
        return (overflow * overflow).mean() * scale
        # overflow = (occ - self.cap_limit).clamp_min(0.0)             # 期望“包数”超 1 的部分
        # return (overflow * overflow).mean()

    # ========== 6) forward：使用“活跃线 + no_grad” ==========
    def forward(self, x_enc_tbchw: torch.Tensor, labels: torch.Tensor, return_disp: bool = False, use_PIL: bool = True, use_cap: bool = True, use_penalty: bool = True, target_label: int = -1):
        x_enc_tbchw = x_enc_tbchw.to(self.device)
        labels = labels.to(self.device)

        T, B, C, H, W = x_enc_tbchw.shape
        # phi 仍用 dense，但只对“有包”的位置反传（屏蔽无意义梯度）
        phi = torch.zeros(T, B, C, H, W, self.K, device=self.device, dtype=x_enc_tbchw.dtype)
        phi[..., self.D - 1 : self.D + 2] = 4
        phi.requires_grad_(True)

        # mask: [T,B,C,H,W,1]
        src_mask = (x_enc_tbchw != 0).unsqueeze(-1)
        def _mask_hook(grad):
            return grad * src_mask.float()
        phi.register_hook(_mask_hook)

        if self.random_start:
            phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)

        # 训练期：soft 用活跃线版；硬投影用“宽松-活跃线版 + PIL”；cap 用活跃线版
        optimization_steps = self.steps if target_label < 0 else 2 * self.steps
        for t in range(optimization_steps):
            tau = self.temperature
            # 对无源位置，softmax 输入置 -inf，避免溢出/无意义概率
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi = F.softmax(phi_masked / tau, dim=-1)
            if use_cap:
                cap = self._occupancy_penalty_active(x_enc_tbchw, pi)
            else:
                cap = 0

            x_soft = self._soft_shift_with_pi_active(x_enc_tbchw, pi)
            if use_PIL:
                with torch.no_grad():
                    x_hard = self._final_projection_relaxed_packets_active(x_enc_tbchw, pi)
                    # x_hard = self._final_projection_packets_greedy_active(x_enc_tbchw, pi)
                x_pil = x_hard + (x_soft - x_soft.detach())
            else:
                x_pil = x_soft

            logits = self.model(x_pil)

            if target_label < 0:
                ce = self.loss_fn(logits, labels)
                # print (cap)
                loss = ce - self.lambda_cap * cap
            else:
                ce = self.loss_fn(logits, target_label * torch.ones_like(labels))
                loss = -ce - self.lambda_cap * cap

            # print (cap)
            print (ce)

            grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                phi += self.alpha_phi * grad.sign()
                phi.clamp_(-10.0, 10.0)
                phi.requires_grad_(True)

        # 最终输出：严格投影（活跃线版，tight + no-loss）
        final_projection = self._final_projection_packets_greedy_active if target_label < 0 else self._final_projection_relaxed_packets_active
        with torch.no_grad():
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(torch.finfo(phi.dtype).min, device=phi.device, dtype=phi.dtype))
            pi = F.softmax(phi_masked / self.temperature, dim=-1)
            if return_disp:
                adv_tbchw, delta_tbchw = final_projection(x_enc_tbchw, pi, return_disp=return_disp)
                return adv_tbchw, delta_tbchw 
            else:
                adv_tbchw = final_projection(x_enc_tbchw, pi)
                return adv_tbchw


# class PGDTimeShiftAfterEncoder_Lowgpu(nn.Module):
#     """
#     Spike Timing Attack（仅 after-encoder）：
#       输入/输出: [T,B,C,H,W]
#       在 Δ∈{-D...0...D} 的 logits 上做 PGD
#       训练期: soft shift + 容量惩罚（cap_limit=1）
#       终轮: 严格投影（tight + no-loss），保证不多占用
#     """
#     def __init__(
#         self,
#         device,
#         model_without_encoder: nn.Module,   # SNN，吃 [T,B,C,H,W]
#         reduction: str = "mean",            # 若 SNN 输出 [T,B,C]，如何聚合
#         D: int = 1,
#         steps: int = 20,
#         alpha_phi: float = 1.0,
#         lambda_cap: float = 5.0,
#         temperature: float = 1.0,
#         random_start: bool = False,
#         cap_limit: float = 1.0,
#     ):
#         super().__init__()
#         self.device = device
#         self.model = SNNWrapTimeMajor(model_without_encoder, reduction=reduction)
#         self.D = int(D)
#         self.K = 2 * self.D + 1
#         self.steps = int(steps)
#         self.alpha_phi = float(alpha_phi)
#         self.lambda_cap = float(lambda_cap)
#         self.temperature = float(temperature)
#         self.random_start = bool(random_start)
#         self.cap_limit = float(cap_limit)
#         self.loss_fn = nn.CrossEntropyLoss()

#     # ---------- 工具 ----------
#     @staticmethod
#     def _flatten_tbchw(x):  # [T,B,C,H,W] -> (x_flat[T,N], B,C,H,W,N)
#         T, B, C, H, W = x.shape
#         N = B*C*H*W
#         return x.view(T, N), (B, C, H, W, N)

#     @staticmethod
#     def _active_index(x_flat):  # x_flat: [T,N]
#         active = (x_flat != 0).any(dim=0)  # [N] bool
#         idx = active.nonzero(as_tuple=False).squeeze(1)  # [N_active]
#         return active, idx

#     # ---------- 2) soft shift（活跃线版） ----------
#     def _soft_shift_with_pi_active(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
#         """
#         只在活跃时间线计算 soft shift；行为与 _soft_shift_with_pi 完全一致
#         x_tbchw: [T,B,C,H,W], pi: [T,B,C,H,W,K]
#         """
#         T = x_tbchw.shape[0]
#         K = pi.shape[-1]
#         x_flat, shape_info = self._flatten_tbchw(x_tbchw)   # [T,N]
#         pi_flat = pi.view(T, shape_info[-1], K)             # [T,N,K]

#         _, idx = self._active_index(x_flat)                 # [N_active]
#         if idx.numel() == 0:
#             return torch.zeros_like(x_tbchw)

#         x_act  = x_flat[:, idx]                             # [T,n]
#         pi_act = pi_flat[:, idx, :]                         # [T,n,K]

#         out_act = torch.zeros_like(x_act)                   # [T,n]
#         for k, d in enumerate(range(-self.D, self.D + 1)):
#             mass = pi_act[..., k] * x_act                   # [T,n]
#             rolled = torch.roll(mass, shifts=d, dims=0)
#             if d > 0:  rolled[:d] = 0
#             elif d < 0: rolled[d:] = 0
#             out_act += rolled

#         out_flat = torch.zeros_like(x_flat)
#         out_flat[:, idx] = out_act
#         B, C, H, W, N = shape_info
#         return out_flat.view(T, B, C, H, W)

#     # ---------- 3) 宽松投影（value-aware，活跃线；已修复一源多投） ----------
#     @torch.no_grad()
#     def _final_projection_relaxed_packets_active(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
#         """
#         宽松投影（不 tight）：值守恒、不重合、|Δ|≤D。
#         修复：先对每个源做一次 argmax_k(v·π_k)（一源最多一个Δ），再把候选 roll 到目标端竞争；
#         未中选的源回原位。
#         """
#         T = x_tbchw.shape[0]
#         x_flat, shape_info = self._flatten_tbchw(x_tbchw)      # [T,N]
#         pi_flat = pi.view(T, shape_info[-1], self.K)           # [T,N,K]
#         _, idx = self._active_index(x_flat)
#         if idx.numel() == 0:
#             return torch.zeros_like(x_tbchw)

#         x_act  = x_flat[:, idx]                                # [T,n]
#         pi_act = pi_flat[:, idx, :]                            # [T,n,K]

#         dev, dtype = x_act.device, x_act.dtype
#         src_mask = (x_act > 0)                                 # [T,n] bool
#         src_val  = x_act                                       # [T,n]

#         # 每源选择一个最佳 k*
#         score_all = src_val.unsqueeze(-1) * pi_act             # [T,n,K]
#         pi_max, idx_k = score_all.max(dim=-1)                  # [T,n]  v·π 最大值及其 k*

#         # 目标端：以“原地不动”的分数为初值
#         stay_score = src_val * pi_act[..., self.D]             # [T,n]
#         best_s = stay_score.clone()
#         best_v = x_act.clone()
#         best_k = torch.full_like(idx_k, self.D, dtype=torch.long)
#         target_reserved = src_mask.clone()                     # d!=0 禁止落到他人原位

#         # 仅用 k* 候选进行目标端竞争
#         for k, d in enumerate(range(-self.D, self.D + 1)):
#             choose  = (idx_k == k) & src_mask
#             score_s = torch.where(choose, pi_max, torch.zeros(1, device=dev, dtype=dtype))
#             value_s = torch.where(choose, src_val, torch.zeros(1, device=dev, dtype=dtype))

#             rolled_s = torch.roll(score_s, shifts=d, dims=0)
#             rolled_v = torch.roll(value_s, shifts=d, dims=0)
#             if d > 0:  rolled_s[:d] = 0; rolled_v[:d] = 0
#             elif d < 0: rolled_s[d:] = 0; rolled_v[d:] = 0

#             if d != 0:
#                 allow = (~target_reserved)
#                 rolled_s = rolled_s * allow
#                 rolled_v = rolled_v * allow

#             take   = rolled_s > best_s
#             best_s = torch.where(take, rolled_s, best_s)
#             best_v = torch.where(take, rolled_v, best_v)
#             best_k = torch.where(take, torch.as_tensor(k, device=dev, dtype=torch.long), best_k)

#         # 反标记已用源；未用源回原位  → 值守恒
#         used_src = torch.zeros_like(src_mask)
#         for k, d in enumerate(range(-self.D, self.D + 1)):
#             sel_tgt_k = (best_k == k)                          # 不再与 (best_s>0) 绑定
#             back = torch.roll(sel_tgt_k, shifts=-d, dims=0)
#             if d > 0:  back[-d:] = False
#             elif d < 0: back[:-d] = False
#             used_src |= back & src_mask

#         leftover = torch.where(~used_src, src_val, torch.zeros(1, device=dev, dtype=dtype))
#         adv_act = best_v + leftover

#         # # 调试期可开启：值守恒
#         # assert torch.allclose(x_act.sum(), adv_act.sum(), atol=1e-6)

#         out_flat = torch.zeros_like(x_flat)
#         out_flat[:, idx] = adv_act
#         B, C, H, W, N = shape_info
#         return out_flat.view(T, B, C, H, W)

#     # ---------- 4) 严格投影（tight + no-loss，value-aware，活跃线） ----------
#     @torch.no_grad()
#     def _final_projection_packets_greedy_active(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
#         """
#         严格投影（tight bound、值守恒、不重合、不丢包），仅对活跃线计算。
#         评分使用 score=v*π_k（value-aware）。
#         """
#         T = x_tbchw.shape[0]
#         x_flat, shape_info = self._flatten_tbchw(x_tbchw)      # [T,N]
#         pi_flat = pi.view(T, shape_info[-1], self.K)           # [T,N,K]
#         _, idx = self._active_index(x_flat)
#         if idx.numel() == 0:
#             return torch.zeros_like(x_tbchw)

#         x_act  = x_flat[:, idx]                                # [T,n]
#         pi_act = pi_flat[:, idx, :]                            # [T,n,K]
#         n = x_act.shape[1]

#         dev, dtype = x_act.device, x_act.dtype
#         NEG_INF  = torch.finfo(dtype).min

#         adv_act  = torch.zeros_like(x_act)
#         occupied = torch.zeros(T, n, dtype=torch.bool, device=dev)
#         reserved = (x_act > 0)                                  # 原位保留
#         unplaced = reserved.clone()

#         def tgt2src(mask_tgt, d):
#             out = torch.roll(mask_tgt, shifts=-d, dims=0)
#             if d > 0:  out[-d:] = False
#             elif d < 0: out[:(-d)] = False
#             return out

#         while unplaced.any():
#             best_score = torch.full((T, n), NEG_INF, device=dev, dtype=dtype)
#             best_k     = torch.zeros((T, n), dtype=torch.long, device=dev)

#             for k, d in enumerate(range(-self.D, self.D + 1)):
#                 # 目标可行域：不得与已占用重叠；落到别人原位必须 d==0
#                 allow_tgt = (~occupied) & ((~reserved) | (d == 0))
#                 allow_src = tgt2src(allow_tgt, d) & unplaced

#                 score_k   = torch.where(
#                     allow_src, x_act * pi_act[..., k],
#                     torch.as_tensor(NEG_INF, device=dev, dtype=dtype)
#                 )
#                 take      = score_k > best_score
#                 best_score = torch.where(take, score_k, best_score)
#                 best_k     = torch.where(take, torch.as_tensor(k, device=dev), best_k)

#             if (best_score == NEG_INF).all():
#                 # 放不进去：回原位（保证不丢包）
#                 adv_act += x_act * unplaced.float()
#                 occupied |= unplaced
#                 reserved &= (~unplaced)
#                 unplaced.zero_()
#                 break

#             # 选择全局最优源-目标对
#             mask_score = torch.where(unplaced, best_score, torch.as_tensor(NEG_INF, device=dev, dtype=dtype))
#             flat = mask_score.view(-1).argmax()
#             t_sel = int(flat // n)
#             i_sel = int(flat %  n)
#             k_sel = int(best_k[t_sel, i_sel].item())
#             d_sel = k_sel - self.D
#             t2_sel = t_sel + d_sel
#             v = x_act[t_sel, i_sel]

#             placed = False
#             if (0 <= t2_sel < T) and (not occupied[t2_sel, i_sel]) and ((not reserved[t2_sel, i_sel]) or (d_sel == 0)):
#                 adv_act[t2_sel, i_sel] = v
#                 occupied[t2_sel, i_sel] = True
#                 reserved[t_sel, i_sel] = False
#                 placed = True
#             else:
#                 # 回退策略：在可行域内找最近的空位
#                 done = False
#                 for r in range(1, self.D + 1):
#                     for t3 in (t_sel - r, t_sel + r):
#                         if 0 <= t3 < T and (not occupied[t3, i_sel]) and (not reserved[t3, i_sel]):
#                             adv_act[t3, i_sel] = v
#                             occupied[t3, i_sel] = True
#                             reserved[t_sel, i_sel] = False
#                             done = True
#                             placed = True
#                             break
#                     if done: break
#                 if not placed:
#                     adv_act[t_sel, i_sel] = v
#                     occupied[t_sel, i_sel] = True
#                     reserved[t_sel, i_sel] = False

#             unplaced[t_sel, i_sel] = False

#         out_flat = torch.zeros_like(x_flat)
#         out_flat[:, idx] = adv_act
#         B, C, H, W, N = shape_info
#         return out_flat.view(T, B, C, H, W)

#     # ---------- 5) 期望占用惩罚（活跃线版） ----------
#     def _occupancy_penalty_active(self, x_src_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
#         """
#         与原 _occupancy_penalty 等价，但只在活跃线计算，显存更小。
#         """
#         T = x_src_tbchw.shape[0]
#         K = pi.shape[-1]
#         x_flat, shape_info = self._flatten_tbchw(x_src_tbchw)  # [T,N]
#         pi_flat = pi.view(T, shape_info[-1], K)                # [T,N,K]
#         _, idx = self._active_index(x_flat)
#         if idx.numel() == 0:
#             return torch.zeros((), device=x_src_tbchw.device, dtype=x_src_tbchw.dtype)

#         x_act  = x_flat[:, idx]                                # [T,n]
#         pi_act = pi_flat[:, idx, :]                            # [T,n,K]

#         src_mask = (x_act > 0).float()
#         occ = torch.zeros_like(x_act)
#         for k, d in enumerate(range(-self.D, self.D + 1)):
#             contrib = src_mask * pi_act[..., k]                # [T,n]
#             rolled  = torch.roll(contrib, shifts=d, dims=0)
#             if d > 0:  rolled[:d] = 0
#             elif d < 0: rolled[d:] = 0
#             occ += rolled

#         # 缩放到“全量均值”的量级
#         overflow = (occ - self.cap_limit).clamp_min(0.0)
#         num_all     = x_src_tbchw.shape[0] * x_src_tbchw.shape[1] * x_src_tbchw.shape[2] * x_src_tbchw.shape[3] * x_src_tbchw.shape[4]
#         num_active  = overflow.numel()   # 实际是 T * N_active
#         scale = (num_active / num_all)
#         return (overflow * overflow).mean() * scale

#     # ---------- 6) forward ----------
#     def forward(self, x_enc_tbchw: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         x_enc_tbchw = x_enc_tbchw.to(self.device)
#         labels = labels.to(self.device)

#         T, B, C, H, W = x_enc_tbchw.shape

#         # phi（仅活跃位置反传）
#         phi = torch.zeros(T, B, C, H, W, self.K, device=self.device, dtype=x_enc_tbchw.dtype)
#         src_mask = (x_enc_tbchw != 0).unsqueeze(-1)
#         phi[..., self.D] += 1.0

#         # 原位先验（用概率→logit 偏置更稳）
#         # with torch.no_grad():
#         #     K = self.K
#         #     p0 = 0.85  # 原位先验（可调 0.7~0.9）
#         #     delta = (torch.tensor(p0, device=self.device).log()
#         #              - torch.tensor((1 - p0) / (K - 1), device=self.device).log())
#         #     phi[..., self.D] += delta
#         phi.requires_grad_(True)

#         def _mask_hook(grad):
#             return grad * src_mask.float()
#         phi.register_hook(_mask_hook)

#         if self.random_start:
#             phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)
#             # 再加一次原位偏置，避免被随机噪声稀释
#             with torch.no_grad():
#                 phi[..., self.D] += delta
#             phi.requires_grad_(True)

#         # 训练：soft（活跃线） + 宽松硬投影（value-aware） + cap（活跃线）
#         for _ in range(self.steps):
#             tau = self.temperature
#             neg_inf = torch.finfo(phi.dtype).min
#             phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
#             pi = F.softmax(phi_masked / tau, dim=-1)

#             x_soft = self._soft_shift_with_pi_active(x_enc_tbchw, pi)
#             with torch.no_grad():
#                 x_hard = self._final_projection_relaxed_packets_active(x_enc_tbchw, pi)
#             x_pil = x_hard + (x_soft - x_soft.detach())

#             logits = self.model(x_pil)
#             ce = self.loss_fn(logits, labels)
#             print (ce)
#             cap = self._occupancy_penalty_active(x_enc_tbchw, pi)
#             loss = ce - self.lambda_cap * cap

#             grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
#             with torch.no_grad():
#                 phi += self.alpha_phi * grad.sign()
#                 phi.clamp_(-10.0, 10.0)
#                 phi.requires_grad_(True)

#         # 最终：严格投影（tight + no-loss，value-aware）
#         with torch.no_grad():
#             phi_masked = torch.where(src_mask, phi, torch.as_tensor(torch.finfo(phi.dtype).min, device=phi.device, dtype=phi.dtype))
#             pi = F.softmax(phi_masked / self.temperature, dim=-1)
#             adv_tbchw = self._final_projection_packets_greedy_active(x_enc_tbchw, pi)
#         return adv_tbchw

class PGDTimeShiftAfterEncoder_L1(nn.Module):
    """
    Spike Timing Attack（after-encoder, 全时轴可移）
      - 输入/输出: [T,B,C,H,W]
      - 训练期: soft shift（全时轴） + 宽松投影（PIL, 一源一投, 禁止落到他人原位）
      - 最终: 严格投影（tight + no-loss + non-overlap），并满足 全局 L1(位移步数) 预算（整数）
      - 预算只按步数 |Δ| 计数（不看幅值）
    """
    def __init__(
        self,
        device,
        model_without_encoder: nn.Module,
        reduction: str = "mean",
        steps: int = 40,
        alpha_phi: float = 1.0,
        lambda_cap: float = 20.0,           # 若不用 cap，可置 0
        temperature: float = 1.0,
        random_start: bool = False,
        cap_limit: float = 1.0,
        # 新增：全局整数预算 + 对偶变量与其步长
        l1_steps_budget: int = 5000,
        lambda_B: float = 0.0,
        dual_lr: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.model = SNNWrapTimeMajor(model_without_encoder, reduction=reduction)
        self.steps = int(steps)
        self.alpha_phi = float(alpha_phi)
        self.lambda_cap = float(lambda_cap)
        self.temperature = float(temperature)
        self.random_start = bool(random_start)
        self.cap_limit = float(cap_limit)
        self.loss_fn = nn.CrossEntropyLoss()

        # 预算（整数步数），对偶乘子与步长
        self.l1_steps_budget = int(l1_steps_budget)
        self.lambda_B = float(lambda_B)
        self.dual_lr = float(dual_lr)

    # ---------- 工具 ----------
    @staticmethod
    def _flatten_tbchw(x):  # [T,B,C,H,W] -> (x_flat[T,N], (B,C,H,W,N))
        T, B, C, H, W = x.shape
        N = B*C*H*W
        return x.view(T, N), (B, C, H, W, N)

    @staticmethod
    def _active_index(x_flat):  # x_flat: [T,N]
        active = (x_flat != 0).any(dim=0)           # [N]
        idx = active.nonzero(as_tuple=False).squeeze(1)  # [n]
        return active, idx

    # ---------- soft shift（全时轴，活跃线） ----------
    def _soft_shift_with_pi_active(self, x_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        """
        x: [T,B,C,H,W], pi_st: [T,B,C,H,W,T]  （每个源时刻 s 的目标分布 t）
        只在活跃线计算： out[t] = Σ_s x[s] * pi[s, t]
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]

        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]

        # out_act[t,n] = sum_s x_act[s,n] * pi_act[s,n,t]
        out_act = torch.einsum('sn,snt->tn', x_act, pi_act)   # [T,n]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = out_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ---------- 训练期：宽松投影（π-only，一源一投；禁止落到他人原位） ----------
    @torch.no_grad()
    def _relaxed_projection_allT_active_strict_lite_topk(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        topk: int = 1,
        step_budget: int | None = None,  # 训练期可设为 B_total // steps；不想限就 None
    ) -> torch.Tensor:
        """
        Strict-Lite（Top-k）松弛投影：基于严格投影的思想进行加速
          - 仅使用每个源 (s,j) 的 Top-k 目标 t（k=1/2 常用）
          - 将所有候选一次性按 π 降序（同分短距优先）排序，线性扫一遍放置
          - 约束：不重合；t 为他人“仍保留”的原位时禁止落；未放置回原位；值守恒
          - 可选：step_budget（整数步数）限制本次总位移，用于训练期平滑；最终导出请用严格投影的全局整数 B
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return x_tbchw.clone()

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)                         # [T,n]
        if not has_src.any():
            return x_tbchw.clone()

        # ---- 构造每个 (s,j) 的 Top-k 候选 ----
        # top_vals/top_idx: [T,n,k]
        k = int(max(1, topk))
        top_vals, top_idx = torch.topk(pi_act, k=min(k, T), dim=-1)     # over t
        # 网格坐标：s_grid, j_grid -> [T,n,k]
        s_grid = torch.arange(T, device=dev).view(T,1,1).expand(T,n,k)
        j_grid = torch.arange(n, device=dev).view(1,n,1).expand(T,n,k)

        # 去掉 t==s（原位不作为候选；最后统一“回原位”）
        move_mask = (top_idx != s_grid)
        if not move_mask.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            return out_flat.view(T, B, C, H, W)

        # 压平成边集 E'
        s_flat  = s_grid[move_mask]     # [E']
        j_flat  = j_grid[move_mask]     # [E']
        t_flat  = top_idx[move_mask]    # [E']
        score   = top_vals[move_mask]   # [E']
        dist    = (t_flat - s_flat).abs()

        # ---- 一次性排序：π 优先，同分短距优先 ----
        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = score + eps * ((T - 1) - dist).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]
        t_flat = t_flat[order]; score  = score[order]
        dist   = dist[order]

        # ---- 线性扫描放置 ----
        adv_act = torch.zeros_like(x_act)                       # [T,n]
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)    # 目标占用（每条线）
        reserved = has_src.transpose(1,0).clone()                     # 原位保留（每条线）
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)    # 源是否已移动

        B_rem = int(1e12) if step_budget is None else int(max(0, step_budget))

        E = s_flat.numel()
        for k in range(E):
            if B_rem <= 0:
                break
            s = int(s_flat[k].item())
            j = int(j_flat[k].item())
            t = int(t_flat[k].item())
            c = int(dist[k].item())
            if moved[j, s]: continue
            if occupied[j, t]: continue
            if reserved[j, t]: continue
            if c > B_rem: continue

            adv_act[t, j]  = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s]    = True
            B_rem -= c

        # ---- 剩余未放置的源回原位 ----
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1,0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)


    # ---------- 最终：严格投影（全时轴 + 全局整数 L1步数预算，跨线共享） ----------
    @torch.no_grad()
    # def _strict_projection_allT_L1_active_global(self, x_tbchw: torch.Tensor, pi_st: torch.Tensor, l1_steps_budget: int) -> torch.Tensor:
    #     """
    #     严格投影（tight + no-loss + non-overlap），全局共享整数预算 B：
    #       1) 构造所有可移动候选 (s -> t), t!=s
    #       2) 以分数优先：按 π[s,t] 降序（同分短距离优先）一次性排序
    #       3) 依序尝试放置：要求该线目标未占用、若 t!=s 则不得落到“仍被保留”的原位、且 |t-s|<=B_rem
    #       4) 成功则放置、释放源原位、预算扣减；预算用尽或候选扫完后，剩余源回原位
    #     """
    #     T = x_tbchw.shape[0]
    #     x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
    #     pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
    #     _, idx = self._active_index(x_flat)
    #     if idx.numel() == 0 or l1_steps_budget <= 0:
    #         return x_tbchw.clone()

    #     x_act  = x_flat[:, idx]       # [T,n]
    #     pi_act = pi_flat[:, idx, :]   # [T,n,T]
    #     dev, dtype = x_act.device, x_act.dtype
    #     n = x_act.shape[1]

    #     has_src = (x_act > 0)                         # [T,n]
    #     if not has_src.any():
    #         return x_tbchw.clone()

    #     # --- 1) 构造 (s,j,t) 候选集合（跳过 t==s） ---
    #     # s_j_pairs: [m,2]，每行是 (s,j) 且该处有源
    #     s_j_pairs = torch.nonzero(has_src, as_tuple=False)           # [m, 2]
    #     m = s_j_pairs.shape[0]

    #     # 展开每个 (s,j) 的所有 t ∈ [0..T-1]
    #     t_all = torch.arange(T, device=dev, dtype=torch.long)        # [T]
    #     s_expand = s_j_pairs[:, 0].unsqueeze(1).expand(m, T)         # [m,T]
    #     j_expand = s_j_pairs[:, 1].unsqueeze(1).expand(m, T)         # [m,T]
    #     t_expand = t_all.unsqueeze(0).expand(m, T)                   # [m,T]

    #     # 去掉 t==s（原位不作为候选；留下的最后统一回原位）
    #     mask_move = (t_expand != s_expand)                           # [m,T]
    #     if not mask_move.any():
    #         # 没有任何可移动候选 → 全部原位
    #         adv_act = torch.zeros_like(x_act)
    #         adv_act[has_src] = x_act[has_src]
    #         out_flat = torch.zeros_like(x_flat)
    #         out_flat[:, idx] = adv_act
    #         B, C, H, W, N = shape_info
    #         return out_flat.view(T, B, C, H, W)

    #     # 取候选的分数与距离
    #     cand_scores = pi_act[s_expand[mask_move], j_expand[mask_move], t_expand[mask_move]]  # [E]
    #     s_flat = s_expand[mask_move]                                                         # [E]
    #     j_flat = j_expand[mask_move]                                                         # [E]
    #     t_flat = t_expand[mask_move]                                                         # [E]
    #     dist_flat = (t_flat - s_flat).abs()                                                  # [E]

    #     # --- 2) 一次性排序：π 优先，同分短距离优先 ---
    #     # 组合键：score + eps * ((T-1) - dist)  （eps足够小，不改变主排序）
    #     eps = torch.tensor(1e-6, device=dev, dtype=dtype)
    #     key = cand_scores + eps * ((T - 1) - dist_flat).to(dtype)
    #     order = torch.argsort(key, descending=True)  # [E]
    #     s_flat = s_flat[order]; j_flat = j_flat[order]; t_flat = t_flat[order]
    #     cand_scores = cand_scores[order]; dist_flat = dist_flat[order]

    #     # --- 3) 扫描候选并放置 ---
    #     adv_act = torch.zeros_like(x_act)              # [T,n]
    #     # 每条活跃线的“目标占用/原位保留/已移动源”布尔表
    #     occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)       # [n,T]
    #     reserved = has_src.transpose(1, 0).clone()                        # [n,T]
    #     moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)        # [n,T]

    #     B_rem = int(max(0, l1_steps_budget))

    #     E = s_flat.numel()
    #     for k in range(E):
    #         if B_rem <= 0:
    #             break
    #         s = int(s_flat[k].item())
    #         j = int(j_flat[k].item())
    #         t = int(t_flat[k].item())
    #         cost = int(dist_flat[k].item())

    #         if moved[j, s]:             # 该源已移动
    #             continue
    #         if occupied[j, t]:          # 目标已被占
    #             continue
    #         if reserved[j, t]:          # 仍是他人原位（不能落）
    #             continue
    #         if cost > B_rem:            # 预算不足
    #             # continue
    #             break

    #         # 放置
    #         adv_act[t, j] = x_act[s, j]
    #         occupied[j, t] = True
    #         reserved[j, s] = False      # 源离开后释放其原位
    #         moved[j, s] = True
    #         B_rem -= cost

    #     # --- 4) 剩余源回原位（不消耗预算） ---
    #     stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1,0)), as_tuple=True)
    #     if stay_s.numel() > 0:
    #         adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

    #     # 写回
    #     out_flat = torch.zeros_like(x_flat)
    #     out_flat[:, idx] = adv_act
    #     B, C, H, W, N = shape_info
    #     return out_flat.view(T, B, C, H, W)
    @torch.no_grad()
    def _strict_projection_allT_L1_active_global(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        l1_steps_budget: int,
        return_disp: bool = False,
    ):
        """
        严格投影（tight + no-loss + non-overlap），全局共享整数预算 B。
        仅在活跃线计算；与原功能等价，并可选返回每个源点的实际位移 Δ（t_target - t_src）。
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0 or l1_steps_budget <= 0:
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)                         # [T,n]
        if not has_src.any():
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        # --- 1) 构造 (s,j,t) 候选（跳过 t==s） ---
        s_j_pairs = torch.nonzero(has_src, as_tuple=False)           # [m,2], 行为 (s,j)
        m = s_j_pairs.shape[0]

        t_all    = torch.arange(T, device=dev, dtype=torch.long)     # [T]
        s_expand = s_j_pairs[:, 0].unsqueeze(1).expand(m, T)         # [m,T]
        j_expand = s_j_pairs[:, 1].unsqueeze(1).expand(m, T)         # [m,T]
        t_expand = t_all.unsqueeze(0).expand(m, T)                   # [m,T]

        mask_move = (t_expand != s_expand)                           # [m,T]
        if not mask_move.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            out_same = out_flat.view(T, B, C, H, W)
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        cand_scores = pi_act[s_expand[mask_move], j_expand[mask_move], t_expand[mask_move]]  # [E]
        s_flat = s_expand[mask_move]                                                         # [E]
        j_flat = j_expand[mask_move]                                                         # [E]
        t_flat = t_expand[mask_move]                                                         # [E]
        dist_flat = (t_flat - s_flat).abs()                                                  # [E]

        # --- 2) 排序：π 优先，同分短距优先 ---
        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = cand_scores + eps * ((T - 1) - dist_flat).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]; t_flat = t_flat[order]
        cand_scores = cand_scores[order]; dist_flat = dist_flat[order]

        # --- 3) 扫描放置（记录位移） ---
        adv_act  = torch.zeros_like(x_act)                                    # [T,n]
        disp_act = torch.zeros(T, n, device=dev, dtype=torch.int16)           # [T,n]  Δ = t - s
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)            # 目标占用
        reserved = has_src.transpose(1, 0).clone()                            # 原位保留
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)            # 源是否已移动

        B_rem = int(max(0, l1_steps_budget))

        E = s_flat.numel()
        for k in range(E):
            if B_rem <= 0:
                break
            s = int(s_flat[k].item())
            j = int(j_flat[k].item())
            t = int(t_flat[k].item())
            cost = int(dist_flat[k].item())

            if moved[j, s]:
                continue
            if occupied[j, t]:
                continue
            if reserved[j, t]:
                continue
            if cost > B_rem:
                # 保持与原实现一致：遇到大于剩余额度的候选直接停止扫描
                break

            # 放置并记录 Δ
            adv_act[t, j] = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s] = True
            disp_act[s, j] = int(t - s)          # 记录实际位移步数
            B_rem -= cost

        # --- 4) 剩余源回原位（Δ=0，不消耗预算） ---
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1, 0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]
            # disp_act 对这些位置保持 0

        # --- 5) 写回 ---
        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        adv_tbchw = out_flat.view(T, B, C, H, W)

        out_disp_flat = torch.zeros_like(x_flat, dtype=torch.int16)
        out_disp_flat[:, idx] = disp_act
        delta_tbchw = out_disp_flat.view(T, B, C, H, W)

        if return_disp:
            return adv_tbchw, delta_tbchw
        return adv_tbchw



    # ---------- 期望占用惩罚（可选；按“包数期望”，与原逻辑一致） ----------
    def _occupancy_penalty_active(self, x_src_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        """
        仅供训练稳定（若不需要可令 lambda_cap=0）：
        occ[t] = Σ_s 1_{src>0} * pi[s,t]    （只按包数，不看幅值）
        """
        T = x_src_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_src_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)                # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros((), device=x_src_tbchw.device, dtype=x_src_tbchw.dtype)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        src_mask = (x_act > 0).float()                          # [T,n]

        # occ[t,n] = sum_s src_mask[s,n]*pi[s,n,t]
        occ = torch.einsum('sn,snt->tn', src_mask, pi_act)      # [T,n]

        overflow = (occ - self.cap_limit).clamp_min(0.0)
        # 缩放回“全量平均”的量级（与你原始写法一致）
        num_all     = x_src_tbchw.numel() // T                  # B*C*H*W
        num_active  = overflow.numel()
        scale = (num_active / (T * num_all))
        return (overflow * overflow).mean() * scale

    # ---------- forward ----------
    def forward(self, x_enc_tbchw: torch.Tensor, labels: torch.Tensor, return_disp: bool = False, use_PIL: bool = True, use_cap: bool = True, use_penalty: bool = True, target_label: int = -1):
        x_enc_tbchw = x_enc_tbchw.to(self.device)
        labels = labels.to(self.device)

        T, Bsz, C, H, W = x_enc_tbchw.shape

        # φ: [T,B,C,H,W,T] —— 每个源时刻 s 的目标时刻 t 的 logit
        phi = torch.zeros(T, Bsz, C, H, W, T, device=self.device, dtype=x_enc_tbchw.dtype)

        src_mask = (x_enc_tbchw != 0).unsqueeze(-1)  # [T,B,C,H,W,1]

        def _mask_hook(grad):
            return grad * src_mask.float()

        phi.requires_grad_(True)
        phi.register_hook(_mask_hook)

        if self.random_start:
            phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)

        # === 预计算步数代价矩阵 C_{s,t} = |t - s|（只看位移步数，不看幅值） ===
        # 形状: [T,1,1,1,1,T]，可与 pi_st / src_mask 广播
        ar_s = torch.arange(T, device=self.device, dtype=x_enc_tbchw.dtype).view(T, 1, 1, 1, 1, 1)
        ar_t = torch.arange(T, device=self.device, dtype=x_enc_tbchw.dtype).view(1, 1, 1, 1, 1, T)
        Cst = (ar_t - ar_s).abs()  # [T,1,1,1,1,T]

        # 训练期：soft + 宽松投影（全时轴），cap(可选) + 预算超限才惩罚；λ_B 做对偶更新
        for t in range(self.steps):
            tau = self.temperature
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi_st = F.softmax(phi_masked / tau, dim=-1)   # [T,B,C,H,W,T]

            x_soft = self._soft_shift_with_pi_active(x_enc_tbchw, pi_st)
            if use_PIL:
                with torch.no_grad():
                    # x_hard = self._relaxed_projection_allT_active_strict_lite_topk(x_enc_tbchw, pi_st, 2, self.l1_steps_budget)
                    x_hard = self._strict_projection_allT_L1_active_global(x_enc_tbchw, pi_st, self.l1_steps_budget)
                x_pil = x_hard + (x_soft - x_soft.detach())
            else:
                x_pil = x_soft
            logits = self.model(x_pil)

            # === 软的总位移步数 S_soft：逐样本统计，再取平均 ===
            S_soft_b = ((pi_st * Cst) * src_mask).sum(dim=(0, 2, 3, 4, 5))  # [Bsz]
            over_b = F.relu(S_soft_b - self.l1_steps_budget)                 # [Bsz]
            # penalty_budget = self.lambda_B * over_b.mean()                   # 标量
            if use_penalty:
                penalty_budget = 10 * over_b.mean()/self.l1_steps_budget
            else:
                penalty_budget = 0

            # cap（保留；不用就把 lambda_cap 设 0）
            if use_cap:
                cap = self._occupancy_penalty_active(x_enc_tbchw, pi_st) if self.lambda_cap != 0 else 0
            else:
                cap = 0

            if target_label < 0:
                ce = self.loss_fn(logits, labels)
                # 注意：我们在做“上升”，所以惩罚项要减
                loss = ce - self.lambda_cap * cap - penalty_budget
            else:
                ce = self.loss_fn(logits, target_label * torch.ones_like(labels))
                loss = -ce - self.lambda_cap * cap - penalty_budget

            # print (ce)
            # print (S_soft_b)
            # print (cap)

            # sign-PGD 更新 φ
            grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                phi += self.alpha_phi * grad.sign()
                phi.clamp_(-10.0, 10.0)
                phi.requires_grad_(True)

                # === 对偶上升更新 λ_B（逐步逼近硬预算 B），λ_B ≥ 0 ===
                gap = (S_soft_b.mean() - self.l1_steps_budget).item()
                self.lambda_B = max(0.0, float(self.lambda_B + self.dual_lr * gap))

        # 最终：严格投影（全时轴 + 全局整数 L1(步数) 预算，跨线共享）
        with torch.no_grad():
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi_st = F.softmax(phi_masked / self.temperature, dim=-1)
            if return_disp:
                adv_tbchw, delta_tbchw = self._strict_projection_allT_L1_active_global(x_enc_tbchw, pi_st, self.l1_steps_budget, return_disp=return_disp)
                return adv_tbchw, delta_tbchw 
            else:
                adv_tbchw = self._strict_projection_allT_L1_active_global(x_enc_tbchw, pi_st, self.l1_steps_budget)
                return adv_tbchw


class PGDTimeShiftAfterEncoder_L0(nn.Module):
    """
    Spike Timing Attack（after-encoder, 全时轴可移；L0 预算 = 移动的“点数”）
      - 输入/输出: [T,B,C,H,W]
      - 训练期: soft shift（全时轴） + （保持你当前习惯）用严格投影的 x_hard 做 PIL
      - 最终: 严格投影（tight + no-loss + non-overlap），并满足 全局 L0(移动个数) 预算（整数）
      - 预算只按“移动个数”计数（t != s 则成本=1；不看距离）
    """
    def __init__(
        self,
        device,
        model_without_encoder: nn.Module,
        reduction: str = "mean",
        steps: int = 40,
        alpha_phi: float = 1.0,
        lambda_cap: float = 20.0,           # 若不用 cap，可置 0
        temperature: float = 1.0,
        random_start: bool = False,
        cap_limit: float = 1.0,
        # L0 预算（整数移动个数）+ 对偶变量与其步长（命名保持一致，方便替换原代码）
        l0_moves_budget: int = 1000,
        lambda_B: float = 0.0,
        dual_lr: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.model = SNNWrapTimeMajor(model_without_encoder, reduction=reduction)
        self.steps = int(steps)
        self.alpha_phi = float(alpha_phi)
        self.lambda_cap = float(lambda_cap)
        self.temperature = float(temperature)
        self.random_start = bool(random_start)
        self.cap_limit = float(cap_limit)
        self.loss_fn = nn.CrossEntropyLoss()

        # 预算（整数“移动个数”），对偶乘子与步长
        self.l0_moves_budget = int(l0_moves_budget)
        self.lambda_B = float(lambda_B)
        self.dual_lr = float(dual_lr)

    # ---------- 工具 ----------
    @staticmethod
    def _flatten_tbchw(x):  # [T,B,C,H,W] -> (x_flat[T,N], (B,C,H,W,N))
        T, B, C, H, W = x.shape
        N = B*C*H*W
        return x.view(T, N), (B, C, H, W, N)

    @staticmethod
    def _active_index(x_flat):  # x_flat: [T,N]
        active = (x_flat != 0).any(dim=0)           # [N]
        idx = active.nonzero(as_tuple=False).squeeze(1)  # [n]
        return active, idx

    # ---------- soft shift（全时轴，活跃线） ----------
    def _soft_shift_with_pi_active(self, x_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        """
        x: [T,B,C,H,W], pi_st: [T,B,C,H,W,T]  （每个源时刻 s 的目标分布 t）
        只在活跃线计算： out[t] = Σ_s x[s] * pi[s, t]
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]

        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]

        # out_act[t,n] = sum_s x_act[s,n] * pi_act[s,n,t]
        out_act = torch.einsum('sn,snt->tn', x_act, pi_act)   # [T,n]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = out_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ---------- 训练期：Strict-Lite（Top-k）松弛投影（这里按“移动个数”计预算） ----------
    @torch.no_grad()
    def _relaxed_projection_allT_active_strict_lite_topk(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        topk: int = 1,
        step_budget: int | None = None,  # 训练期可设为 moves_B // steps；不想限就 None
    ) -> torch.Tensor:
        """
        Strict-Lite（Top-k）松弛投影（L0 版本）：
          - 候选：每个 (s,j) 取 Top-k 的 t（去掉 t==s）
          - 全局按 π 降序（同分短距优先做次序而已，预算仍是“计数”）
          - 线性扫：不重合、原位保留、未放置回原位
          - 预算：每成功放置一个，B_rem -= 1
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return x_tbchw.clone()

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)                         # [T,n]
        if not has_src.any():
            return x_tbchw.clone()

        # ---- 每个 (s,j) 的 Top-k 候选 ----
        k = int(max(1, topk))
        top_vals, top_idx = torch.topk(pi_act, k=min(k, T), dim=-1)     # over t
        s_grid = torch.arange(T, device=dev).view(T,1,1).expand(T,n,k)
        j_grid = torch.arange(n, device=dev).view(1,n,1).expand(T,n,k)

        # 去掉 t==s
        move_mask = (top_idx != s_grid)
        if not move_mask.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            return out_flat.view(T, B, C, H, W)

        # 压平成边集
        s_flat  = s_grid[move_mask]
        j_flat  = j_grid[move_mask]
        t_flat  = top_idx[move_mask]
        score   = top_vals[move_mask]
        dist    = (t_flat - s_flat).abs()

        # 排序键：π 优先，同分短距优先（仅作 tie-breaker）
        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = score + eps * ((T - 1) - dist).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]
        t_flat = t_flat[order]; score  = score[order]
        # dist   = dist[order]  # L0 不用成本，保留与否均可

        # 放置
        adv_act = torch.zeros_like(x_act)                       # [T,n]
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)
        reserved = has_src.transpose(1,0).clone()
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)

        B_rem = int(1e12) if step_budget is None else int(max(0, step_budget))

        E = s_flat.numel()
        for k_ in range(E):
            if B_rem <= 0:
                break
            s = int(s_flat[k_].item())
            j = int(j_flat[k_].item())
            t = int(t_flat[k_].item())
            if moved[j, s]: continue
            if occupied[j, t]: continue
            if reserved[j, t]: continue

            adv_act[t, j]  = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s]    = True
            B_rem -= 1  # L0：每移动一个，预算减 1

        # 未放置的回原位
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1,0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    @torch.no_grad()
    def _strict_projection_allT_L0_active_global(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        l0_moves_budget: int,
        return_disp: bool = False,
    ):
        """
        严格投影（tight + no-loss + non-overlap），全局共享整数 L0 预算（移动“点数”）。
        功能与原版一致；可选返回每个源点的实际位移 Δ（t_target - t_src）。
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0 or l0_moves_budget <= 0:
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)         # [T,n]
        if not has_src.any():
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        # --- 构造候选集（去掉 t==s） ---
        s_j_pairs = torch.nonzero(has_src, as_tuple=False)        # [m,2] (s,j)
        m = s_j_pairs.shape[0]
        t_all    = torch.arange(T, device=dev, dtype=torch.long)  # [T]
        s_expand = s_j_pairs[:, 0].unsqueeze(1).expand(m, T)      # [m,T]
        j_expand = s_j_pairs[:, 1].unsqueeze(1).expand(m, T)      # [m,T]
        t_expand = t_all.unsqueeze(0).expand(m, T)                # [m,T]

        mask_move = (t_expand != s_expand)                        # [m,T]
        if not mask_move.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            out_same = out_flat.view(T, B, C, H, W)
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        # 得分与距离（距离仅作同分次序的 tie-breaker）
        cand_scores = pi_act[s_expand[mask_move], j_expand[mask_move], t_expand[mask_move]]  # [E]
        s_flat = s_expand[mask_move]                                                         # [E]
        j_flat = j_expand[mask_move]                                                         # [E]
        t_flat = t_expand[mask_move]                                                         # [E]
        dist_flat = (t_flat - s_flat).abs()                                                  # [E]

        # --- 排序：π 优先，同分短距优先 ---
        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = cand_scores + eps * ((T - 1) - dist_flat).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]; t_flat = t_flat[order]
        # cand_scores = cand_scores[order]  # 如需调试可保留
        # dist_flat   = dist_flat[order]

        # --- 放置（记录位移） ---
        adv_act  = torch.zeros_like(x_act)                              # [T,n]
        disp_act = torch.zeros(T, n, device=dev, dtype=torch.int16)     # [T,n] Δ = t - s
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)      # 目标占用
        reserved = has_src.transpose(1, 0).clone()                      # 原位保留
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)      # 源是否已移动

        B_rem = int(max(0, l0_moves_budget))

        E = s_flat.numel()
        for k_ in range(E):
            if B_rem <= 0:
                break
            s = int(s_flat[k_].item())
            j = int(j_flat[k_].item())
            t = int(t_flat[k_].item())

            if moved[j, s]:
                continue
            if occupied[j, t]:
                continue
            if reserved[j, t]:
                continue

            adv_act[t, j]  = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s]    = True
            disp_act[s, j] = int(t - s)   # 记录实际位移步数 Δ
            B_rem -= 1                     # L0：每移动一个，预算减 1

        # 未移动者回原位（Δ=0）
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1, 0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

        # --- 写回原形状 ---
        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        adv_tbchw = out_flat.view(T, B, C, H, W)

        out_disp_flat = torch.zeros_like(x_flat, dtype=torch.int16)
        out_disp_flat[:, idx] = disp_act
        delta_tbchw = out_disp_flat.view(T, B, C, H, W)

        if return_disp:
            return adv_tbchw, delta_tbchw
        return adv_tbchw


    # ---------- 期望占用惩罚（与你原逻辑一致，可选） ----------
    def _occupancy_penalty_active(self, x_src_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        T = x_src_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_src_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)                # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros((), device=x_src_tbchw.device, dtype=x_src_tbchw.dtype)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        src_mask = (x_act > 0).float()

        # occ[t,n] = sum_s src_mask[s,n]*pi[s,n,t]
        occ = torch.einsum('sn,snt->tn', src_mask, pi_act)      # [T,n]

        overflow = (occ - self.cap_limit).clamp_min(0.0)
        num_all     = x_src_tbchw.numel() // T
        num_active  = overflow.numel()
        scale = (num_active / (T * num_all))
        return (overflow * overflow).mean() * scale

    # ---------- forward ----------
    def forward(self, x_enc_tbchw: torch.Tensor, labels: torch.Tensor, return_disp: bool = False, use_PIL: bool = True, use_cap: bool = True, use_penalty: bool = True, target_label: int = -1):
        x_enc_tbchw = x_enc_tbchw.to(self.device)
        labels = labels.to(self.device)

        T, Bsz, C, H, W = x_enc_tbchw.shape

        # φ: [T,B,C,H,W,T]
        phi = torch.zeros(T, Bsz, C, H, W, T, device=self.device, dtype=x_enc_tbchw.dtype)
        src_mask = (x_enc_tbchw != 0).unsqueeze(-1)  # [T,B,C,H,W,1]

        def _mask_hook(grad):
            return grad * src_mask.float()
        phi.requires_grad_(True)
        phi.register_hook(_mask_hook)

        if self.random_start:
            phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)
            # （按你原版保持一致：不重复挂 hook）

        # 训练循环
        for _ in range(self.steps):
            tau = self.temperature
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi_st = F.softmax(phi_masked / tau, dim=-1)   # [T,B,C,H,W,T]

            x_soft = self._soft_shift_with_pi_active(x_enc_tbchw, pi_st)
            if use_PIL:
                with torch.no_grad():
                    # 保持你当前习惯：训练期也用严格投影作为 x_hard（L0 版本）
                    x_hard = self._strict_projection_allT_L0_active_global(x_enc_tbchw, pi_st, self.l0_moves_budget)
                    # 若想更快：可改成 strict-lite-topk，并把 step_budget 设为 self.l0_moves_budget//self.steps
                x_pil = x_hard + (x_soft - x_soft.detach())
            else:
                x_pil = x_soft

            logits = self.model(x_pil)

            # cap（保留；不用就把 lambda_cap 设 0）
            if use_cap:
                cap = self._occupancy_penalty_active(x_enc_tbchw, pi_st) if self.lambda_cap != 0 else 0
            else:
                cap = 0

            # === L0 的 soft 预算：期望移动“个数” ===
            # pi_diag: 取 t==s 的对角，形状转成 [T,B,C,H,W]
            pi_diag = pi_st.diagonal(dim1=0, dim2=-1).permute(4,0,1,2,3)  # [T,B,C,H,W]
            move_prob = 1.0 - pi_diag                                     # [T,B,C,H,W]
            S_soft_b = (move_prob * src_mask.squeeze(-1).float()).sum(dim=(0, 2, 3, 4))  # [B]

            over_b = F.relu(S_soft_b - self.l0_moves_budget)              # [B]
            # 为保持你当前代码风格：用一个固定系数（不直接用 lambda_B）
            if use_penalty:
                penalty_budget = 5 * over_b.mean() / max(1, self.l0_moves_budget)
            else:
                penalty_budget = 0

            if target_label < 0:
                ce = self.loss_fn(logits, labels)
                # 注意：我们在做“上升”，所以惩罚项要减
                loss = ce - self.lambda_cap * cap - penalty_budget
            else:
                ce = self.loss_fn(logits, target_label * torch.ones_like(labels))
                loss = -ce - self.lambda_cap * cap - penalty_budget
            # print (ce)
            # print (S_soft_b)

            # sign-PGD 更新 φ（保持与你现有一致）
            grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                phi += self.alpha_phi * grad.sign()
                phi.clamp_(-10.0, 10.0)
                phi.requires_grad_(True)

                # 对偶上升（保留你现有写法，尽管上面没直接用 lambda_B）
                gap = (S_soft_b.mean() - self.l0_moves_budget).item()
                self.lambda_B = max(0.0, float(self.lambda_B + self.dual_lr * gap))

        # 最终：严格投影（全候选 + 全局整数 L0 预算）
        with torch.no_grad():
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi_st = F.softmax(phi_masked / self.temperature, dim=-1)
            if return_disp:
                adv_tbchw, delta_tbchw = self._strict_projection_allT_L0_active_global(x_enc_tbchw, pi_st, self.l0_moves_budget, return_disp=return_disp)
                return adv_tbchw, delta_tbchw 
            else:
                adv_tbchw = self._strict_projection_allT_L0_active_global(x_enc_tbchw, pi_st, self.l0_moves_budget)
                return adv_tbchw





class PGDTimeShiftAfterEncoder_Lowgpu_Multi(nn.Module):
    """
    Multi-model Spike Timing Attack（仅 after-encoder）：
      输入/输出: [T,B,C,H,W]
      在 Δ∈{-D...0...D} 的 logits 上做 PGD
      模型为多个 SNN: net = [net1, ..., netM]
    """
    def __init__(
        self,
        device,
        models_without_encoder,          # list like [net1, net2, ...]
        reduction: str = "mean",
        D: int = 1,
        steps: int = 20,
        alpha_phi: float = 1.0,
        lambda_cap: float = 20.0,
        temperature: float = 1.0,
        random_start: bool = False,
        cap_limit: float = 1.0,
    ):
        super().__init__()
        self.device = device

        # —— 多模型包装成 ModuleList —— #
        # 每个网络都用 SNNWrapTimeMajor 包一层
        self.models = nn.ModuleList(
            [SNNWrapTimeMajor(m, reduction=reduction) for m in models_without_encoder]
        )
        self.M = len(self.models)

        self.D = int(D)
        self.K = 2 * self.D + 1
        self.steps = int(steps)
        self.alpha_phi = float(alpha_phi)
        self.lambda_cap = float(lambda_cap)
        self.temperature = float(temperature)
        self.random_start = bool(random_start)
        self.cap_limit = float(cap_limit)
        self.loss_fn = nn.CrossEntropyLoss()
        self.alpha_lr = 0.05  # 比较小的步长，自己调

    # ========== 1) 活跃线工具 ==========
    @staticmethod
    def _flatten_tbchw(x):  # [T,B,C,H,W] -> (x_flat[T,N], B,C,H,W,N)
        T, B, C, H, W = x.shape
        N = B * C * H * W
        return x.view(T, N), (B, C, H, W, N)

    @staticmethod
    def _active_index(x_flat):  # x_flat: [T,N]
        active = (x_flat != 0).any(dim=0)  # [N] bool
        idx = active.nonzero(as_tuple=False).squeeze(1)  # [N_active]
        return active, idx

    # ========== 2) soft shift（活跃线版） ==========
    def _soft_shift_with_pi_active(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        T = x_tbchw.shape[0]
        K = pi.shape[-1]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)   # [T,N]
        pi_flat = pi.view(T, shape_info[-1], K)             # [T,N,K]

        active_mask, idx = self._active_index(x_flat)       # [N], [N_active]
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]                             # [T,n]
        pi_act = pi_flat[:, idx, :]                         # [T,n,K]

        out_act = torch.zeros_like(x_act)                   # [T,n]
        for k, d in enumerate(range(-self.D, self.D + 1)):
            mass = pi_act[..., k] * x_act                   # [T,n]
            rolled = torch.roll(mass, shifts=d, dims=0)
            if d > 0:  rolled[:d] = 0
            elif d < 0: rolled[d:] = 0
            out_act += rolled

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = out_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ========== 3) 宽松投影（活跃线 + 流式逐Δ，省内存） ==========
    @torch.no_grad()
    def _final_projection_relaxed_packets_active(self, x_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        T = x_tbchw.shape[0]
        K = pi.shape[-1]
        dev, dtype = x_tbchw.device, x_tbchw.dtype

        x_flat, shape_info = self._flatten_tbchw(x_tbchw)      # [T,N]
        pi_flat = pi.view(T, shape_info[-1], K)                # [T,N,K]
        active_mask, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]                                # [T,n]
        pi_act = pi_flat[:, idx, :]                            # [T,n,K]

        src_mask = (x_act > 0)                                 # [T,n] bool
        src_val  = x_act                                       # [T,n]
        pi_max, idx_k = pi_act.max(dim=-1)                     # [T,n]

        best_s = torch.full_like(x_act, torch.finfo(dtype).min)
        best_v = torch.zeros_like(x_act)
        best_k = torch.zeros_like(idx_k, dtype=torch.long)
        target_reserved = src_mask

        for k, d in enumerate(range(-self.D, self.D + 1)):
            choose  = (idx_k == k) & src_mask
            score_s = torch.where(choose, pi_max, torch.zeros(1, device=dev, dtype=dtype))
            value_s = torch.where(choose, src_val, torch.zeros(1, device=dev, dtype=dtype))

            rolled_s = torch.roll(score_s, shifts=d, dims=0)
            rolled_v = torch.roll(value_s, shifts=d, dims=0)
            if d > 0:  rolled_s[:d] = 0; rolled_v[:d] = 0
            elif d < 0: rolled_s[d:] = 0; rolled_v[d:] = 0

            if d != 0:
                allow = (~target_reserved)
                rolled_s = rolled_s * allow
                rolled_v = rolled_v * allow

            take   = rolled_s > best_s
            best_s = torch.where(take, rolled_s, best_s)
            best_v = torch.where(take, rolled_v, best_v)
            best_k = torch.where(take, torch.as_tensor(k, device=dev, dtype=torch.long), best_k)

        adv_moved = best_v

        used_src = torch.zeros_like(src_mask)
        for k, d in enumerate(range(-self.D, self.D + 1)):
            sel_tgt_k = (best_k == k) & (best_s > 0)
            back = torch.roll(sel_tgt_k, shifts=-d, dims=0)
            if d > 0:  back[-d:] = False
            elif d < 0: back[:-d] = False
            used_src |= back & (idx_k == k) & src_mask

        leftover = torch.where(~used_src, src_val, torch.zeros(1, device=dev, dtype=dtype))
        adv_act = adv_moved + leftover                          # [T,n]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    @torch.no_grad()
    def _final_projection_packets_greedy_active(
        self,
        x_tbchw: torch.Tensor,
        pi: torch.Tensor,
        return_disp: bool = False,
    ):
        T = x_tbchw.shape[0]
        K = pi.shape[-1]
        dev, dtype = x_tbchw.device, x_tbchw.dtype

        x_flat, shape_info = self._flatten_tbchw(x_tbchw)      # [T,N]
        pi_flat = pi.view(T, shape_info[-1], K)                # [T,N,K]
        active_mask, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            out_zero = torch.zeros_like(x_tbchw)
            if return_disp:
                return out_zero, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_zero

        x_act  = x_flat[:, idx]                                # [T,n]
        pi_act = pi_flat[:, idx, :]
        n = x_act.shape[1]

        adv_act  = torch.zeros_like(x_act)                     # [T,n]
        occupied = torch.zeros(T, n, dtype=torch.bool, device=dev)
        reserved = (x_act > 0)
        unplaced = reserved.clone()

        disp_act = torch.zeros(T, n, device=dev, dtype=torch.int16)

        NEG_INF  = torch.finfo(dtype).min

        def tgt2src(mask_tgt, d):
            out = torch.roll(mask_tgt, shifts=-d, dims=0)
            if d > 0:
                out[-d:] = False
            elif d < 0:
                out[:(-d)] = False
            return out

        while unplaced.any():
            best_score = torch.full((T, n), NEG_INF, device=dev, dtype=dtype)
            best_k     = torch.zeros((T, n), dtype=torch.long, device=dev)

            for k, d in enumerate(range(-self.D, self.D + 1)):
                allow_tgt = (~occupied) & ((~reserved) | (d == 0))
                allow_src = tgt2src(allow_tgt, d) & unplaced
                score_k   = torch.where(allow_src, pi_act[..., k], torch.as_tensor(NEG_INF, device=dev, dtype=dtype))
                take      = score_k > best_score
                best_score = torch.where(take, score_k, best_score)
                best_k     = torch.where(take, torch.as_tensor(k, device=dev), best_k)

            if (best_score == NEG_INF).all():
                adv_act += x_act * unplaced.float()
                occupied |= unplaced
                reserved &= (~unplaced)
                unplaced.zero_()
                break

            mask_score = torch.where(unplaced, best_score, torch.as_tensor(NEG_INF, device=dev, dtype=dtype))
            flat = mask_score.view(-1).argmax()
            t_sel = int(flat // n)
            i_sel = int(flat %  n)
            k_sel = int(best_k[t_sel, i_sel].item())
            d_sel = k_sel - self.D
            t2_sel = t_sel + d_sel
            v = x_act[t_sel, i_sel]

            placed = False
            if (0 <= t2_sel < T) and (not occupied[t2_sel, i_sel]) and ((not reserved[t2_sel, i_sel]) or (d_sel == 0)):
                adv_act[t2_sel, i_sel] = v
                occupied[t2_sel, i_sel] = True
                reserved[t_sel, i_sel] = False
                disp_act[t_sel, i_sel] = int(t2_sel - t_sel)
                placed = True
            else:
                done = False
                for r in range(1, self.D + 1):
                    for t3 in (t_sel - r, t_sel + r):
                        if 0 <= t3 < T and (not occupied[t3, i_sel]) and (not reserved[t3, i_sel]):
                            adv_act[t3, i_sel] = v
                            occupied[t3, i_sel] = True
                            reserved[t_sel, i_sel] = False
                            disp_act[t_sel, i_sel] = int(t3 - t_sel)
                            done = True
                            placed = True
                            break
                    if done:
                        break
                if not placed:
                    adv_act[t_sel, i_sel] = v
                    occupied[t_sel, i_sel] = True
                    reserved[t_sel, i_sel] = False
                    disp_act[t_sel, i_sel] = 0

            unplaced[t_sel, i_sel] = False

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        adv_tbchw = out_flat.view(T, B, C, H, W)

        out_disp_flat = torch.zeros_like(x_flat, dtype=torch.int16)
        out_disp_flat[:, idx] = disp_act
        disp_tbchw = out_disp_flat.view(T, B, C, H, W)

        if return_disp:
            return adv_tbchw, disp_tbchw
        return adv_tbchw

    # ========== 5) 期望占用惩罚（活跃线版） ==========
    def _occupancy_penalty_active(self, x_src_tbchw: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        T = x_src_tbchw.shape[0]
        K = pi.shape[-1]
        x_flat, shape_info = self._flatten_tbchw(x_src_tbchw)  # [T,N]
        pi_flat = pi.view(T, shape_info[-1], K)                # [T,N,K]
        active_mask, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros((), device=x_src_tbchw.device, dtype=x_src_tbchw.dtype)

        x_act  = x_flat[:, idx]                                # [T,n]
        pi_act = pi_flat[:, idx, :]                            # [T,n,K]

        src_mask = (x_act > 0).float()
        occ = torch.zeros_like(x_act)
        for k, d in enumerate(range(-self.D, self.D + 1)):
            contrib = src_mask * pi_act[..., k]                # [T,n]
            rolled  = torch.roll(contrib, shifts=d, dims=0)
            if d > 0:  rolled[:d] = 0
            elif d < 0: rolled[d:] = 0
            occ += rolled

        overflow = (occ - self.cap_limit).clamp_min(0.0)
        num_all     = x_src_tbchw.numel()
        num_active  = overflow.numel()
        scale = (num_active / num_all)
        return (overflow * overflow).mean() * scale

    # ========== 6) forward：多模型版 ==========
    def forward(
        self,
        x_enc_tbchw: torch.Tensor,
        labels: torch.Tensor,
        return_disp: bool = False,
        use_PIL: bool = True,
        use_cap: bool = True,
        use_penalty: bool = True,
        target_label: int = -1,
    ):
        x_enc_tbchw = x_enc_tbchw.to(self.device)
        labels = labels.to(self.device)

        T, B, C, H, W = x_enc_tbchw.shape

        phi = torch.zeros(T, B, C, H, W, self.K, device=self.device, dtype=x_enc_tbchw.dtype)
        phi[..., self.D - 1 : self.D + 2] = 4
        phi.requires_grad_(True)

        src_mask = (x_enc_tbchw != 0).unsqueeze(-1)
        def _mask_hook(grad):
            return grad * src_mask.float()
        phi.register_hook(_mask_hook)

        if self.random_start:
            phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)

        alpha = torch.ones(self.M, device=self.device) / self.M
        optimization_steps = self.steps if target_label < 0 else 2 * self.steps
        for t in range(optimization_steps):
            tau = self.temperature
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(
                src_mask,
                phi,
                torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype),
            )
            pi = F.softmax(phi_masked / tau, dim=-1)

            if use_cap:
                cap = self._occupancy_penalty_active(x_enc_tbchw, pi)
            else:
                cap = 0

            x_soft = self._soft_shift_with_pi_active(x_enc_tbchw, pi)
            if use_PIL:
                with torch.no_grad():
                    x_hard = self._final_projection_relaxed_packets_active(x_enc_tbchw, pi)
                x_pil = x_hard + (x_soft - x_soft.detach())
            else:
                x_pil = x_soft

            # ======= 关键改动：多模型融合 ======= #
            ce_list = []
            for m in self.models:
                logits_m = m(x_pil)
                if target_label < 0:
                    ce_m = self.loss_fn(logits_m, labels)
                else:
                    ce_m = self.loss_fn(logits_m, target_label * torch.ones_like(labels))
                ce_list.append(ce_m)

            ce_stack = torch.stack(ce_list)      # [M]

            # 用当前 alpha 做混合损失
            alpha_sum = alpha.sum().detach()
            F_mix = (alpha * ce_stack).sum() / alpha_sum

            if target_label < 0:
                loss = F_mix - self.lambda_cap * cap
            else:
                loss = -F_mix - self.lambda_cap * cap

            # 对 phi 求梯度（跟之前一样）
            grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                phi += self.alpha_phi * grad.sign()
                phi.clamp_(-10.0, 10.0)
                phi.requires_grad_(True)

            # ===== 这里新增：更新 alpha =====
            with torch.no_grad():
                if target_label < 0:
                    alpha = alpha - self.alpha_lr * ce_stack.detach()
                else:
                    alpha = alpha + self.alpha_lr * ce_stack.detach()

                # 限制为非负并归一化
                alpha.clamp_(min=0.0)
                alpha /= (alpha.mean() + 1e-8)

            print(F_mix.detach())

        final_projection = (
            self._final_projection_packets_greedy_active
            if target_label < 0
            else self._final_projection_relaxed_packets_active
        )
        with torch.no_grad():
            phi_masked = torch.where(
                src_mask,
                phi,
                torch.as_tensor(torch.finfo(phi.dtype).min, device=phi.device, dtype=phi.dtype),
            )
            pi = F.softmax(phi_masked / self.temperature, dim=-1)
            if return_disp:
                adv_tbchw, delta_tbchw = final_projection(x_enc_tbchw, pi, return_disp=return_disp)
                return adv_tbchw, delta_tbchw
            else:
                adv_tbchw = final_projection(x_enc_tbchw, pi)
                return adv_tbchw



class PGDTimeShiftAfterEncoder_L1_Multi(nn.Module):
    """
    Spike Timing Attack（after-encoder, 全时轴可移）
      - 输入/输出: [T,B,C,H,W]
      - 训练期: soft shift（全时轴） + 宽松投影（PIL, 一源一投, 禁止落到他人原位）
      - 最终: 严格投影（tight + no-loss + non-overlap），并满足 全局 L1(位移步数) 预算（整数）
      - 预算只按步数 |Δ| 计数（不看幅值）
    """
    def __init__(
        self,
        device,
        models_without_encoder: nn.Module,
        reduction: str = "mean",
        steps: int = 40,
        alpha_phi: float = 1.0,
        lambda_cap: float = 20.0,           # 若不用 cap，可置 0
        temperature: float = 1.0,
        random_start: bool = False,
        cap_limit: float = 1.0,
        # 新增：全局整数预算 + 对偶变量与其步长
        l1_steps_budget: int = 5000,
        lambda_B: float = 0.0,
        dual_lr: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList(
            [SNNWrapTimeMajor(m, reduction=reduction) for m in models_without_encoder]
        )
        self.M = len(self.models)
        self.steps = int(steps)
        self.alpha_phi = float(alpha_phi)
        self.lambda_cap = float(lambda_cap)
        self.temperature = float(temperature)
        self.random_start = bool(random_start)
        self.cap_limit = float(cap_limit)
        self.loss_fn = nn.CrossEntropyLoss()

        # 预算（整数步数），对偶乘子与步长
        self.l1_steps_budget = int(l1_steps_budget)
        self.lambda_B = float(lambda_B)
        self.dual_lr = float(dual_lr)
        self.alpha_lr = 0.05  # 比较小的步长，自己调

    # ---------- 工具 ----------
    @staticmethod
    def _flatten_tbchw(x):  # [T,B,C,H,W] -> (x_flat[T,N], (B,C,H,W,N))
        T, B, C, H, W = x.shape
        N = B*C*H*W
        return x.view(T, N), (B, C, H, W, N)

    @staticmethod
    def _active_index(x_flat):  # x_flat: [T,N]
        active = (x_flat != 0).any(dim=0)           # [N]
        idx = active.nonzero(as_tuple=False).squeeze(1)  # [n]
        return active, idx

    # ---------- soft shift（全时轴，活跃线） ----------
    def _soft_shift_with_pi_active(self, x_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        """
        x: [T,B,C,H,W], pi_st: [T,B,C,H,W,T]  （每个源时刻 s 的目标分布 t）
        只在活跃线计算： out[t] = Σ_s x[s] * pi[s, t]
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]

        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]

        # out_act[t,n] = sum_s x_act[s,n] * pi_act[s,n,t]
        out_act = torch.einsum('sn,snt->tn', x_act, pi_act)   # [T,n]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = out_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ---------- 训练期：宽松投影（π-only，一源一投；禁止落到他人原位） ----------
    @torch.no_grad()
    def _relaxed_projection_allT_active_strict_lite_topk(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        topk: int = 1,
        step_budget: int | None = None,  # 训练期可设为 B_total // steps；不想限就 None
    ) -> torch.Tensor:
        """
        Strict-Lite（Top-k）松弛投影：基于严格投影的思想进行加速
          - 仅使用每个源 (s,j) 的 Top-k 目标 t（k=1/2 常用）
          - 将所有候选一次性按 π 降序（同分短距优先）排序，线性扫一遍放置
          - 约束：不重合；t 为他人“仍保留”的原位时禁止落；未放置回原位；值守恒
          - 可选：step_budget（整数步数）限制本次总位移，用于训练期平滑；最终导出请用严格投影的全局整数 B
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return x_tbchw.clone()

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)                         # [T,n]
        if not has_src.any():
            return x_tbchw.clone()

        # ---- 构造每个 (s,j) 的 Top-k 候选 ----
        # top_vals/top_idx: [T,n,k]
        k = int(max(1, topk))
        top_vals, top_idx = torch.topk(pi_act, k=min(k, T), dim=-1)     # over t
        # 网格坐标：s_grid, j_grid -> [T,n,k]
        s_grid = torch.arange(T, device=dev).view(T,1,1).expand(T,n,k)
        j_grid = torch.arange(n, device=dev).view(1,n,1).expand(T,n,k)

        # 去掉 t==s（原位不作为候选；最后统一“回原位”）
        move_mask = (top_idx != s_grid)
        if not move_mask.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            return out_flat.view(T, B, C, H, W)

        # 压平成边集 E'
        s_flat  = s_grid[move_mask]     # [E']
        j_flat  = j_grid[move_mask]     # [E']
        t_flat  = top_idx[move_mask]    # [E']
        score   = top_vals[move_mask]   # [E']
        dist    = (t_flat - s_flat).abs()

        # ---- 一次性排序：π 优先，同分短距优先 ----
        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = score + eps * ((T - 1) - dist).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]
        t_flat = t_flat[order]; score  = score[order]
        dist   = dist[order]

        # ---- 线性扫描放置 ----
        adv_act = torch.zeros_like(x_act)                       # [T,n]
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)    # 目标占用（每条线）
        reserved = has_src.transpose(1,0).clone()                     # 原位保留（每条线）
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)    # 源是否已移动

        B_rem = int(1e12) if step_budget is None else int(max(0, step_budget))

        E = s_flat.numel()
        for k in range(E):
            if B_rem <= 0:
                break
            s = int(s_flat[k].item())
            j = int(j_flat[k].item())
            t = int(t_flat[k].item())
            c = int(dist[k].item())
            if moved[j, s]: continue
            if occupied[j, t]: continue
            if reserved[j, t]: continue
            if c > B_rem: continue

            adv_act[t, j]  = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s]    = True
            B_rem -= c

        # ---- 剩余未放置的源回原位 ----
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1,0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)


    # ---------- 最终：严格投影（全时轴 + 全局整数 L1步数预算，跨线共享） ----------
    @torch.no_grad()
    def _strict_projection_allT_L1_active_global(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        l1_steps_budget: int,
        return_disp: bool = False,
    ):
        """
        严格投影（tight + no-loss + non-overlap），全局共享整数预算 B。
        仅在活跃线计算；与原功能等价，并可选返回每个源点的实际位移 Δ（t_target - t_src）。
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0 or l1_steps_budget <= 0:
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)                         # [T,n]
        if not has_src.any():
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        # --- 1) 构造 (s,j,t) 候选（跳过 t==s） ---
        s_j_pairs = torch.nonzero(has_src, as_tuple=False)           # [m,2], 行为 (s,j)
        m = s_j_pairs.shape[0]

        t_all    = torch.arange(T, device=dev, dtype=torch.long)     # [T]
        s_expand = s_j_pairs[:, 0].unsqueeze(1).expand(m, T)         # [m,T]
        j_expand = s_j_pairs[:, 1].unsqueeze(1).expand(m, T)         # [m,T]
        t_expand = t_all.unsqueeze(0).expand(m, T)                   # [m,T]

        mask_move = (t_expand != s_expand)                           # [m,T]
        if not mask_move.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            out_same = out_flat.view(T, B, C, H, W)
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        cand_scores = pi_act[s_expand[mask_move], j_expand[mask_move], t_expand[mask_move]]  # [E]
        s_flat = s_expand[mask_move]                                                         # [E]
        j_flat = j_expand[mask_move]                                                         # [E]
        t_flat = t_expand[mask_move]                                                         # [E]
        dist_flat = (t_flat - s_flat).abs()                                                  # [E]

        # --- 2) 排序：π 优先，同分短距优先 ---
        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = cand_scores + eps * ((T - 1) - dist_flat).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]; t_flat = t_flat[order]
        cand_scores = cand_scores[order]; dist_flat = dist_flat[order]

        # --- 3) 扫描放置（记录位移） ---
        adv_act  = torch.zeros_like(x_act)                                    # [T,n]
        disp_act = torch.zeros(T, n, device=dev, dtype=torch.int16)           # [T,n]  Δ = t - s
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)            # 目标占用
        reserved = has_src.transpose(1, 0).clone()                            # 原位保留
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)            # 源是否已移动

        B_rem = int(max(0, l1_steps_budget))

        E = s_flat.numel()
        for k in range(E):
            if B_rem <= 0:
                break
            s = int(s_flat[k].item())
            j = int(j_flat[k].item())
            t = int(t_flat[k].item())
            cost = int(dist_flat[k].item())

            if moved[j, s]:
                continue
            if occupied[j, t]:
                continue
            if reserved[j, t]:
                continue
            if cost > B_rem:
                # 保持与原实现一致：遇到大于剩余额度的候选直接停止扫描
                break

            # 放置并记录 Δ
            adv_act[t, j] = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s] = True
            disp_act[s, j] = int(t - s)          # 记录实际位移步数
            B_rem -= cost

        # --- 4) 剩余源回原位（Δ=0，不消耗预算） ---
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1, 0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]
            # disp_act 对这些位置保持 0

        # --- 5) 写回 ---
        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        adv_tbchw = out_flat.view(T, B, C, H, W)

        out_disp_flat = torch.zeros_like(x_flat, dtype=torch.int16)
        out_disp_flat[:, idx] = disp_act
        delta_tbchw = out_disp_flat.view(T, B, C, H, W)

        if return_disp:
            return adv_tbchw, delta_tbchw
        return adv_tbchw



    # ---------- 期望占用惩罚（可选；按“包数期望”，与原逻辑一致） ----------
    def _occupancy_penalty_active(self, x_src_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        """
        仅供训练稳定（若不需要可令 lambda_cap=0）：
        occ[t] = Σ_s 1_{src>0} * pi[s,t]    （只按包数，不看幅值）
        """
        T = x_src_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_src_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)                # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros((), device=x_src_tbchw.device, dtype=x_src_tbchw.dtype)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        src_mask = (x_act > 0).float()                          # [T,n]

        # occ[t,n] = sum_s src_mask[s,n]*pi[s,n,t]
        occ = torch.einsum('sn,snt->tn', src_mask, pi_act)      # [T,n]

        overflow = (occ - self.cap_limit).clamp_min(0.0)
        # 缩放回“全量平均”的量级（与你原始写法一致）
        num_all     = x_src_tbchw.numel() // T                  # B*C*H*W
        num_active  = overflow.numel()
        scale = (num_active / (T * num_all))
        return (overflow * overflow).mean() * scale

    # ---------- forward ----------
    def forward(self, x_enc_tbchw: torch.Tensor, labels: torch.Tensor, return_disp: bool = False, use_PIL: bool = True, use_cap: bool = True, use_penalty: bool = True, target_label: int = -1):
        x_enc_tbchw = x_enc_tbchw.to(self.device)
        labels = labels.to(self.device)

        T, Bsz, C, H, W = x_enc_tbchw.shape

        # φ: [T,B,C,H,W,T] —— 每个源时刻 s 的目标时刻 t 的 logit
        phi = torch.zeros(T, Bsz, C, H, W, T, device=self.device, dtype=x_enc_tbchw.dtype)

        src_mask = (x_enc_tbchw != 0).unsqueeze(-1)  # [T,B,C,H,W,1]

        def _mask_hook(grad):
            return grad * src_mask.float()

        phi.requires_grad_(True)
        phi.register_hook(_mask_hook)

        if self.random_start:
            phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)

        # === 预计算步数代价矩阵 C_{s,t} = |t - s|（只看位移步数，不看幅值） ===
        # 形状: [T,1,1,1,1,T]，可与 pi_st / src_mask 广播
        ar_s = torch.arange(T, device=self.device, dtype=x_enc_tbchw.dtype).view(T, 1, 1, 1, 1, 1)
        ar_t = torch.arange(T, device=self.device, dtype=x_enc_tbchw.dtype).view(1, 1, 1, 1, 1, T)
        Cst = (ar_t - ar_s).abs()  # [T,1,1,1,1,T]

        # 训练期：soft + 宽松投影（全时轴），cap(可选) + 预算超限才惩罚；λ_B 做对偶更新
        alpha = torch.ones(self.M, device=self.device) / self.M
        for t in range(self.steps):
            tau = self.temperature
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi_st = F.softmax(phi_masked / tau, dim=-1)   # [T,B,C,H,W,T]

            x_soft = self._soft_shift_with_pi_active(x_enc_tbchw, pi_st)
            if use_PIL:
                with torch.no_grad():
                    # x_hard = self._relaxed_projection_allT_active_strict_lite_topk(x_enc_tbchw, pi_st, 2, self.l1_steps_budget)
                    x_hard = self._strict_projection_allT_L1_active_global(x_enc_tbchw, pi_st, self.l1_steps_budget)
                x_pil = x_hard + (x_soft - x_soft.detach())
            else:
                x_pil = x_soft

            # === 软的总位移步数 S_soft：逐样本统计，再取平均 ===
            S_soft_b = ((pi_st * Cst) * src_mask).sum(dim=(0, 2, 3, 4, 5))  # [Bsz]
            over_b = F.relu(S_soft_b - self.l1_steps_budget)                 # [Bsz]
            if use_penalty:
                penalty_budget = 10 * over_b.mean()/self.l1_steps_budget
            else:
                penalty_budget = 0

            # cap（保留；不用就把 lambda_cap 设 0）
            if use_cap:
                cap = self._occupancy_penalty_active(x_enc_tbchw, pi_st) if self.lambda_cap != 0 else 0
            else:
                cap = 0

            # ======= 关键改动：多模型融合 ======= #
            ce_list = []
            for m in self.models:
                logits_m = m(x_pil)
                if target_label < 0:
                    ce_m = self.loss_fn(logits_m, labels)
                else:
                    ce_m = self.loss_fn(logits_m, target_label * torch.ones_like(labels))
                ce_list.append(ce_m)

            ce_stack = torch.stack(ce_list)      # [M]

            # 用当前 alpha 做混合损失
            alpha_sum = alpha.sum().detach()
            F_mix = (alpha * ce_stack).sum() / alpha_sum

            if target_label < 0:
                loss = F_mix - self.lambda_cap * cap - penalty_budget
            else:
                loss = -F_mix - self.lambda_cap * cap - penalty_budget

            # 对 phi 求梯度（跟之前一样）
            grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                phi += self.alpha_phi * grad.sign()
                phi.clamp_(-10.0, 10.0)
                phi.requires_grad_(True)

            # ===== 这里新增：更新 alpha =====
            with torch.no_grad():
                if target_label < 0:
                    alpha = alpha - self.alpha_lr * ce_stack.detach()
                else:
                    alpha = alpha + self.alpha_lr * ce_stack.detach()

                # 限制为非负并归一化
                alpha.clamp_(min=0.0)
                alpha /= (alpha.mean() + 1e-8)

            print(F_mix.detach())

        # 最终：严格投影（全时轴 + 全局整数 L1(步数) 预算，跨线共享）
        with torch.no_grad():
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi_st = F.softmax(phi_masked / self.temperature, dim=-1)
            if return_disp:
                adv_tbchw, delta_tbchw = self._strict_projection_allT_L1_active_global(x_enc_tbchw, pi_st, self.l1_steps_budget, return_disp=return_disp)
                return adv_tbchw, delta_tbchw 
            else:
                adv_tbchw = self._strict_projection_allT_L1_active_global(x_enc_tbchw, pi_st, self.l1_steps_budget)
                return adv_tbchw


class PGDTimeShiftAfterEncoder_L0_Multi(nn.Module):
    """
    Spike Timing Attack（after-encoder, 全时轴可移；L0 预算 = 移动的“点数”）
      - 输入/输出: [T,B,C,H,W]
      - 训练期: soft shift（全时轴） + （保持你当前习惯）用严格投影的 x_hard 做 PIL
      - 最终: 严格投影（tight + no-loss + non-overlap），并满足 全局 L0(移动个数) 预算（整数）
      - 预算只按“移动个数”计数（t != s 则成本=1；不看距离）
    """
    def __init__(
        self,
        device,
        models_without_encoder: nn.Module,
        reduction: str = "mean",
        steps: int = 40,
        alpha_phi: float = 1.0,
        lambda_cap: float = 20.0,           # 若不用 cap，可置 0
        temperature: float = 1.0,
        random_start: bool = False,
        cap_limit: float = 1.0,
        # L0 预算（整数移动个数）+ 对偶变量与其步长（命名保持一致，方便替换原代码）
        l0_moves_budget: int = 1000,
        lambda_B: float = 0.0,
        dual_lr: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList(
            [SNNWrapTimeMajor(m, reduction=reduction) for m in models_without_encoder]
        )
        self.M = len(self.models)
        self.steps = int(steps)
        self.alpha_phi = float(alpha_phi)
        self.lambda_cap = float(lambda_cap)
        self.temperature = float(temperature)
        self.random_start = bool(random_start)
        self.cap_limit = float(cap_limit)
        self.loss_fn = nn.CrossEntropyLoss()

        # 预算（整数“移动个数”），对偶乘子与步长
        self.l0_moves_budget = int(l0_moves_budget)
        self.lambda_B = float(lambda_B)
        self.dual_lr = float(dual_lr)
        self.alpha_lr = 0.05  # 比较小的步长，自己调

    # ---------- 工具 ----------
    @staticmethod
    def _flatten_tbchw(x):  # [T,B,C,H,W] -> (x_flat[T,N], (B,C,H,W,N))
        T, B, C, H, W = x.shape
        N = B*C*H*W
        return x.view(T, N), (B, C, H, W, N)

    @staticmethod
    def _active_index(x_flat):  # x_flat: [T,N]
        active = (x_flat != 0).any(dim=0)           # [N]
        idx = active.nonzero(as_tuple=False).squeeze(1)  # [n]
        return active, idx

    # ---------- soft shift（全时轴，活跃线） ----------
    def _soft_shift_with_pi_active(self, x_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        """
        x: [T,B,C,H,W], pi_st: [T,B,C,H,W,T]  （每个源时刻 s 的目标分布 t）
        只在活跃线计算： out[t] = Σ_s x[s] * pi[s, t]
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]

        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]

        # out_act[t,n] = sum_s x_act[s,n] * pi_act[s,n,t]
        out_act = torch.einsum('sn,snt->tn', x_act, pi_act)   # [T,n]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = out_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ---------- 训练期：Strict-Lite（Top-k）松弛投影（这里按“移动个数”计预算） ----------
    @torch.no_grad()
    def _relaxed_projection_allT_active_strict_lite_topk(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        topk: int = 1,
        step_budget: int | None = None,  # 训练期可设为 moves_B // steps；不想限就 None
    ) -> torch.Tensor:
        """
        Strict-Lite（Top-k）松弛投影（L0 版本）：
          - 候选：每个 (s,j) 取 Top-k 的 t（去掉 t==s）
          - 全局按 π 降序（同分短距优先做次序而已，预算仍是“计数”）
          - 线性扫：不重合、原位保留、未放置回原位
          - 预算：每成功放置一个，B_rem -= 1
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return x_tbchw.clone()

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)                         # [T,n]
        if not has_src.any():
            return x_tbchw.clone()

        # ---- 每个 (s,j) 的 Top-k 候选 ----
        k = int(max(1, topk))
        top_vals, top_idx = torch.topk(pi_act, k=min(k, T), dim=-1)     # over t
        s_grid = torch.arange(T, device=dev).view(T,1,1).expand(T,n,k)
        j_grid = torch.arange(n, device=dev).view(1,n,1).expand(T,n,k)

        # 去掉 t==s
        move_mask = (top_idx != s_grid)
        if not move_mask.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            return out_flat.view(T, B, C, H, W)

        # 压平成边集
        s_flat  = s_grid[move_mask]
        j_flat  = j_grid[move_mask]
        t_flat  = top_idx[move_mask]
        score   = top_vals[move_mask]
        dist    = (t_flat - s_flat).abs()

        # 排序键：π 优先，同分短距优先（仅作 tie-breaker）
        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = score + eps * ((T - 1) - dist).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]
        t_flat = t_flat[order]; score  = score[order]
        # dist   = dist[order]  # L0 不用成本，保留与否均可

        # 放置
        adv_act = torch.zeros_like(x_act)                       # [T,n]
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)
        reserved = has_src.transpose(1,0).clone()
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)

        B_rem = int(1e12) if step_budget is None else int(max(0, step_budget))

        E = s_flat.numel()
        for k_ in range(E):
            if B_rem <= 0:
                break
            s = int(s_flat[k_].item())
            j = int(j_flat[k_].item())
            t = int(t_flat[k_].item())
            if moved[j, s]: continue
            if occupied[j, t]: continue
            if reserved[j, t]: continue

            adv_act[t, j]  = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s]    = True
            B_rem -= 1  # L0：每移动一个，预算减 1

        # 未放置的回原位
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1,0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    @torch.no_grad()
    def _strict_projection_allT_L0_active_global(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        l0_moves_budget: int,
        return_disp: bool = False,
    ):
        """
        严格投影（tight + no-loss + non-overlap），全局共享整数 L0 预算（移动“点数”）。
        功能与原版一致；可选返回每个源点的实际位移 Δ（t_target - t_src）。
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0 or l0_moves_budget <= 0:
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)         # [T,n]
        if not has_src.any():
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        # --- 构造候选集（去掉 t==s） ---
        s_j_pairs = torch.nonzero(has_src, as_tuple=False)        # [m,2] (s,j)
        m = s_j_pairs.shape[0]
        t_all    = torch.arange(T, device=dev, dtype=torch.long)  # [T]
        s_expand = s_j_pairs[:, 0].unsqueeze(1).expand(m, T)      # [m,T]
        j_expand = s_j_pairs[:, 1].unsqueeze(1).expand(m, T)      # [m,T]
        t_expand = t_all.unsqueeze(0).expand(m, T)                # [m,T]

        mask_move = (t_expand != s_expand)                        # [m,T]
        if not mask_move.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            out_same = out_flat.view(T, B, C, H, W)
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        # 得分与距离（距离仅作同分次序的 tie-breaker）
        cand_scores = pi_act[s_expand[mask_move], j_expand[mask_move], t_expand[mask_move]]  # [E]
        s_flat = s_expand[mask_move]                                                         # [E]
        j_flat = j_expand[mask_move]                                                         # [E]
        t_flat = t_expand[mask_move]                                                         # [E]
        dist_flat = (t_flat - s_flat).abs()                                                  # [E]

        # --- 排序：π 优先，同分短距优先 ---
        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = cand_scores + eps * ((T - 1) - dist_flat).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]; t_flat = t_flat[order]
        # cand_scores = cand_scores[order]  # 如需调试可保留
        # dist_flat   = dist_flat[order]

        # --- 放置（记录位移） ---
        adv_act  = torch.zeros_like(x_act)                              # [T,n]
        disp_act = torch.zeros(T, n, device=dev, dtype=torch.int16)     # [T,n] Δ = t - s
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)      # 目标占用
        reserved = has_src.transpose(1, 0).clone()                      # 原位保留
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)      # 源是否已移动

        B_rem = int(max(0, l0_moves_budget))

        E = s_flat.numel()
        for k_ in range(E):
            if B_rem <= 0:
                break
            s = int(s_flat[k_].item())
            j = int(j_flat[k_].item())
            t = int(t_flat[k_].item())

            if moved[j, s]:
                continue
            if occupied[j, t]:
                continue
            if reserved[j, t]:
                continue

            adv_act[t, j]  = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s]    = True
            disp_act[s, j] = int(t - s)   # 记录实际位移步数 Δ
            B_rem -= 1                     # L0：每移动一个，预算减 1

        # 未移动者回原位（Δ=0）
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1, 0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

        # --- 写回原形状 ---
        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        adv_tbchw = out_flat.view(T, B, C, H, W)

        out_disp_flat = torch.zeros_like(x_flat, dtype=torch.int16)
        out_disp_flat[:, idx] = disp_act
        delta_tbchw = out_disp_flat.view(T, B, C, H, W)

        if return_disp:
            return adv_tbchw, delta_tbchw
        return adv_tbchw


    # ---------- 期望占用惩罚（与你原逻辑一致，可选） ----------
    def _occupancy_penalty_active(self, x_src_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        T = x_src_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_src_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)                # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros((), device=x_src_tbchw.device, dtype=x_src_tbchw.dtype)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        src_mask = (x_act > 0).float()

        # occ[t,n] = sum_s src_mask[s,n]*pi[s,n,t]
        occ = torch.einsum('sn,snt->tn', src_mask, pi_act)      # [T,n]

        overflow = (occ - self.cap_limit).clamp_min(0.0)
        num_all     = x_src_tbchw.numel() // T
        num_active  = overflow.numel()
        scale = (num_active / (T * num_all))
        return (overflow * overflow).mean() * scale

    # ---------- forward ----------
    def forward(self, x_enc_tbchw: torch.Tensor, labels: torch.Tensor, return_disp: bool = False, use_PIL: bool = True, use_cap: bool = True, use_penalty: bool = True, target_label: int = -1):
        x_enc_tbchw = x_enc_tbchw.to(self.device)
        labels = labels.to(self.device)

        T, Bsz, C, H, W = x_enc_tbchw.shape

        # φ: [T,B,C,H,W,T]
        phi = torch.zeros(T, Bsz, C, H, W, T, device=self.device, dtype=x_enc_tbchw.dtype)
        src_mask = (x_enc_tbchw != 0).unsqueeze(-1)  # [T,B,C,H,W,1]

        def _mask_hook(grad):
            return grad * src_mask.float()
        phi.requires_grad_(True)
        phi.register_hook(_mask_hook)

        if self.random_start:
            phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)
            # （按你原版保持一致：不重复挂 hook）

        # 训练循环
        alpha = torch.ones(self.M, device=self.device) / self.M
        for _ in range(self.steps):
            tau = self.temperature
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi_st = F.softmax(phi_masked / tau, dim=-1)   # [T,B,C,H,W,T]

            x_soft = self._soft_shift_with_pi_active(x_enc_tbchw, pi_st)
            if use_PIL:
                with torch.no_grad():
                    # 保持你当前习惯：训练期也用严格投影作为 x_hard（L0 版本）
                    x_hard = self._strict_projection_allT_L0_active_global(x_enc_tbchw, pi_st, self.l0_moves_budget)
                    # 若想更快：可改成 strict-lite-topk，并把 step_budget 设为 self.l0_moves_budget//self.steps
                x_pil = x_hard + (x_soft - x_soft.detach())
            else:
                x_pil = x_soft

            # cap（保留；不用就把 lambda_cap 设 0）
            if use_cap:
                cap = self._occupancy_penalty_active(x_enc_tbchw, pi_st) if self.lambda_cap != 0 else 0
            else:
                cap = 0

            # === L0 的 soft 预算：期望移动“个数” ===
            pi_diag = pi_st.diagonal(dim1=0, dim2=-1).permute(4,0,1,2,3)  # [T,B,C,H,W]
            move_prob = 1.0 - pi_diag                                     # [T,B,C,H,W]
            S_soft_b = (move_prob * src_mask.squeeze(-1).float()).sum(dim=(0, 2, 3, 4))  # [B]

            over_b = F.relu(S_soft_b - self.l0_moves_budget)              # [B]
            # 为保持你当前代码风格：用一个固定系数（不直接用 lambda_B）
            if use_penalty:
                penalty_budget = 5 * over_b.mean() / max(1, self.l0_moves_budget)
            else:
                penalty_budget = 0

            # ======= 关键改动：多模型融合 ======= #
            ce_list = []
            for m in self.models:
                logits_m = m(x_pil)
                if target_label < 0:
                    ce_m = self.loss_fn(logits_m, labels)
                else:
                    ce_m = self.loss_fn(logits_m, target_label * torch.ones_like(labels))
                ce_list.append(ce_m)

            ce_stack = torch.stack(ce_list)      # [M]

            # 用当前 alpha 做混合损失
            alpha_sum = alpha.sum().detach()
            F_mix = (alpha * ce_stack).sum() / alpha_sum

            if target_label < 0:
                loss = F_mix - self.lambda_cap * cap - penalty_budget
            else:
                loss = -F_mix - self.lambda_cap * cap - penalty_budget

            # 对 phi 求梯度（跟之前一样）
            grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                phi += self.alpha_phi * grad.sign()
                phi.clamp_(-10.0, 10.0)
                phi.requires_grad_(True)

            # ===== 这里新增：更新 alpha =====
            with torch.no_grad():
                if target_label < 0:
                    alpha = alpha - self.alpha_lr * ce_stack.detach()
                else:
                    alpha = alpha + self.alpha_lr * ce_stack.detach()

                # 限制为非负并归一化
                alpha.clamp_(min=0.0)
                alpha /= (alpha.mean() + 1e-8)

            print(F_mix.detach())


        # 最终：严格投影（全候选 + 全局整数 L0 预算）
        with torch.no_grad():
            neg_inf = torch.finfo(phi.dtype).min
            phi_masked = torch.where(src_mask, phi, torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype))
            pi_st = F.softmax(phi_masked / self.temperature, dim=-1)
            if return_disp:
                adv_tbchw, delta_tbchw = self._strict_projection_allT_L0_active_global(x_enc_tbchw, pi_st, self.l0_moves_budget, return_disp=return_disp)
                return adv_tbchw, delta_tbchw 
            else:
                adv_tbchw = self._strict_projection_allT_L0_active_global(x_enc_tbchw, pi_st, self.l0_moves_budget)
                return adv_tbchw


class PGDTimeShiftAfterEncoder_MultiNorm_Multi(nn.Module):
    """
    Multi-model Spike Timing Attack with multi-norm budgets:
      - 输入/输出: [T,B,C,H,W]
      - 同时满足:
          * B_inf: |Δ| <= D  (local jitter bound)
          * B_1  : sum |Δ| <= l1_steps_budget (全局 L1 步数预算)
          * B_0  : #moved   <= l0_moves_budget (全局 L0 移动次数预算)
      - 模型: models_without_encoder = [net1, ..., netM] (SNN, time-major)
      - 训练期: soft shift + 松弛投影 (L1+L0) 做 PIL
      - 最终: 严格投影 (tight + no-loss + non-overlap + 三种预算同时满足)
    """
    def __init__(
        self,
        device,
        models_without_encoder,          # list-like [net1, net2, ...]
        reduction: str = "mean",
        steps: int = 40,
        alpha_phi: float = 1.0,
        lambda_cap: float = 20.0,
        temperature: float = 1.0,
        random_start: bool = False,
        cap_limit: float = 1.0,
        # 三种预算
        D: int = 3,                      # B_inf: 最大步数
        l1_steps_budget: int = 1000,     # B_1: 全局 L1 步数预算
        l0_moves_budget: int = 400,     # B_0: 全局 L0 移动个数预算
        # soft 预算惩罚 & alpha 更新步长
        alpha_lr: float = 0.05,
        w_l1_penalty: float = 10.0,
        w_l0_penalty: float = 5.0,
    ):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList(
            [SNNWrapTimeMajor(m, reduction=reduction) for m in models_without_encoder]
        )
        self.M = len(self.models)

        self.steps = int(steps)
        self.alpha_phi = float(alpha_phi)
        self.lambda_cap = float(lambda_cap)
        self.temperature = float(temperature)
        self.random_start = bool(random_start)
        self.cap_limit = float(cap_limit)
        self.loss_fn = nn.CrossEntropyLoss()

        # budgets
        self.D = int(D)
        self.l1_steps_budget = int(l1_steps_budget)
        self.l0_moves_budget = int(l0_moves_budget)

        self.alpha_lr = float(alpha_lr)
        self.w_l1_penalty = float(w_l1_penalty)
        self.w_l0_penalty = float(w_l0_penalty)

    # ---------- 工具 ----------
    @staticmethod
    def _flatten_tbchw(x):  # [T,B,C,H,W] -> (x_flat[T,N], (B,C,H,W,N))
        T, B, C, H, W = x.shape
        N = B * C * H * W
        return x.view(T, N), (B, C, H, W, N)

    @staticmethod
    def _active_index(x_flat):  # x_flat: [T,N]
        active = (x_flat != 0).any(dim=0)  # [N]
        idx = active.nonzero(as_tuple=False).squeeze(1)  # [n]
        return active, idx

    # ---------- soft shift（全时轴，活跃线） ----------
    def _soft_shift_with_pi_active(self, x_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        """
        x: [T,B,C,H,W], pi_st: [T,B,C,H,W,T]  （每个源时刻 s 的目标分布 t）
        只在活跃线计算： out[t] = Σ_s x[s] * pi[s, t]
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]

        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros_like(x_tbchw)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]

        # out_act[t,n] = sum_s x_act[s,n] * pi_act[s,n,t]
        out_act = torch.einsum('sn,snt->tn', x_act, pi_act)   # [T,n]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = out_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ---------- 训练期：L1+L0 松弛投影（Top-k, 全局双预算，|Δ|<=D） ----------
    @torch.no_grad()
    def _relaxed_projection_allT_L1L0_active_strict_lite_topk(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        topk: int,
        l1_step_budget: int | None,
        l0_move_budget: int | None,
    ) -> torch.Tensor:
        """
        Strict-Lite (Top-k) 松弛投影，同时考虑：
          - |Δ| <= D
          - sum|Δ| <= l1_step_budget (若 None 则不约束)
          - #move <= l0_move_budget (若 None 则不约束)
        未放置的源回原位；值守恒；不重合。
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return x_tbchw.clone()

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)                         # [T,n]
        if not has_src.any():
            return x_tbchw.clone()

        # 每个 (s,j) 的 Top-k 候选
        k = int(max(1, topk))
        top_vals, top_idx = torch.topk(pi_act, k=min(k, T), dim=-1)     # over t
        s_grid = torch.arange(T, device=dev).view(T,1,1).expand(T,n,k)
        j_grid = torch.arange(n, device=dev).view(1,n,1).expand(T,n,k)

        # 去掉 t==s，且 |t-s| <= D
        dist_grid = (top_idx - s_grid).abs()
        move_mask = (top_idx != s_grid) & (dist_grid <= self.D)
        if not move_mask.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            return out_flat.view(T, B, C, H, W)

        s_flat  = s_grid[move_mask]
        j_flat  = j_grid[move_mask]
        t_flat  = top_idx[move_mask]
        score   = top_vals[move_mask]
        dist    = dist_grid[move_mask]

        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = score + eps * ((self.D) - dist).to(dtype)       # D 范围内短距优先
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]
        t_flat = t_flat[order]; score  = score[order]
        dist   = dist[order]

        adv_act = torch.zeros_like(x_act)                       # [T,n]
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)
        reserved = has_src.transpose(1,0).clone()
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)

        B1_rem = 10**12 if l1_step_budget is None else int(max(0, l1_step_budget))
        B0_rem = 10**12 if l0_move_budget is None else int(max(0, l0_move_budget))

        E = s_flat.numel()
        for k_ in range(E):
            if B1_rem <= 0 or B0_rem <= 0:
                break
            s = int(s_flat[k_].item())
            j = int(j_flat[k_].item())
            t = int(t_flat[k_].item())
            c = int(dist[k_].item())  # 步数成本

            if moved[j, s]:
                continue
            if occupied[j, t]:
                continue
            if reserved[j, t]:
                continue
            if c > B1_rem:
                continue
            if 1 > B0_rem:
                continue

            adv_act[t, j]  = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s]    = True
            B1_rem -= c
            B0_rem -= 1

        # 未放置的源回原位
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1,0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        return out_flat.view(T, B, C, H, W)

    # ---------- 严格投影：全候选 + D / L1 / L0 预算 ----------
    @torch.no_grad()
    def _strict_projection_allT_L1L0_active_global(
        self,
        x_tbchw: torch.Tensor,
        pi_st: torch.Tensor,
        l1_steps_budget: int,
        l0_moves_budget: int,
        return_disp: bool = False,
    ):
        """
        严格投影（tight + no-loss + non-overlap），同时满足：
          - |Δ| <= D
          - sum|Δ| <= l1_steps_budget
          - #move <= l0_moves_budget
        """
        T = x_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)            # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0 or l1_steps_budget <= 0 or l0_moves_budget <= 0:
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        dev, dtype = x_act.device, x_act.dtype
        n = x_act.shape[1]

        has_src = (x_act > 0)         # [T,n]
        if not has_src.any():
            out_same = x_tbchw.clone()
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        # --- 构造候选 (s,j,t)，t!=s 且 |t-s|<=D ---
        s_j_pairs = torch.nonzero(has_src, as_tuple=False)         # [m,2]
        m = s_j_pairs.shape[0]
        t_all    = torch.arange(T, device=dev, dtype=torch.long)   # [T]
        s_expand = s_j_pairs[:, 0].unsqueeze(1).expand(m, T)       # [m,T]
        j_expand = s_j_pairs[:, 1].unsqueeze(1).expand(m, T)       # [m,T]
        t_expand = t_all.unsqueeze(0).expand(m, T)                 # [m,T]

        dist_all = (t_expand - s_expand).abs()                     # [m,T]
        mask_move = (t_expand != s_expand) & (dist_all <= self.D)
        if not mask_move.any():
            adv_act = torch.zeros_like(x_act)
            adv_act[has_src] = x_act[has_src]
            out_flat = torch.zeros_like(x_flat)
            out_flat[:, idx] = adv_act
            B, C, H, W, N = shape_info
            out_same = out_flat.view(T, B, C, H, W)
            if return_disp:
                return out_same, torch.zeros_like(x_tbchw, dtype=torch.int16)
            return out_same

        cand_scores = pi_act[s_expand[mask_move], j_expand[mask_move], t_expand[mask_move]]  # [E]
        s_flat = s_expand[mask_move]                                                         # [E]
        j_flat = j_expand[mask_move]                                                         # [E]
        t_flat = t_expand[mask_move]                                                         # [E]
        dist_flat = dist_all[mask_move]                                                      # [E]

        eps = torch.tensor(1e-6, device=dev, dtype=dtype)
        key = cand_scores + eps * (self.D - dist_flat).to(dtype)
        order = torch.argsort(key, descending=True)
        s_flat = s_flat[order]; j_flat = j_flat[order]; t_flat = t_flat[order]
        dist_flat = dist_flat[order]

        adv_act  = torch.zeros_like(x_act)                              # [T,n]
        disp_act = torch.zeros(T, n, device=dev, dtype=torch.int16)     # [T,n]
        occupied = torch.zeros(n, T, dtype=torch.bool, device=dev)
        reserved = has_src.transpose(1, 0).clone()
        moved    = torch.zeros(n, T, dtype=torch.bool, device=dev)

        B1_rem = int(max(0, l1_steps_budget))
        B0_rem = int(max(0, l0_moves_budget))

        E = s_flat.numel()
        for k_ in range(E):
            if B1_rem <= 0 or B0_rem <= 0:
                break
            s = int(s_flat[k_].item())
            j = int(j_flat[k_].item())
            t = int(t_flat[k_].item())
            cost = int(dist_flat[k_].item())   # |Δ|

            if moved[j, s]:
                continue
            if occupied[j, t]:
                continue
            if reserved[j, t]:
                continue
            if cost > B1_rem:
                break  # 保持与 L1 版本一致：遇到 cost>B_rem 直接 break

            adv_act[t, j] = x_act[s, j]
            occupied[j, t] = True
            reserved[j, s] = False
            moved[j, s] = True
            disp_act[s, j] = int(t - s)
            B1_rem -= cost
            B0_rem -= 1

        # 未移动的回原位
        stay_s, stay_j = torch.nonzero(has_src & (~moved.transpose(1, 0)), as_tuple=True)
        if stay_s.numel() > 0:
            adv_act[stay_s, stay_j] = x_act[stay_s, stay_j]

        out_flat = torch.zeros_like(x_flat)
        out_flat[:, idx] = adv_act
        B, C, H, W, N = shape_info
        adv_tbchw = out_flat.view(T, B, C, H, W)

        out_disp_flat = torch.zeros_like(x_flat, dtype=torch.int16)
        out_disp_flat[:, idx] = disp_act
        delta_tbchw = out_disp_flat.view(T, B, C, H, W)

        if return_disp:
            return adv_tbchw, delta_tbchw
        return adv_tbchw

    # ---------- 期望占用惩罚（与你现在的 active 写法一致） ----------
    def _occupancy_penalty_active(self, x_src_tbchw: torch.Tensor, pi_st: torch.Tensor) -> torch.Tensor:
        T = x_src_tbchw.shape[0]
        x_flat, shape_info = self._flatten_tbchw(x_src_tbchw)     # [T,N]
        pi_flat = pi_st.view(T, shape_info[-1], T)                # [T,N,T]
        _, idx = self._active_index(x_flat)
        if idx.numel() == 0:
            return torch.zeros((), device=x_src_tbchw.device, dtype=x_src_tbchw.dtype)

        x_act  = x_flat[:, idx]       # [T,n]
        pi_act = pi_flat[:, idx, :]   # [T,n,T]
        src_mask = (x_act > 0).float()

        occ = torch.einsum('sn,snt->tn', src_mask, pi_act)      # [T,n]
        overflow = (occ - self.cap_limit).clamp_min(0.0)
        num_all     = x_src_tbchw.numel() // T
        num_active  = overflow.numel()
        scale = (num_active / (T * num_all))
        return (overflow * overflow).mean() * scale

    def forward(
        self,
        x_enc_tbchw: torch.Tensor,
        labels: torch.Tensor,
        return_disp: bool = False,
        use_PIL: bool = True,
        use_cap: bool = True,
        use_penalty: bool = True,
        target_label: int = -1,
    ):
        x_enc_tbchw = x_enc_tbchw.to(self.device)
        labels = labels.to(self.device)

        T, Bsz, C, H, W = x_enc_tbchw.shape

        # φ: [T,B,C,H,W,T]
        phi = torch.zeros(T, Bsz, C, H, W, T,
                          device=self.device,
                          dtype=x_enc_tbchw.dtype)
        src_mask = (x_enc_tbchw != 0).unsqueeze(-1)  # [T,B,C,H,W,1]

        # D-mask：|t-s| <= D
        ar_s = torch.arange(T, device=self.device, dtype=x_enc_tbchw.dtype).view(T,1,1,1,1,1)
        ar_t = torch.arange(T, device=self.device, dtype=x_enc_tbchw.dtype).view(1,1,1,1,1,T)
        D_mask = (ar_t - ar_s).abs() <= self.D      # [T,1,1,1,1,T]

        def _mask_hook(grad):
            return grad * src_mask.float()
        phi.requires_grad_(True)
        phi.register_hook(_mask_hook)

        if self.random_start:
            phi = (phi + 0.01 * torch.randn_like(phi)).detach().requires_grad_(True)

        # soft L1 的 Cst
        Cst = (ar_t - ar_s).abs()  # [T,1,1,1,1,T]

        # multi-model 权重
        alpha = torch.ones(self.M, device=self.device) / self.M

        for _ in range(self.steps):
            tau = self.temperature
            neg_inf = torch.finfo(phi.dtype).min

            # 只允许 (src_mask & D_mask) 的位置有 logits
            allowed = src_mask & D_mask
            phi_masked = torch.where(
                allowed,
                phi,
                torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype),
            )
            pi_st = F.softmax(phi_masked / tau, dim=-1)   # [T,B,C,H,W,T]

            # soft shift
            x_soft = self._soft_shift_with_pi_active(x_enc_tbchw, pi_st)

            # === 关键改动：训练期也用严格投影（和 L1 / L0 原版保持一致） ===
            if use_PIL:
                with torch.no_grad():
                    x_hard = self._strict_projection_allT_L1L0_active_global(
                        x_enc_tbchw,
                        pi_st,
                        l1_steps_budget=self.l1_steps_budget,
                        l0_moves_budget=self.l0_moves_budget,
                        return_disp=False,
                    )
                x_pil = x_hard + (x_soft - x_soft.detach())
            else:
                x_pil = x_soft

            # -------- soft 预算统计（可选，仅做轻度 regularization） --------
            src_mask_float = src_mask.float()

            # soft L1：期望总步数
            S1_soft_b = ((pi_st * Cst) * src_mask_float).sum(dim=(0, 2, 3, 4, 5))  # [B]
            over1_b = F.relu(S1_soft_b - self.l1_steps_budget)

            # soft L0：期望移动个数
            pi_diag = pi_st.diagonal(dim1=0, dim2=-1).permute(4, 0, 1, 2, 3)      # [T,B,C,H,W]
            move_prob = 1.0 - pi_diag
            S0_soft_b = (move_prob * src_mask_float.squeeze(-1)).sum(dim=(0, 2, 3, 4))  # [B]
            over0_b = F.relu(S0_soft_b - self.l0_moves_budget)

            if use_penalty and (self.w_l1_penalty > 0 or self.w_l0_penalty > 0):
                penalty_l1 = over1_b.mean() / (self.l1_steps_budget + 1e-8)
                penalty_l0 = over0_b.mean() / (self.l0_moves_budget + 1e-8)
                penalty_budget = self.w_l1_penalty * penalty_l1 + self.w_l0_penalty * penalty_l0
            else:
                penalty_budget = 0.0

            # cap 惩罚（保持和原始 active 写法一致）
            if use_cap:
                cap = (self._occupancy_penalty_active(x_enc_tbchw, pi_st)
                       if self.lambda_cap != 0 else 0)
            else:
                cap = 0

            # ======= 多模型交叉熵 ======= #
            ce_list = []
            for m in self.models:
                logits_m = m(x_pil)
                if target_label < 0:
                    ce_m = self.loss_fn(logits_m, labels)
                else:
                    ce_m = self.loss_fn(
                        logits_m,
                        target_label * torch.ones_like(labels),
                    )
                ce_list.append(ce_m)
            ce_stack = torch.stack(ce_list)      # [M]

            # 固定把 alpha 归一化到 sum=1
            alpha = alpha / (alpha.sum() + 1e-8)
            F_mix = (alpha * ce_stack).sum()

            if target_label < 0:
                loss = F_mix - self.lambda_cap * cap - penalty_budget
            else:
                loss = -F_mix - self.lambda_cap * cap - penalty_budget

            # 对 φ 做 PGD 上升
            grad = torch.autograd.grad(loss, phi, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                phi += self.alpha_phi * grad.sign()
                phi.clamp_(-10.0, 10.0)
                phi.requires_grad_(True)

            # ===== alpha 更新（如果先想 debug，可以把 alpha_lr 设为 0） =====
            if self.alpha_lr > 0:
                with torch.no_grad():
                    if target_label < 0:
                        # untargeted：攻得越好 (CE 越大) 权重越小，照顾难攻模型
                        alpha = alpha - self.alpha_lr * ce_stack.detach()
                    else:
                        # targeted：CE 越小越好，反向
                        alpha = alpha + self.alpha_lr * ce_stack.detach()

                    alpha.clamp_(min=0.0)
                    alpha = alpha / (alpha.sum() + 1e-8)

            print(F_mix.detach())

        # -------- 最终：严格投影，真·满足 B_inf + B_1 + B_0 --------
        with torch.no_grad():
            neg_inf = torch.finfo(phi.dtype).min
            allowed = src_mask & D_mask
            phi_masked = torch.where(
                allowed,
                phi,
                torch.as_tensor(neg_inf, device=phi.device, dtype=phi.dtype),
            )
            pi_st = F.softmax(phi_masked / self.temperature, dim=-1)
            if return_disp:
                adv_tbchw, delta_tbchw = self._strict_projection_allT_L1L0_active_global(
                    x_enc_tbchw, pi_st,
                    l1_steps_budget=self.l1_steps_budget,
                    l0_moves_budget=self.l0_moves_budget,
                    return_disp=True,
                )
                return adv_tbchw, delta_tbchw
            else:
                adv_tbchw = self._strict_projection_allT_L1L0_active_global(
                    x_enc_tbchw, pi_st,
                    l1_steps_budget=self.l1_steps_budget,
                    l0_moves_budget=self.l0_moves_budget,
                    return_disp=False,
                )
                return adv_tbchw
