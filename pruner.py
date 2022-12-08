import torch
from torch.nn import Conv2d
from torch.autograd import grad
from torch.nn.utils import prune
from copy import deepcopy

def fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break
    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    del dataloader_iter
    return X, y

def mp_importance_score(model):
    score_dict = {}
    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            score_dict[(m, 'weight')] = m.weight.data.abs()
    return score_dict

def snip_importance_score(
    model,
    dataloader, 
    samples_per_class,
    loss_func = torch.nn.CrossEntropyLoss()
    ):

    score_dict = {}
    model.zero_grad()
    device = next(model.parameters()).device
    x, y = fetch_data(dataloader, model.fc.out_features, samples_per_class)
    x, y = x.to(device), y.to(device)
    loss = loss_func(model(x), y)
    loss.backward()
    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            score_dict[(m, 'weight')] = m.weight.grad.clone().detach().abs()
    model.zero_grad()
    return score_dict

def grasp_importance_score(
    model,
    dataloader, 
    samples_per_class,
    loss_func = torch.nn.CrossEntropyLoss()
    ):

    score_dict = {}
    model.zero_grad()
    device = next(model.parameters()).device
    x, y = fetch_data(dataloader, model.fc.out_features, samples_per_class)
    x, y = x.to(device), y.to(device)
    loss = loss_func(model(x), y)
    gs = grad(loss, model.parameters(), create_graph=True)
    model.zero_grad()
    t = sum([(g*g.data).sum() for g in gs])
    t.backward()

    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            score_dict[(m, 'weight')] = -m.weight.clone().detach() * m.weight.grad.clone().detach()
    model.zero_grad()
    return score_dict

def synflow_importance_score(
    model,
    dataloader,
    ):
    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    model.eval() # Crucial! BatchNorm will break the conservation laws for synaptic saliency
    model.zero_grad()
    score_dict = {}
    signs = linearize(model)

    (data, _) = next(iter(dataloader))
    input_dim = list(data[0,:].shape)
    input = torch.ones([1] + input_dim).to(next(model.parameters()).device)
    output = model(input)
    torch.sum(output).backward()

    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            if hasattr(m, "weight_orig"):
                score_dict[(m, 'weight')] = (m.weight_orig.grad.clone().detach() * m.weight.clone().detach()).abs()
            else:
                score_dict[(m, 'weight')] = (m.weight.grad.clone().detach() * m.weight.clone().detach()).abs()
    model.zero_grad()
    nonlinearize(model, signs)
    return score_dict

def global_prune_model(model, ratio, method, dataloader=None, sample_per_classes=5):
    if method in ['snip', 'grasp']:
        score_dict = eval(f"{method}_importance_score")(model, dataloader, sample_per_classes)
        prune.global_unstructured(
            parameters=score_dict.keys(),
            pruning_method=prune.L1Unstructured,
            amount=ratio,
            importance_scores=score_dict,
        )
    elif method == 'synflow':
        iteration_number = 100 # In SynFlow Paper, an iteration number of 100 performs well
        each_ratio = 1 - (1-ratio)**(1/iteration_number)
        for _ in range(iteration_number):
            score_dict = synflow_importance_score(model, dataloader)
            prune.global_unstructured(
                parameters=score_dict.keys(),
                pruning_method=prune.L1Unstructured,
                amount=each_ratio,
                importance_scores=score_dict,
            )
    else:
        raise NotImplementedError(f'Pruning Method {method} not Implemented')
    torch.cuda.empty_cache()

def check_sparsity(model, if_print=False):
    sum_list = 0
    zero_sum = 0

    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    if zero_sum:
        remain_weight_ratie = 100*(1-zero_sum/sum_list)
        if if_print:
            print('* remain weight ratio = ', 100*(1-zero_sum/sum_list),'%')
    else:
        if if_print:
            print('no weight for calculating sparsity')
        remain_weight_ratie = 100

    return remain_weight_ratie

def prune_model_custom(model, mask_dict):
    # print('Pruning with custom mask (all conv layers)')
    for name, m in model.named_modules():
        if isinstance(m, (Conv2d,)):
            mask_name = name + '.weight_mask'
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print('Can not fing [{}] in mask_dict'.format(mask_name))

def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = deepcopy(model_dict[key])
    return new_dict

def remove_prune(model):
    # print('Remove hooks for multiplying masks (all conv layers)')
    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            prune.remove(m,'weight')