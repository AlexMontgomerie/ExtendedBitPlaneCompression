import sys

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

sys.path.append("algoEvals")

from dataCollect import getModel
from bpcUtils import quantize

def accuracy(output, target) :
    batchSize = target.size(0)
    # torch.topk returns the values and indices of the k(5) largest elements in dimension 1 in a sorted manner
    _, indices = output.topk(5, 1, True, True)
    indices.t_()
    correctPredictions = indices.eq(target.view(1,-1).expand_as(indices))

    res = []
    for k in (1,5) :
        correctK = correctPredictions[:k].reshape(-1).float().sum(0)
        res.append(correctK.mul_(100.0 / batchSize))

    return res


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

device = "cuda:0"
modelName = "alexnet"
quantisation = "fixed8"

criterion = torch.nn.CrossEntropyLoss().to(device, non_blocking=True)

# load the model
model, _ = getModel(modelName)

# create dataloader
data = datasets.ImageFolder(
	'/idslF/data/imagenet/validation',
	transform=transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(
			[0.485, 0.456, 0.406],
			[0.229, 0.224, 0.225]
		)
	])
)
dataloader = torch.utils.data.DataLoader(data,
        batch_size=128, shuffle=True, drop_last=False,
        num_workers=4)

# add quantisation as a forward hook
def quantise_forward_hook(m, i, o):
    if quantisation == "fixed8":
        type = torch.int8
        qmax = (2**7)-1
        qmin = -(2**7)
        scale = qmax
    if quantisation == "fixed16":
        type = torch.int16
        qmax = (2**15)-1
        qmin = -(2**15)
        scale = qmax

    # o = o.div(o.abs().max().item()/1.0)
    quantised = o.mul(scale)
    # normalise
    # quantised = torch.clip(quantised, min=qmin, max=qmax)
    quantised = quantised.type(type)
    quantised = quantised.type(torch.float)
    quantised = quantised.div(scale)
    #print(o.cpu().detach()[0,0,0], quantised.cpu().detach()[0,0,0])
    return quantised

for module in model.modules():
    module.register_forward_hook(quantise_forward_hook)

# test network
model.to(device)
model.eval()

losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()


with torch.no_grad():
    with tqdm(total=len(dataloader), desc='Inference', leave=True) as t:
        for batchIdx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data)
            losses.update(loss.item())
            top1.update(prec1.item())
            top5.update(prec5.item())

            t.set_postfix({
                'loss': losses.avg,
                'top1': top1.avg,
                'top5': top5.avg
            })
            t.update(1)
