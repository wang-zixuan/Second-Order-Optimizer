import jax
import jax.numpy as jnp
import flax.linen as nn
import torch
import torchvision
from jax.flatten_util import ravel_pytree
import wandb
import argparse
from functools import partial
from jax.config import config
config.update('jax_enable_x64', True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--eta', type=float, default=0.1)
parser.add_argument('-dp', '--damping', type=float)
parser.add_argument('-ni', '--niter', type=int)
parser.add_argument('-on', '--optimizer_name', type=str, default='sgd', choices=['sgd', 'newton', 'cg', 'gauss-newton'])
parser.add_argument('--seed',  type=int, default=42)

args = parser.parse_args()

torch.manual_seed(args.seed)

class Network(nn.Module):
    hidden: int = 10

    @nn.compact
    def __call__(
        self,
        x,
    ):
        x = x.reshape(x.shape[0], -1)
        return nn.Dense(10)(nn.relu(nn.Dense(self.hidden)(x)))

def apply_fn(flattened, img):
    return net.apply(pytree(flattened), img)

def hessian(loss_fn):
    return jax.jacfwd(jax.jacrev(loss_fn))

def hvp(loss_fn, x, v, damping=0.0, dtype=jnp.float64):
    res = jax.grad(lambda x: jnp.vdot(jax.grad(loss_fn)(x), v))(x) + damping * v
    return res.astype(dtype)

def gnhvp(loss_fn, apply_fn, flattened, v, damping=0.0):
    flattened = flattened.astype(v.dtype)
    z, R_z = jax.jvp(apply_fn, (flattened,), (v,))
    R_gz = hvp(loss_fn, z, R_z)
    _, f_vjp = jax.vjp(apply_fn, flattened)
    return f_vjp(R_gz)[0] + v * damping

def approx_solve(A_mvp, b, niter=None):
    # dim = b.size
    # A_linop = scipy.sparse.linalg.LinearOperator((dim,dim), matvec=A_mvp)
    b = b.astype(jnp.float64)
    res = jax.scipy.sparse.linalg.cg(A_mvp, b, maxiter=niter)
    return res[0]

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=100, shuffle=True)

def transform_fn(data):
    th_img, th_lab = data
    np_img = jnp.asarray(th_img)
    np_lab = jnp.asarray(th_lab)
    return np_img, np_lab

rng = jax.random.PRNGKey(args.seed)
net = Network()
dummy_img = transform_fn(next(iter(train_loader)))[0]
params = net.init(rng, dummy_img)
flattened, pytree = ravel_pytree(params)

def loss_fn(pred, lab):
    one_hot_lab = jax.nn.one_hot(lab, 10)
    lprobs = jax.nn.log_softmax(pred)
    nll = (lprobs * one_hot_lab).sum(-1).mean()
    return -nll

def fused_loss_fn(flattened, img, lab):
    params = pytree(flattened)
    pred = net.apply(params, img)
    return loss_fn(pred, lab)

@jax.jit
def eval_fn(flattened, img, lab):
    params = pytree(flattened)
    pred = net.apply(params, img)
    one_hot_lab = jax.nn.one_hot(lab, 10)
    lprobs = jax.nn.log_softmax(pred)
    nll = (lprobs * one_hot_lab).sum(-1).mean()
    pred_cls = pred.argmax(-1)
    acc = (pred_cls == lab).mean()
    return -nll, acc

@jax.jit
def newton_update_fn(flattened, img, lab):
    hessian_mat = hessian(fused_loss_fn)(flattened, img, lab)
    hessian_mat = jnp.linalg.pinv(hessian_mat + jnp.eye(*hessian_mat.shape) * args.damping)
    loss, grad_vec = jax.value_and_grad(fused_loss_fn)(flattened, img, lab)
    update = hessian_mat @ grad_vec
    return loss, flattened - args.eta * update

@jax.jit
def sgd_update_fn(flattened, img, lab):
    loss, grad_vec = jax.value_and_grad(fused_loss_fn)(flattened, img, lab)
    update = grad_vec
    return loss, flattened - args.eta * update

@partial(jax.jit, static_argnames=('damping', 'niter'))
def cg_update_fn(flattened, img, lab, damping, niter):
    loss, grad_vec = jax.value_and_grad(fused_loss_fn)(flattened, img, lab)
    _local_loss_fn = partial(fused_loss_fn, img=img, lab=lab)
    update = approx_solve(lambda v: hvp(_local_loss_fn, flattened, v, damping), grad_vec, niter)
    return loss, flattened - args.eta * update.astype(flattened.dtype)

@partial(jax.jit, static_argnames=('damping', 'niter'))
def gauss_newton_update_fn(flattened, img, lab, damping, niter):
    loss, grad_vec = jax.value_and_grad(fused_loss_fn)(flattened, img, lab)
    _local_loss_fn = partial(loss_fn, lab=lab)
    _local_apply_fn = partial(apply_fn, img=img)
    update = approx_solve(lambda v: gnhvp(_local_loss_fn, _local_apply_fn, flattened, v, damping), grad_vec, niter)
    return loss, flattened - args.eta * update.astype(flattened.dtype)

def evaluate(flattened):
    gloss = 0.0
    gacc =  0.0
    for idx, data in enumerate(test_loader):
        img, lab = transform_fn(data)
        loss, acc = eval_fn(flattened, img, lab)
        gloss += loss
        gacc += acc

    return gloss, gacc


wandb.init(project='optimizer', config=args)

if args.optimizer_name == "newton":
    update_fn = newton_update_fn
elif args.optimizer_name == "sgd":
    update_fn = sgd_update_fn
elif args.optimizer_name == "cg":
    update_fn = partial(cg_update_fn, damping=args.damping, niter=args.niter)
elif args.optimizer_name == "gauss-newton":
    update_fn = partial(gauss_newton_update_fn, damping=args.damping, niter=args.niter)

else:
    raise ValueError(args.optimizer_name)

for epoch in range(10):
    loss = 0
    for idx, data in enumerate(train_loader):
        img, lab = transform_fn(data)
        cur_loss, flattened = update_fn(flattened, img, lab)
        loss += cur_loss
        wandb.log({'train/loss': cur_loss})
        # print(cur_loss)
    print(f'Epoch Average Training Loss = {loss / len(train_loader)}')
    wandb.log({'train/epoch_loss': loss / len(train_loader)})
    loss, acc = evaluate(flattened)
    print(f'Epoch Average Eval Loss = {loss / len(test_loader)}')
    print(f'Epoch Average Eval Acc = {acc / len(test_loader)}')
    wandb.log({'eval/loss': loss / len(test_loader), 
               'eval/acc': acc / len(test_loader)})
