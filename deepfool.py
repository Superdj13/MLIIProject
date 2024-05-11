import numpy as np
from torch.autograd import Variable
import torch as torch
import copy

def zero_gradients(p):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        p.grad.detach_()
        p.grad.zero_()


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size 1xHxW
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    net.eval()
    is_cuda = torch.cuda.is_available()


    f_image = net.forward(Variable(image, requires_grad=True)).data.flatten()
    I = f_image.argsort().flip(0)

    I = I[0:num_classes]
    label = I[0]

    input_shape = image[:].shape
    if is_cuda:
        pert_image = copy.deepcopy(image).cuda()
        w = torch.zeros(input_shape).cuda()
        r_tot = torch.zeros(input_shape).cuda()
    else:
        pert_image = copy.deepcopy(image)
        w = torch.zeros(input_shape)
        r_tot = torch.zeros(input_shape)
    

    loop_i = 0

    x = Variable(pert_image[:], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.clone()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.clone()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = abs(f_k)/torch.linalg.norm(w_k)

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / torch.linalg.norm(w)
        r_tot = (r_tot + r_i).type(torch.float32)

        if is_cuda:
            
            pert_image = image + (1+overshoot)*r_tot.cuda()
        else:
            pert_image = image + (1+overshoot)*r_tot

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = fs.data.flatten().argmax()

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image