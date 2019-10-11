import torch

def batch_jacobian(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


def parametriser_regulariser(x, g, theta, h):
    batch_size = x.shape[0]
    grad_f = torch.zeros(batch_size, 10, 1, 28, 28)
    for i in range(10):
        grad_f[:, i] = torch.autograd.grad(g[:, i].sum(), x, retain_graph=True)[0].data
    # grad_f = torch.autograd.grad(g, x)
    jacob_h = batch_jacobian(h, x)
    # theta_times_jacob_h = torch.einsum('boi,bo->bi', jacob_h, theta)
    theta_times_jacob_h = torch.einsum('ijiklm,ij->ijklm', jacob_h, theta)
    reg = torch.sum((grad_f.cuda() - theta_times_jacob_h.cuda()) ** 2)
    return reg

