import torch
import torch.nn as nn
import schnetpack_nobias as spk

def small_target_model_kwargs():
    return {
        "representation": dict(
            n_interactions=1,
            n_atom_basis=16,
            n_filters=16,
            n_gaussians=16,
        ),
        "output_modules": dict(
            n_neurons=[16],
            n_out=8,
            n_layers=2,
        )
    }

def medium_target_model_kwargs():
    return {
        "representation": dict(
            n_interactions=2,
            n_atom_basis=32,
            n_filters=32,
            n_gaussians=32,
        ),
        "output_modules": dict(
            n_neurons=[32],
            n_out=16,
            n_layers=2,
        )
    }

def medium_ext_target_model_kwargs():
    return {
        "representation": dict(
            n_interactions=2,
            n_atom_basis=32,
            n_filters=32,
            n_gaussians=32,
        ),
        "output_modules": dict(
            n_neurons=[32],
            n_out=128,
            n_layers=2,
        )
    }

def large_target_model_kwargs():
    return {
        "representation": dict(
            n_interactions=3,
            n_atom_basis=64,
            n_filters=64,
            n_gaussians=64,
        ),
        "output_modules": dict(
            n_neurons=[64, 32],
            n_out=32,
            n_layers=3,
        )
    }


def large_ext_target_model_kwargs():
    return {
        "representation": dict(
            n_interactions=3,
            n_atom_basis=64,
            n_filters=64,
            n_gaussians=64,
        ),
        "output_modules": dict(
            n_neurons=[64, 64],
            n_out=128,
            n_layers=3,
        )
    }

def get_target_model(model_type, rcut):
    
    if model_type == "small":
        kwargs = small_target_model_kwargs()
    elif model_type == "medium":
        kwargs = medium_target_model_kwargs()
    elif model_type == "medium-ext":
        kwargs = medium_ext_target_model_kwargs()
    elif model_type == "large":
        kwargs = large_target_model_kwargs()
    elif model_type == "large-ext":
        kwargs = large_ext_target_model_kwargs()
    else:
        raise ValueError(f"Unknown model type '{model_type}'")

    representation = spk.SchNet(        
        cutoff=rcut,
        **kwargs["representation"]
    )

    add_kwargs = {
            "property": "energy",
            "contributions": "target"
    }

    # if len(elements) > 1:
    #     kwargs["output_modules"]["n_hidden"] = kwargs["output_modules"].pop("n_neurons")
    #     output_modules = [
    #         spk.atomistic.ElementalAtomwise(
    #             n_in=representation.n_atom_basis,
    #             elements=frozenset(elements),
    #             **kwargs["output_modules"],
    #             **add_kwargs,
    #         )
    #         for i in range(len(elements))]
    # else:
    output_modules = [
        spk.atomistic.Atomwise(
            n_in=representation.n_atom_basis,
            **kwargs["output_modules"],
            **add_kwargs,
        )
    ]

    model = spk.atomistic.model.AtomisticModel(representation, output_modules)
    return model


def build_kld_nov_score():

    softmax = nn.Softmax(dim=-1)
    log_softmax = nn.LogSoftmax(dim=-1)

    def kld_novelty_score(target, pred):
        return torch.sum(softmax(target) * (log_softmax(target)-log_softmax(pred)), dim=-1)
    
    return kld_novelty_score


def build_kld_loss(properties, loss_tradeoff=None):
    """
    Build the Kulback-Leibler-Divergence loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        Kulback-Leibler-Divergence loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise ValueError("loss_tradeoff must have same length as properties!")

    kld = nn.KLDivLoss(reduction='batchmean', log_target=False)
    softmax = nn.Softmax(dim=-1)
    raise ValueError("Check implementation first") # TODO see cross-entropy loss
    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            loss += kld(result[prop], batch[prop])*factor
        return loss

    return loss_fn


def build_ce_loss(properties, loss_tradeoff=None):
    """
    Build the Cross-Entropy loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        Kulback-Leibler-Divergence loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise ValueError("loss_tradeoff must have same length as properties!")

    softmax = torch.nn.Softmax(dim=-1)
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            loss -= factor*torch.sum(softmax(batch[prop])*log_softmax(result[prop])) / batch[prop].size()[0]
        return loss

    return loss_fn


def build_cossim_loss(properties, loss_tradeoff=None):
    """
    Build the Cosine-Similarity loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        Cosine-Similarity loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise ValueError("loss_tradeoff must have same length as properties!")

    crit = nn.CosineSimilarity(dim=2)

    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            cossim = crit(batch[prop], result[prop])
            loss = -(torch.mean(torch.sum(cossim-1, dim=1)))*factor
        return loss

    return loss_fn

    crit = torch.nn.CosineSimilarity(dim=2)
    novelty_func = lambda target, pred: -(crit(target, pred)-1) 
