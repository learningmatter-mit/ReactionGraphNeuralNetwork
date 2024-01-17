from rgnn.models.reaction_models.painn import PaiNN

MODEL_DICT = {"painn_reaction": PaiNN}


def get_model(params, **kwargs):
    """Create new model with the given parameters.

    Args:
            params (dict): parameters used to construct the model
            model_type (str): name of the model to be used

    Returns:
            model (nff.nn.models)
    """

    # check_parameters(PARAMS_TYPE[model_type], params)
    model_type = params["name"]
    del params["name"]
    model = MODEL_DICT[model_type](**params, **kwargs)

    return model
