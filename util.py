from os import path
import json
from esm import pretrained
from argparse import Namespace
from typing import Tuple

from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.model.esm2 import ESM2
from esm.data import Alphabet

SAVE_DIR = path.join(path.dirname(path.abspath(__file__)), "config")


def generate_alphabet_args(model="esm2"):
    """
    Generate args dict for ESM2 Alphabet from Hub.

    Args:
        model (str, optional): Model name: esm2/esm_if. Defaults to "esm2".
        save (bool, optional): Save args to json file. Defaults to True.
    """
    assert model in ["esm2", "esm_if"], "Model Name must be esm2 or esm_if"
    if model == "esm2":
        _, alphabet = pretrained.esm2_t6_8M_UR50D()
    else:
        _, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50()
    args = {
        "standard_toks": alphabet.standard_toks,
        "prepend_toks": alphabet.prepend_toks,
        "append_toks": alphabet.append_toks,
        "prepend_bos": alphabet.prepend_bos,
        "append_eos": alphabet.append_eos,
        "use_msa": alphabet.use_msa,
    }

    return args
    # with open(path.join(SAVE_DIR, "alphabet_args.json"), "w") as f:
    #     json.dump(args, f, indent=4)


def generate_esm_if_args():
    model, _ = pretrained.esm_if1_gvp4_t16_142M_UR50()
    model_args = vars(model.args)
    return model_args
    with open(path.join(SAVE_DIR, "esm_if_base_args.json"), "w") as f:
        json.dump(model_args, f, indent=4)


def load_esm_2(model_config="base_8M") -> Tuple[ESM2, Alphabet]:
    """Load ESM2 model using saved args

    Args:
        model_config (str | json): Model Type - base_8M/base_650M. Defaults to "base_8M".

    Returns:
        Tuple[ESM2, Alphabet]: ESM2 model and Alphabet
    """
    # Load ESM-2 Base model args, alphabet args, and the given model config

    # If model_config is a string, load the default model config
    if isinstance(model_config, str):
        with open(path.join(SAVE_DIR, "default_model_configs.json"), "r") as f:
            esm2_config = json.load(f)["esm2"]
            assert (
                model_config in esm2_config.keys()
            ), f"Model Type {model_config} not found in default_model_configs.json"
            esm2_config = esm2_config[model_config]

    elif isinstance(model_config, dict):
        esm2_config = model_config
    
    else:
        raise ValueError(
            f"model_config must be a string or dict, not {type(model_config)}"
        )
    
    with open(path.join(SAVE_DIR, "default_alphabet_args.json"), "r") as f:
        alphabet_args = json.load(f)["esm2"]

    alphabet = Alphabet(**alphabet_args)
    esm2_config["alphabet"] = alphabet
    esm2 = ESM2(**esm2_config)

    return esm2, alphabet


def load_esm_if(
    model_config="base_7M",
) -> Tuple[GVPTransformerModel, Alphabet]:
    """Load ESM-IF model using saved args

    Args:
        model_config (str | json): Model Type - base_7M/base_142M. Defaults to "base_7M".

    Returns:
        Tuple[GVPTransformerModel, Alphabet]: ESM-IF model and Alphabet
    """
    # Load ESM-IF Base model args, alphabet args, and the given model config
    if isinstance(model_config, str):
        with open(path.join(SAVE_DIR, "default_model_configs.json"), "r") as f:
            all_configs = json.load(f)["esm_if"]
            assert (
                model_config in all_configs.keys()
            ), f"Model Type {model_config} not found in default_model_configs.json"
            model_config = all_configs[model_config]

    elif not isinstance(model_config, dict):
        raise ValueError(
            f"model_config must be a string or dict, not {type(model_config)}"
        )
    
    with open(path.join(SAVE_DIR, "default_alphabet_args.json"), "r") as f:
        alphabet_args = json.load(f)["esm_if"]

    with open(path.join(SAVE_DIR, "default_esm_if_args.json"), "r") as f:
        esm_if_args = json.load(f)

    # Update model args with the given model type args
    for arg, val in model_config.items():
        esm_if_args[arg] = val

    alphabet = Alphabet(**alphabet_args)
    esm_if_args = Namespace(**esm_if_args)
    esm_if = GVPTransformerModel(
        esm_if_args,
        alphabet,
    )

    # delete decoder TODO: do this from args
    del esm_if.decoder

    return esm_if, alphabet
