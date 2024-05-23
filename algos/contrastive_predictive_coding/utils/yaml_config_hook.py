import os
import yaml


def yaml_config_hook(config_file, ex):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    ex.add_config(config_file)

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                cfg.update(yaml.safe_load(f))

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    ex.add_config(cfg)
    del f
    del cfg
