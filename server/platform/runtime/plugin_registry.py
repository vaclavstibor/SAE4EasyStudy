from importlib import import_module

from .plugin_contract import StudyPluginContract

CANONICAL_PLUGIN_MODULES = [
    "server.plugins.steering",
    "server.plugins.fastcompare",
    "server.plugins.empty_template",
]


def load_plugin_contract(module_path: str) -> StudyPluginContract:
    module = import_module(module_path)
    if hasattr(module, "get_plugin"):
        plugin = module.get_plugin()
    elif hasattr(module, "PLUGIN"):
        plugin = module.PLUGIN
    else:
        raise RuntimeError(f"Module '{module_path}' does not expose a plugin contract.")
    if not isinstance(plugin, StudyPluginContract):
        raise TypeError(
            f"Module '{module_path}' returned unsupported plugin contract: {type(plugin)!r}"
        )
    return plugin


def load_canonical_plugin_contracts() -> list[StudyPluginContract]:
    return [load_plugin_contract(module_path) for module_path in CANONICAL_PLUGIN_MODULES]
