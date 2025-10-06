import importlib
import pkgutil


# Dynamically import all modules in this directory
# This allows the runner to discover all evaluator classes automatically.


def get_evaluator_class(tool_name: str):
    """
    Dynamically finds and returns the evaluator class for a given tool name.
    Handles tool names with hyphens (e.g., 'otalign-progressive').
    - 'otalign-progressive' -> module 'otalign_progressive_evaluator', class 'OTAlignProgressiveEvaluator'
    - 'otalign' -> module 'otalign_evaluator', class 'OtalignEvaluator'
    """
    # Sanitize tool_name for module import (e.g., 'otalign-progressive' -> 'otalign_progressive')
    sanitized_tool_name = tool_name.lower().replace("-", "_")
    module_name = f"{sanitized_tool_name}_evaluator"

    # Construct the class name (e.g., 'otalign-progressive' -> 'OTAlignProgressiveEvaluator')
    class_name_parts = [part.capitalize() for part in tool_name.split("-")]
    class_name = "".join(class_name_parts) + "Evaluator"

    try:
        # The package is 'benchmark.modules'
        module = importlib.import_module(f".{module_name}", package="benchmark.modules")
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not find or import evaluator for '{tool_name}'. Ensure a module named '{module_name}.py' with a class '{class_name}' exists in the 'benchmark/modules' directory."
        ) from e


# Import all modules in the current package to register the evaluators
for _, module_name, _ in pkgutil.walk_packages(__path__, f"{__name__}."):
    importlib.import_module(module_name)
