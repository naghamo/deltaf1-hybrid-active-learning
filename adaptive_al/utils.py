import json
import inspect

def save_model_hyperparams_json(model, save_path: str, root_key: str = "params"):
    sig = inspect.signature(model.__init__)
    params = {}

    for arg in sig.parameters:
        if arg == "self":
            continue
        if hasattr(model, arg):
            value = getattr(model, arg)
            try:
                json.dumps(value)  # test if it's serializable
                params[arg] = value
            except TypeError:
                print(f"Skipping non-serializable param: {arg}={value}")

    wrapped = {root_key: params}

    with open(save_path, "w") as f:
        json.dump(wrapped, f, indent=2)

    print(f"Saved model params under '{root_key}' to {save_path}")
