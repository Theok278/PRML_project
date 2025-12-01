import numpy as np
from rnn import RNN
from lstm import LSTM

def save_rnn_model(model, path):
    saving = {
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "output_size": model.output_size,
        "num_layers": model.num_layers,
        "lr": model.lr,
        "adam": model.adam,
        "Why": model.Why,
        "by": model.by,
    }

    # Save weights and biases of all layers
    for l in range(model.num_layers):
        saving[f"Wxh_{l}"] = model.Wxh[l]
        saving[f"Whh_{l}"] = model.Whh[l]
        saving[f"bh_{l}"] = model.bh[l]

    if model.adam:
        saving.update({
            "beta1": model.beta1,
            "beta2": model.beta2,
            "eps": model.eps,
            "t": model.t,
        })
        for l in range(model.num_layers):
            saving[f"m_Wxh_{l}"] = model.m_Wxh[l]
            saving[f"v_Wxh_{l}"] = model.v_Wxh[l]
            saving[f"m_Whh_{l}"] = model.m_Whh[l]
            saving[f"v_Whh_{l}"] = model.v_Whh[l]
            saving[f"m_bh_{l}"] = model.m_bh[l]
            saving[f"v_bh_{l}"] = model.v_bh[l]

        saving["m_Why"] = model.m_Why
        saving["v_Why"] = model.v_Why
        saving["m_by"] = model.m_by
        saving["v_by"] = model.v_by

    np.savez(path, **saving)
    print(f"Model saved to: {path}")

def load_rnn_model(path, adam=False):
    data = np.load(path)
    num_layers = int(data["num_layers"])
    model = RNN(
        input_size=int(data["input_size"]),
        hidden_size=int(data["hidden_size"]),
        output_size=int(data["output_size"]),
        lr=float(data["lr"]),
        adam=adam,
        num_layers=num_layers
    )

    model.Why = data["Why"]
    model.by = data["by"]

    # Load weights and biases for each layer
    model.Wxh = [data[f"Wxh_{l}"] for l in range(num_layers)]
    model.Whh = [data[f"Whh_{l}"] for l in range(num_layers)]
    model.bh  = [data[f"bh_{l}"] for l in range(num_layers)]

    if adam:
        model.beta1 = float(data["beta1"])
        model.beta2 = float(data["beta2"])
        model.eps = float(data["eps"])
        model.t = int(data["t"])

        model.m_Wxh = [data[f"m_Wxh_{l}"] for l in range(num_layers)]
        model.v_Wxh = [data[f"v_Wxh_{l}"] for l in range(num_layers)]
        model.m_Whh = [data[f"m_Whh_{l}"] for l in range(num_layers)]
        model.v_Whh = [data[f"v_Whh_{l}"] for l in range(num_layers)]
        model.m_bh = [data[f"m_bh_{l}"] for l in range(num_layers)]
        model.v_bh = [data[f"v_bh_{l}"] for l in range(num_layers)]

        model.m_Why = data["m_Why"]
        model.v_Why = data["v_Why"]
        model.m_by = data["m_by"]
        model.v_by = data["v_by"]

    print(f"Model loaded from: {path}")
    return model

def save_lstm_model(model, path):
    saving = {
        "Wy": model.Wy,
        "by": model.by,
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "output_size": model.output_size,
        "num_layers": model.num_layers
    }

    # Save weights and biases of all layers
    for l in range(model.num_layers):
        saving[f"Wf_{l}"] = model.Wf[l]
        saving[f"Wi_{l}"] = model.Wi[l]
        saving[f"Wo_{l}"] = model.Wo[l]
        saving[f"Wc_{l}"] = model.Wc[l]
        saving[f"bf_{l}"] = model.bf[l]
        saving[f"bi_{l}"] = model.bi[l]
        saving[f"bo_{l}"] = model.bo[l]
        saving[f"bc_{l}"] = model.bc[l]

    np.savez(path, **saving)

def load_lstm_model(path, adam):
    data = np.load(path)
    num_layers = int(data["num_layers"])
    model = LSTM(
        input_size=int(data["input_size"]),
        hidden_size=int(data["hidden_size"]),
        output_size=int(data["output_size"]),
        adam=adam,
        num_layers=num_layers
    )

    # Load weights and biases for each layer
    for l in range(num_layers):
        model.Wf[l] = data[f"Wf_{l}"]
        model.Wi[l] = data[f"Wi_{l}"]
        model.Wo[l] = data[f"Wo_{l}"]
        model.Wc[l] = data[f"Wc_{l}"]
        model.bf[l] = data[f"bf_{l}"]
        model.bi[l] = data[f"bi_{l}"]
        model.bo[l] = data[f"bo_{l}"]
        model.bc[l] = data[f"bc_{l}"]

    model.Wy = data["Wy"]
    model.by = data["by"]

    return model

