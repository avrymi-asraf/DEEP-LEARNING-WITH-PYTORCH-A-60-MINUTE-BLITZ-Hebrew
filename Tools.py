from IPython import display
import torch
import torch.nn.functional as F
from torch import nn  # for

import numpy as np  # for
import pandas as pd  # for


import plotly.graph_objects as go
import plotly.express as px
from plotly import subplots

from typing import Callable, List, Tuple
from torch import Tensor
from torch.nn import Module

TensorType = torch.Tensor
FigType = go.Figure


def plot_model(model, input_data, target, list_loss=None):
    """
    evaluate input by model and print it
    model:one linear model
    input_data: (*,1) tnesor
    target: (*,1) tnesor
    loss_function: loss function
    list_loss: list of prives loss to plot it
    """
    y = model(input_data)  # to plot it we need change output to regular tensor
    W, b = None, None
    if type(model) == nn.Linear:
        W = model.weight.data.tolist()  # get W and b as float
        b = model.bias.data.item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.7))
    ax1.scatter(input_data, target, label="Y")
    ax1.plot(input_data, y.detach(), label=f"{W=},{b=:.3f}")
    ax1.legend()

    if list_loss is not None:
        x_loss = range(len(list_loss))
        loss_label = 0 if not list_loss else list_loss[-1]
        ax2.plot(x_loss, list_loss, label=loss_label)
        ax2.legend()
    display.clear_output(wait=True)
    return fig


def plot_3d_model(model, input_data, target, list_loss=None):
    """
    evaluate input by model and print it
    model:one linear model
    input_data: (*,2) tnesor
    target: (*,1) tnesor
    loss_function: loss function
    list_loss: list of prives loss to plot it
    """
    x_grid, y_grid = torch.meshgrid(*input_data.T, indexing="xy")
    pairs = torch.cartesian_prod(*input_data.T)
    out = model(pairs).detach()
    z_grid = out.view(x_grid.shape)

    fig = plt.figure(figsize=(15, 7))
    ax1, ax2 = fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2)

    ax1.set_ylim(0, 15)
    ax1.set_xlim(0, 15)
    ax1.set_zlim(0, 15)

    ax1.scatter(*input_data.T, target, label="Y", color="green")
    ax1.plot_surface(x_grid, y_grid, z_grid)
    ax1.legend()

    if list_loss is not None:
        x_loss = range(len(list_loss))
        xlim = max(len(list_loss), 10)
        ylim = max(max(list_loss), 10)
        ax2.set_xlim(0, xlim)
        ax2.set_ylim(0, ylim)
        ax2.plot(x_loss, list_loss, label=f"{list_loss[-1]}")
        ax2.legend()
    return fig


def ploter(plot_func, model, x, target, op_list_loss=None):
    loss = [] if op_list_loss is None else op_list_loss

    def f():
        plot_func(model, x, target, loss)

    return f


from plotly import graph_objects as go
from plotly import subplots


def plot_model_plotly(model, input_data, target, list_loss=None):
    """
    evaluate input by model and print it
    model:one linear model
    input_data: (*,1) tnesor
    target: (*,1) tnesor
    loss_function: loss function
    list_loss: list of prives loss to plot it
    """
    y = model(input_data)  # to plot it we need change output to regular tensor
    W, b = None, None
    if type(model) == nn.Linear:
        W = model.weight.data.tolist()  # get W and b as float
        b = model.bias.data.item()

    fig = subplots.make_subplots(1, 2)
    fig.add_trace(
        go.Scatter(
            x=input_data.flatten(), y=target.flatten(), mode="markers", name="target"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Line(
            x=input_data.flatten(), y=y.detach().flatten(), name="model prediction"
        ),
        row=1,
        col=1,
    )

    if list_loss is not None:
        x_loss = list(range(len(list_loss)))
        loss_label = 0 if not list_loss else list_loss[-1]
        fig.add_trace(
            go.Line(x=x_loss, y=list_loss, name="loss", text=list_loss), row=1, col=2
        )
    # display.clear_output(wait=True)
    return fig.show()


def ask_user(ask):
    ask = f"{ask}\nPress Y to confirm: "
    return input(ask) in "Yy"


def ten_image_mnist(numbers: TensorType, clear_screen: bool = True) -> FigType:
    fig = subplots.make_subplots(
        row=3, col=3, shared_xaxes=True, shared_yaxes=True
    ).add_traces(
        data=[go.Heatmap(z=numbers[i]) for i in range(9)],
        cols=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        rows=[1, 2, 3, 1, 2, 3, 1, 2, 3],
    ).update_layout(yaxis = dict(scaleanchor = 'x'))
    fig.show()
    return fig
