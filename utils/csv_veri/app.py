from dash import Dash, html, dcc, callback, Output, Input, MATCH, ALL, ctx, State
from dash.exceptions import PreventUpdate
from dash_extensions import Keyboard
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import math
import safetensors.torch
import base64
import torch
import polars as pl
import numpy as np
import plotly.express as px
from safetensors import safe_open
import safetensors
import random
from zipfile import ZipFile
from torchvision.io import decode_image
import json

base_dir = os.path.expanduser('~/dataset/sampling/content_filtering/v2/')
pic_dir = os.path.expanduser('~/dataset/images/')
current_zip:ZipFile = None
current_zip_name: str = None

studies = [study for study in os.listdir(base_dir) if 'log' not in study ]
studies.sort()

study_divs = []
for study in studies:
    study_path = os.path.join(base_dir, study)
    plates = os.listdir(study_path)
    plates_divs = [html.Summary(study)]
    plates.sort(key=lambda x: int(x.split("_")[-1].split('-')[0][:-1]))
    plates_divs.extend([html.Div(children=plate, className='sidebar-child', id={"index": plate}, n_clicks=0) for plate in plates])
    study_divs.append(html.Details(plates_divs, className="sidebar-parent"))
    
studies = html.Div(study_divs, id="study_list")

app = Dash(__name__)

app.layout = html.Div([
    studies,
    html.Div([
        html.Div("CSV loaded", id='image-select'),
        html.Div([
            dcc.Graph(id="picture", figure={})
        ], id='figure'),
        html.Div(id="images")
    ], id='display')
        
], id='app_layout')

IMAGES: ZipFile = None
CSV_DF: pl.DataFrame = None
RANDOM_IMAGES = True
NUM_IMAGES = 400

@callback(
    Output("image-select", "children"),
    Output("images", "children"),
    Input({"index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def display_output(index):
    global IMAGES, CSV_DF, RANDOM_IMAGES, NUM_IMAGES
    csv_name:str = ctx.triggered_id['index']
    plate = csv_name.split('-')[0]
    
    csv_path = os.path.join(base_dir, plate, csv_name)
    image_path = os.path.join(pic_dir, plate, csv_name.replace('.csv', '.zip'))
    
    CSV_DF = pl.read_csv(csv_path, separator=',')    
    IMAGES = ZipFile(image_path)
    image_list = IMAGES.filelist
    
    if RANDOM_IMAGES:
        random.shuffle(image_list)
    else:
        image_list = sorted(image_list, key=lambda x: x.filename)
    
    image_list = image_list[:NUM_IMAGES]
    return csv_name, [html.Div(os.path.basename(pic.filename), id={"picture": pic.filename}) for pic in image_list]

@callback(
    Output("picture", "figure"),
    Input("image-select", "children"),
    Input({"picture": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def display_output(plate_name:str, index):
    if isinstance(ctx.triggered_id, str):
        return {}
    plate_name = plate_name.replace('.csv', '.zip')
    image_name:str = ctx.triggered_id['picture']
    
    image = CSV_DF.filter(
        pl.col('filename') == image_name,
        pl.col('plate') == plate_name
    )
 
    with IMAGES.open(image_name) as myfile:
        img_bytes = bytearray(myfile.read())
        torch_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
        image_tensor = decode_image(torch_buffer)
        fig_raw = px.imshow(image_tensor[0].numpy(), binary_string=True)
    
    if len(image) == 0:
        return fig_raw
    
    # active_tiles = image['tiles'][0].split(':')
    # tile_width, tile_height = image['tile_width'][0], image['tile_height'][0]
    
    # orig_h, orig_w = image_tensor.shape[1:] # Original height and width
    # image_height, image_width = orig_h, orig_w
    # while image_height % tile_height != 0: image_height -= 1
    # while image_width % tile_width   != 0: image_width  -= 1

    # tiles_per_row = image_width//tile_width
    
    # shapes = []
    
    # for tile in active_tiles:
    #     tiles_leftover = int(tile)%tiles_per_row
    #     num_tiles_tall = int(tile)//tiles_per_row
        
    #     y1 = num_tiles_tall * tile_height
    #     x1  = tiles_leftover * tile_width
    
    #     shapes.append(
    #         dict(
    #             type="rect",
    #             xref="x", yref="y",
    #             x0=x1, y0=y1,
    #             x1=x1+tile_width, y1=y1+tile_width,
    #             line=dict(
    #                 color="Red",
    #                 width=1.5,
    #             )
    #         )
    #     )
    # fig_raw.update_layout(shapes=shapes)
    
    return fig_raw
 
 
def get_all_tiles(tensor: torch.Tensor, tile_width: int, tile_height: int):
    """ Returns a tensor in format batch, tile_number, tile_height, tile_width
    tensor: A torch tensor in with shape: batch, width, height
    """
    orig_h, orig_w = tensor.shape[1:] # Original height and width
    image_height, image_width = orig_h, orig_w
    while image_height % tile_height != 0: image_height -= 1
    while image_width % tile_width   != 0: image_width  -= 1
    
    if orig_w != image_width or orig_h != image_height:
        tensor = tensor[:, :image_height, :image_width]
    
    tiles = []
    rows = tensor.split(tile_height, dim=1)
    for row in rows: tiles.extend(row.split(tile_width, dim=2))
    
    return torch.stack(tiles, dim=1)
 
if __name__ == '__main__':
    app.run_server(debug=True)
