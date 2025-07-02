from dash import Dash, html, dcc, callback, Output, Input, MATCH, ALL, ctx, State
from dash.exceptions import PreventUpdate
from dash_extensions import Keyboard
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import math
import safetensors.numpy
import safetensors.torch
import base64
import torch
import numpy as np
import plotly.express as px
from safetensors import safe_open
import safetensors
import random
from zipfile import ZipFile
from torchvision.io import decode_image
import json

base_dir = os.path.expanduser('~/jpeters/histograms/')
pic_dir = os.path.expanduser('~/dataset/images/')
current_zip:ZipFile = None
current_zip_name: str = None

if not os.path.exists('./thresholds.json'):
    with open('./thresholds.json', 'w') as f:
        f.write("{}")
    thresholds = {}
else:
    with open('./thresholds.json', 'r') as f:
        thresholds = json.load(f)

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
    
app = Dash(__name__)

app.layout = html.Div([
    html.Div(
        children=[
            html.Div(children=study_divs, className='app-sidebar'),
            html.Div(
                children=[
                    html.Div(
                        children=[
                                html.Div(id="dataset-plate"),
                                html.Div(id="image-select"),
                                html.Div(id="graph-wrapper", children=[
                                        dcc.Graph(id="heatmap", figure={}),
                                        dcc.Graph(id="raw-image", figure={})    
                                ]),        
                            ]),
                    html.Div(children = [
                        html.Div(children=[
                            html.Span("Start: "),
                            dcc.Input(value=0, id="start-value", type="number", debounce=True),
                            html.Span("End:"),
                            dcc.Input(value=100, id="end-value", type="number", debounce=True),
                            html.Span("Randomize:"),
                            dcc.Checklist([""],[], id="randomize"),
                            html.Span("K:"),
                            dcc.Input(value=10, id="top-k", type="number", debounce=True),
                            html.Span("Type:"),
                            dbc.Select(options=[
                                {'label': 'Top', 'value': 'top'},
                                {'label': 'Bottom', 'value': 'bottom'}
                            ], id='k-type', value="top"),
                            html.Span("Percentage:"),
                            dcc.Checklist([""],[""], id="top-k-percent"),
                            ], className="range-wrapper"),
                        html.Div(children=[
                            html.Span("Save Plate Thresh: "),
                            html.Button("Save", id="save-plate-thresh", className="save"),
                            html.Span("Save Study Thresh: "),
                            html.Button("Save", id="save-study-thresh", className="save"),
                            html.Span("Save Threshs to file: "),
                            html.Button("Save", id="save-thresh-file", className="save"),
                            ], className="range-wrapper"),
                        html.Div(children=[
                            html.Span("PC:"),
                            dcc.Checklist([""],[], id="is_pc_img"),
                            html.Span("Live:"),
                            dcc.Checklist([""],[], id="render-live"),
                            ], className="range-wrapper"),
                        html.Div(id='image-dropdown')    
                    ], className="dropdown-wrapper"),
                    html.Button(n_clicks=0, hidden=True, id="draw-boxes-auto"),
                    Keyboard(captureKeys=['l', 'k'], id="next_image")
                ],
                className="graph-container"
            ),
        ],
        className="app-layout"
        ),
])

@callback(
    Output("dataset-plate", "children", allow_duplicate=True),
    Output("image-dropdown", 'children'),
    Input({"index": ALL}, "n_clicks"),
    Input("start-value", "value"),
    Input('end-value', "value"),
    Input('randomize', 'value'),
    State('dataset-plate', 'children'),
    prevent_initial_call = True
)
def display_output(_, start:int, stop:int, randomize:list, dataset_plate):
    if "index" in ctx.triggered_id:    
        plate_filename = ctx.triggered_id['index']
    else:
        if dataset_plate is None:
            return None, "empty dataset"
        else:
            plate_filename = dataset_plate
            
    files = load_plate(plate_filename, randomize, start, stop)    
    return plate_filename, files

def load_plate(plate_filename, randomize, start, stop):
    global current_zip_name
    global current_zip
    study = plate_filename.split('-')[0]
    file_path = os.path.join(base_dir, study, plate_filename)  
    
    with safe_open(file_path, framework="numpy", device="cpu") as f:
        image_names = f.keys()
    
    if not isinstance(start, int) or not isinstance(stop, int) or start < 0 or stop < 0 or start > stop:
        start = 0
        stop = 100
    
    if len(randomize) > 0:
        random.shuffle(image_names)
    
    image_names_to_display = list(image_names)[start:stop]
    
    study = plate_filename.split('-')[0]
    file_path = os.path.join(pic_dir, study, plate_filename)

    if current_zip_name != file_path:
        if current_zip is not None:
            current_zip.close()
        zip_path = file_path.replace("safetensors", "zip")
        current_zip = ZipFile(zip_path)
        current_zip_name = file_path

    if current_zip_name is None:
        current_zip_name = file_path
    
    return [html.Div(pic.split('/')[1], id={"picture": pic}) for pic in image_names_to_display]

@callback(
    Output("image-select", "children"),
    Output("heatmap", "figure"),
    Output("raw-image", "figure", allow_duplicate=True),
    Output("draw-boxes-auto", "n_clicks"),
    Input("dataset-plate", "children"),
    Input('is_pc_img', 'value'),
    Input('render-live', 'value'),
    Input("image-select", "children"),
    State("image-dropdown", 'children'),
    Input({"picture": ALL}, "n_clicks"),
    prevent_initial_call='initial_duplicate'
)
def display_output(dataset_plate, is_pc, render_live, image_select, image_list, _):  
    global current_zip   
    if ctx.triggered_id is None:
        return "", {}, {}, 0
    
    tile_height = 32
    tile_width = 32
    
    if ctx.triggered_id == 'dataset-plate':
        image = image_list[0]['props']['id']['picture']
    elif isinstance(ctx.triggered_id, dict):
        image = ctx.triggered_id['picture']
    else:
        image = image_select

    study = dataset_plate.split('-')[0]
    
    image_tensor_name = image.split('/')[-1]
    
    if image_tensor_name not in current_zip.namelist():
        image = image_list[0]['props']['id']['picture']
    
    with current_zip.open(image_tensor_name) as myfile:
        img_bytes = bytearray(myfile.read())
        torch_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
        image_tensor = decode_image(torch_buffer)[0]
        fig_raw = px.imshow(image_tensor.numpy(), binary_string=True)
    
    if len(is_pc) == 0 and len(render_live) == 0:
        file_path = os.path.join(base_dir, study, dataset_plate)
        with safe_open(file_path, framework="numpy", device="cpu") as f:
            tensor:np.ndarray = f.get_tensor(image)
    else:    
        print(tile_width, tile_height, image_tensor.shape[1], image_tensor.shape[0], image)
        temp_tile_width, temp_tile_height, img_width, img_height = get_tile_image_dims(tile_width, 
                                                                                       tile_height, 
                                                                                       image_tensor.shape[1], 
                                                                                       image_tensor.shape[0])
        
        if tuple(image_tensor.shape) != (img_height, img_width):
            image_tensor = image_tensor[:img_height,:img_width]
        tiles = get_all_tiles(image_tensor.unsqueeze(0), temp_tile_width, temp_tile_height)[0]
        if len(render_live) > 0:
            diffs = tiles.to(torch.float32).mean(dim=(1, 2))
        else:
            tile_maxes = tiles.amax(dim=(1,2))
            tile_mins = tiles.amin(dim=(1,2))
            diffs = tile_maxes - tile_mins

        num_tiles_wide = image_tensor.shape[1] // temp_tile_width
        num_tiles_tall = image_tensor.shape[0] // temp_tile_height
        print(image, img_height, img_width, num_tiles_tall, num_tiles_wide)

        tensor = diffs.reshape((num_tiles_tall, num_tiles_wide)).numpy()
        
    fig_heat = px.imshow(tensor)
    
    return image, fig_heat, fig_raw, 0#, class_names

def get_img_idx(image_path, images):
    split_img = image_path.split('/')
    file_name = split_img[1]
    try:
        current_index = images.index(file_name)
    except:
        return 0
    return current_index

def get_tile_image_dims(tile_width, tile_height, image_width, image_height):
    max_height = 45
    max_width = 45
    temp_tile_height = tile_height
    temp_tile_width = tile_width
    
    while True:
        temp_tile_height = tile_height
        temp_tile_width = tile_width
    
        if image_height % temp_tile_height != 0 or image_width % temp_tile_width != 0:
            while image_height % temp_tile_height != 0 and temp_tile_height < max_height:
                temp_tile_height += 1
            while image_width % temp_tile_width != 0 and temp_tile_width < max_width:
                temp_tile_width += 1
                
        if image_height % temp_tile_height != 0:
            image_height -= 1
        if image_width % temp_tile_width != 0:
            image_width -= 1
    
        if image_width % temp_tile_width == 0 and image_height % temp_tile_height == 0:
            break
        
    return temp_tile_width, temp_tile_height, image_width, image_height

@callback(
    Output("raw-image", "figure", allow_duplicate=True),
    Input("top-k", "value"),
    Input("top-k-percent", "value"),
    Input('k-type', "value"),
    State("heatmap", "figure"),
    State("raw-image", "figure"),
    State("dataset-plate", "children"),
    Input("draw-boxes-auto", "n_clicks"),
    prevent_initial_call='initial_duplicate'
)
def draw_boxes(top_k, top_k_percent, k_type, heatmap, raw_image, dataset_plate, _):
    if ctx.triggered_id is None or heatmap == {} or raw_image == {}:
        return raw_image

    heatmap_data = heatmap['data'][0]['z']
    raw_data = raw_image['data'][0]['source'].replace("data:image/png;base64,", "")
    img_bytes = bytearray(base64.b64decode(raw_data))
    torch_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
    image_tensor = decode_image(torch_buffer)[0]
            
    tiled_width = len(heatmap_data[0])
    tiled_height = len(heatmap_data)
    
    img_width = image_tensor.shape[1]
    img_height = image_tensor.shape[0]

    tile_width = img_width/tiled_width
    tile_height = img_height/tiled_height    
    
    values_with_coords = []
    for i, row in enumerate(heatmap_data):
        for j, tile_val in enumerate(row):
            values_with_coords.append((i, j, tile_val))

    reverse = True
    if k_type == "bottom":    
        reverse = False
        
    values_with_coords.sort(key=lambda x: x[-1], reverse=reverse)
        
    if len(top_k_percent) > 0:
        top_k = math.ceil(tiled_height * tiled_width * (top_k / 100))

    top_coordinates = values_with_coords[:top_k]
            
    fig = go.Figure(raw_image)
    shapes = []
    for i, j, _ in top_coordinates:
        raw_y = i * tile_height
        raw_x = j * tile_width
        
        shapes.append(
            dict(
                type="rect",
                xref="x", yref="y",
                x0=raw_x, y0=raw_y,
                x1=raw_x+tile_width, y1=raw_y+tile_height,
                line=dict(
                    color="Red",
                    width=1.5,
                )
            )
        )
    fig.update_layout(shapes=shapes)
    
    return fig

@callback(
    Input("save-plate-thresh", 'n_clicks'),
    Input("save-study-thresh", 'n_clicks'),
    State("dataset-plate", 'children'),
    State('top-k-percent', 'value'),
    State('k-type', 'value'),
    State('top-k', 'value'),
    prevent_initial_call=True
)
def save_plate_thresh(_1, _2, plate, is_percent, k_type, top_k):
    study = plate.split('-')[0]
    is_percent = len(is_percent) > 0
    plate = plate.split('.')[0]
    
    if 'study' in ctx.triggered_id:
        thresholds[study] = {}
        thresholds[study]['is_percent'] = is_percent
        thresholds[study]['k_type'] = k_type
        thresholds[study]['k'] = top_k
    else:
        if study not in thresholds:
            thresholds[study] = {} 
        thresholds[study][plate] = {}    
        thresholds[study][plate]['is_percent'] = is_percent
        thresholds[study][plate]['k_type'] = k_type
        thresholds[study][plate]['k'] = top_k
        
@callback(
    Input("save-thresh-file", "n_clicks")
)
def save_to_file(_):
    with open('./thresholds.json', 'w') as f:
        f.write(json.dumps(thresholds))
 
@callback(
    Output("image-select", 'children', allow_duplicate=True),
    Input("next_image", "n_keydowns"),
    Input("next_image", "keydown"),
    State({'picture': ALL}, 'children'),
    State('image-select', 'children'),
    prevent_initial_call=True
)
def next_image(_, key, images:list, current_image):
    if current_image == '':
        raise PreventUpdate
    
    key = key['key']
    offset = 1 if key == "l" else -1
    
    split_img = current_image.split('/')
    zip_file = split_img[0]
    file_name = split_img[1]
    
    current_index = images.index(file_name)
    new_index = current_index + offset
    if new_index >= len(images):
        new_index = len(images)-1
    elif new_index < 0:
        new_index = 0
    
    return zip_file + '/' + images[new_index]
 
def can_tile_tensor(tensor: torch.Tensor, tile_width: int, tile_height: int):
    tensor_shape = tensor[0].shape
    assert tensor_shape[1] % tile_width == 0, "Tile width is not divisible by image width"
    assert tensor_shape[0] % tile_height == 0, "Tile height is not divisible by image width"

def get_all_tiles(tensor: torch.Tensor, tile_width: int, tile_height: int):
    """ Returns a tensor in format batch, tile_number, tile_height, tile_width
    tensor: A torch tensor in with shape: batch, width, height
    """
    can_tile_tensor(tensor, tile_width, tile_height)
    
    tiles = []
    rows = tensor.split(tile_height, dim=1)
    for row in rows: tiles.extend(row.split(tile_width, dim=2))
    
    return torch.stack(tiles, dim=1)

if __name__ == '__main__':
    app.run_server(debug=True)
