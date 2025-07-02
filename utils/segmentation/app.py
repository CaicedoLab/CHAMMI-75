from dash import Dash, html, dcc, callback, Output, Input, MATCH, ALL, ctx, State
import plotly.express as px
import os
import scipy.ndimage
from skimage.filters.rank import modal
from skimage.morphology import footprint_rectangle
import torch
import polars as pl
import numpy as np
import plotly.express as px
from torchvision.io import decode_image
import yaml
from cellpose import models
import cv2
import skimage
import scipy
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_table_cell_list_len(-1)
pl.Config.set_fmt_str_lengths(1000)


OVERRIDES = {'experiment.well':pl.String, 
             'experiment.plate':pl.String, 
             'microscopy.fov': pl.String,  
             'microscopy.magnification': pl.String, 
             'geometry.depth': pl.String,
             'geometry.z_slice': pl.String
             }

device = torch.device(f'cuda:3' if torch.cuda.is_available() else 'cpu')
cellpose_model = models.Cellpose(model_type='cyto3', device=device)

ds10  = ["hpa0001", "idr0002", "idr0008", "idr0086", "idr0088", "idr0089", "jump0001", "nidr0031", "nidr0032", "wtc0001"]
imgs_base = os.path.expanduser('~/dataset/sampling/')
meta_path = os.path.expanduser("/scr/vidit/metadata/fixed/75ds_small_meta_fixes.csv")  
meta = pl.read_csv(meta_path, schema_overrides=OVERRIDES)
# meta = meta.filter(pl.col('experiment.study').is_in(ds10))
unique_chans = meta.group_by(pl.col('experiment.study'), pl.col('imaging.multi_channel_id')).agg(uniq_cols=pl.col('imaging.channel_type').sort()).unique(['uniq_cols', 'experiment.study']).drop('imaging.multi_channel_id')
study_groups = unique_chans.sort('experiment.study').group_by('experiment.study', maintain_order=True)


study_divs = []
for crit, group in study_groups:
    study = crit[0]
    imaging_configs = [','.join(config) for config in group['uniq_cols'].sort().to_list()]
    plates_divs = [html.Summary(study)]
    
    plates_divs.extend([html.Div(children=plate, className='sidebar-child', id={"index": plate, 'study':study}, n_clicks=0) for plate in imaging_configs])
    study_divs.append(html.Details(plates_divs, className="sidebar-parent", open=True))
    
studies = html.Div(study_divs, id="study_list")

app = Dash(__name__)

app.layout = html.Div([
    studies,
    html.Div([
        html.Div('Multichannel ID', id='image_name'),
        html.Div([
            dcc.Graph(id="picture", figure={})
        ], id='figure'),
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Total Images: "),
                    html.Span("0", id='total_ims'),    
                ], 'total_ims_container'),                
                html.Div(id="images"), 
            ], id='image_list_wrap'),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("col_eq: "),
                        dcc.Input(value=0, id="col_eq", debounce=True),
                        html.Button("segment", id="seg_button", className="save"),
                    ]),
                    html.Div([
                        html.Span("Diameter: "),
                        dcc.Input(value=30, type='number', id="diameter", debounce=True),
                    ]),
                    html.Div([
                        html.Span("View only: "),
                        dcc.Checklist([""],[""], id="classical"),
                    ], id='view_only_container')
                ],id='seg_in_wrapper'),
                html.Div("Channel info appears here",id="image_info"),
            ], id='int_wrapper'),
        ], id='image_inf_container'),
    ], id='display')
        
], id='app_layout')

@callback(
    Output("images", "children"),
    Output('image_name', 'children', allow_duplicate=True),
    Output('total_ims', 'children'),
    Input({"index": ALL, 'study': ALL}, "n_clicks"),
    prevent_initial_call=True
)
def display_output(index):
    study = ctx.triggered_id['study']
    channel_config = ctx.triggered_id['index']
    channel_config = channel_config.split(',') if ',' in channel_config else [channel_config]
    
    filtered = meta.filter(pl.col('experiment.study')==study).group_by('imaging.multi_channel_id').agg(config=pl.col('imaging.channel_type').sort()).filter(pl.col('config')==channel_config)
    top_100_id = filtered['imaging.multi_channel_id'].to_list()[:100]
    return [html.Div(pic, id={"multi_chan_id": pic}) for pic in top_100_id], top_100_id[0], f"{len(filtered):,}"
 
@callback(
    Output("image_info", "children"),
    Output('picture', 'figure'),
    Output('image_name', 'children'),
    State('col_eq', 'value'),
    Input({"multi_chan_id": ALL}, "n_clicks"),
    Input('seg_button', 'n_clicks'),
    State('image_name', 'children'),
    Input('classical', 'value'),
    Input('diameter', 'value'),
    prevent_initial_call=True
)
def display_image_stats(col_eq, index, _, image_name, classical, diameter):
    if 'multi_chan_id' in ctx.triggered_id:
        id = ctx.triggered_id['multi_chan_id']
    else:
        id = image_name

    cols = meta.filter(pl.col('imaging.multi_channel_id')==id).sort('imaging.channel')
    id_channel_map = {}
    for chan_id, channel_type in zip(cols['imaging.channel'].to_list(), cols['imaging.channel_type'].to_list()):
        id_channel_map[chan_id] = channel_type
        
    # nuc_idx = cols['imaging.channel_type'].to_list().index('nucleus')    
    
    images_paths = [os.path.join(imgs_base, row['storage.path']) for row in cols.iter_rows(named=True)]

    if len(classical) == 0:
        images = [decode_image(image)[0].numpy() for image in images_paths]
        # images = [cv2.medianBlur(image, ksize=3) for image in images]
        
        for image in images:
            image[image>0] == 155
        
        col_eq = [col_eq] if isinstance(col_eq, int) else col_eq.split(',')
        col_eq = [int(col) for col in col_eq]
        
        channel_axis = 1
        if len(col_eq) == 2:
                channels_config = [1, 2]
        else:
            channel_axis = None
            channels_config = [0, 0]
        
        if col_eq[0] != 0:
            cellpose_images = [images[idx-1] for idx in col_eq]
        else:
            cellpose_images = [image for image in images]
        
        # cellpose_images = [images[nuc_idx]]
            
        masks, _, _, _ = cellpose_model.eval(cellpose_images, channels=channels_config, channel_axis=channel_axis, do_3D=False, diameter=diameter, batch_size=32)
        mask: np.ndarray = masks[0].astype(np.uint8) 

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if mask.max() > 0:
            out_images = [cv2.cvtColor(((masks[0]-masks[0].min())/masks[0].max()).astype(np.float32), cv2.COLOR_BGR2RGB)]
        else:
            out_images = [cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)]
        for image in images:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            drawn_image = cv2.drawContours(image_rgb, contours, -1, color=(0, 255, 0), thickness=2)
            out_images.append((drawn_image-drawn_image.min())/drawn_image.max())
        
        big_image = np.concatenate(out_images, axis=1)
    else:
        images = [decode_image(image) for image in images_paths]
        ims = torch.concat(images).type(torch.float32)
        avg_img = ims.mean(dim=0).type(torch.uint8).numpy()
        _, binary_mask = cv2.threshold(avg_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        avg_img = cv2.GaussianBlur(binary_mask, ksize=[51,51], sigmaX=30, sigmaY=30)
        avg_img = cv2.GaussianBlur(avg_img, ksize=[25,25], sigmaX=30, sigmaY=30)
        avg_img = cv2.GaussianBlur(avg_img, ksize=[7,7], sigmaX=30, sigmaY=30)
        _, avg_img = cv2.threshold(avg_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        avg_img = scipy.ndimage.binary_fill_holes(avg_img > 0)
        avg_img = avg_img.astype(np.uint8) * 255
        # avg_img = cv2.erode(avg_img, np.ones((3,3), np.uint8), iterations=4)
        
        # labeled_mask, num_labels = skimage.measure.label(avg_img, return_num=True)
        # component_sizes = [labeled_mask[labeled_mask==label+1].sum() for label in range(num_labels)]
        # max_label = np.argmax(component_sizes)
        # largest_component_mask = labeled_mask == max_label+1
        
        # avg_img[largest_component_mask] = 255
        # avg_img[~largest_component_mask] = 0
        
        # row_indices, col_indices = np.where(avg_img == 255)
        # avg_x, avg_y = row_indices.mean(), col_indices.mean()
        # center_x = int(round(avg_x))
        # center_y = int(round(avg_y))
                
        # out_images = [cv2.cvtColor(avg_img, cv2.COLOR_BGR2RGB)]
        # contours, _ = cv2.findContours(avg_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for image in images:
        #     image_rgb = cv2.cvtColor(image[0].numpy(), cv2.COLOR_BGR2RGB)
        #     drawn_image = cv2.drawContours(image_rgb, contours, -1, color=(0, 255, 0), thickness=2)
        #     out_images.append(drawn_image)
        
        # big_image = np.concatenate(out_images, axis=1)
        big_image = np.concatenate([image[0] for image in images], axis=1)
        
    fig = px.imshow(big_image, binary_string=True, aspect='auto')
    
    # with open('output.txt', 'w') as f:
    #     f.write(str(cols[['imaging.channel', 'imaging.channel_type', 'storage.path']]))
    print('Displayed!')
    return html.Pre(yaml.dump(id_channel_map, indent=2)), fig, id
 
if __name__ == '__main__':
    app.run(debug=True, port=8050)
