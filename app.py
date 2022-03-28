import base64
import datetime
import io

#--------------------------------Prediction Model------------------------------------
import pgeocode# for getting lat & long from zipcode
nomi = pgeocode.Nominatim('us')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import joblib
# model = joblib.load("GBR.gz")
file="kc_house_data.csv" # data file
data = pd.read_csv(file)
data = data.drop(['id','date'],axis =1)
Y = data['price']
del data['price']
X = data.values
X_train, X_test , Y_train , Y_test =train_test_split(X, Y , test_size = 0.10,random_state =2)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
model.fit(X_train, Y_train)
#--------------------------------Line-Graph-data-----------------------------------------------------------------------
df=pd.read_csv('compare.csv')
dx=df.iloc[:,1:]
#--------------------------------Correlation-bar-graph-data--------------------------------------------------------------
# dc = pd.read_csv('Features_corr.csv')
#--------------------------------disturbution-graph------------------------------------------------------------------------
kc = pd.read_csv(file)
features = ['bedrooms','bathrooms','floors','view','condition','grade'] # option for Distrubution graph
#--------------------------------Data-for-scatter---------------------------------------------------------------------------
kx=pd.read_csv(file)
kx['prices'] = kx.apply(lambda x: "{:,}".format(x['price']), axis=1)
kx['txt'] = 'Price:'+kx['prices'].astype(str)+' | '+'Zipcode:'+kx['zipcode'].astype(str)
feature = ['bedrooms','bathrooms','sqft_living','floors','grade','sqft_living15']# options for Scatter plot|kc.columns
#--------------------------------Dashboard-main-------------------------------------------------------=====================

import dash
from dash import dash_table
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import flask
import dash_auth

app = dash.Dash(
    __name__,
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
)
url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

other_layout = html.Div([ # this code section taken from Dash docs https://dash.plotly.com/dash-core-components/upload
     html.Table([
        html.Tr([
        html.Th(html.Img(src=app.get_asset_url('infogen.png')), style={'textAlign': 'left'}),
        html.Th(html.H2("House Price Prediction")),
        html.Th(dcc.Link('Home', href='/page-0'),style={'text-align': 'center'}),                                                                                                                           
        html.Th(dcc.Link('Data', href='/page-1'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Distribution', href='/page-2'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Scatter', href='/page-3'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Zipcode', href='/page-4'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Price Predictor', href='/page-5'),style={'text-align': 'right'}),
       ])]),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    # html.Div(id='output-div'),
    html.Div(id='output-datatable'),
])



auth = dash_auth.BasicAuth(
    app,
    {'bugsbunny': 'topsecret',
     'pajaroloco': 'unsecreto'}
)

#-----------------------------------------page-0-Layout home page----------------------------------------------------------------
layout_index = html.Div([
    html.Table([
        html.Tr([
        html.Th(html.Img(src=app.get_asset_url('infogen.png')), style={'textAlign': 'left' }),
        html.Th(html.H2("House Price Prediction Using ML")),
        #  html.Th(dcc.Link('Login', href='/login'),style={'text-align': 'center'}), 
        html.Th(dcc.Link('Home', href='/page-0'),style={'text-align': 'center'}),                                                                                                                           
        html.Th(dcc.Link('Data', href='/page-1'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Distribution', href='/page-2'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Scatter', href='/page-3'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Zipcode', href='/page-4'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Price Predictor', href='/page-5'),style={'text-align': 'justify'}),
       ])]),
    html.Div(html.Img(src=app.get_asset_url('hpp1.png')), style={'textAlign': 'left', 'width':'50px'}),
])
#-----------------------------------------page1-Data----------------------------------------------------------------------
layout_page_1 = html.Div([
    html.Table([
        html.Tr([
        html.Th(html.Img(src=app.get_asset_url('infogen.png')), style={'textAlign': 'left'}),
        html.Th(html.H2("House Price Prediction Using ML")),
        html.Th(dcc.Link('Home', href='/page-0'),style={'text-align': 'center'}),                                                                                                                           
        html.Th(dcc.Link('Data', href='/page-1'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Distribution', href='/page-2'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Scatter', href='/page-3'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Zipcode', href='/page-4'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Price Predictor', href='/page-5'),style={'text-align': 'justify'}),
       ])]),

    html.Div([
    dash_table.DataTable(
    id='table-filtering-be',
    columns=[
        {"name": i, "id": i} for i in sorted(kc.columns)
         ],
    # filtering='be', # for filtering
    # filter=''
    )])

    # html.H2('Page 1'),
#     dcc.Graph(id='bar',
#         figure={'data':[
#         go.Bar(
#         x=dc['Features'], # assign x as the dataframe column 'x'
#         y=dc['Cor_value'],
#         marker=dict(
#             cmax=39,
#             cmin=0,
#             color=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,40],
# #             colorbar=dict(
# #                 title='Colorbar'
# #             ),
#             colorscale='Earth'),
#         )],
#         'layout': go.Layout(
#             title = 'Most important features relative to target(Price)',
#             yaxis = {'title': 'Features'},
#             hovermode='closest',
#             autosize=False,
#             width=1600,
#             height=700,
#             margin=go.layout.Margin(
#                 l=50,
#                 r=50,
#                 b=100,
#                 t=100,
#                 pad=4
#             ),
#             paper_bgcolor='#ffffff',
#             plot_bgcolor='#ffffff'
#             )}
#         )
    ])
#-----------------------------------------page-2-bar graph-counts--------------------------------------------------------------
layout_page_2 = html.Div([
    html.Table([
        html.Tr([
        html.Th(html.Img(src=app.get_asset_url('infogen.png')), style={'textAlign': 'left'}),
        html.Th(html.H2("House Price Prediction")),
        html.Th(dcc.Link('Home', href='/page-0'),style={'text-align': 'center'}),                                                                                                                           
        html.Th(dcc.Link('Data', href='/page-1'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Distribution', href='/page-2'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Scatter', href='/page-3'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Zipcode', href='/page-4'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Price Predictor', href='/page-5'),style={'text-align': 'right'}),
       ])]),
    # html.H2('Page 2'),
    html.Div([
        dcc.Dropdown(
            id='feature',
            options=[{'label': i.title(), 'value': i} for i in features],
            value='bathrooms',
            clearable=False,
            searchable=True,
        )
    ],style={'width': '30%', 'display': 'inline-block'}),
    dcc.Graph(id='bar-graph')
        ])
#-----------------------------------------page3-scatter-plot---------------------------------------------------------------
layout_page_3 = html.Div([
    html.Table([
        html.Tr([
        html.Th(html.Img(src=app.get_asset_url('infogen.png')), style={'textAlign': 'left'}),
        html.Th(html.H2("House Price Prediction")),
        html.Th(dcc.Link('Home', href='/page-0'),style={'text-align': 'center'}),                                                                                                                           
        html.Th(dcc.Link('Data', href='/page-1'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Distribution', href='/page-2'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Scatter', href='/page-3'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Zipcode', href='/page-4'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Price Predictor', href='/page-5'),style={'text-align': 'right'}),
       ])]),
    # html.H2('Page 3'),
    html.Div([
        dcc.Dropdown(
            id='xaxis',
            options=[{'label': i.title(), 'value': i} for i in feature],
            value='sqft_living'
        )],style={'width': '30%', 'display': 'inline-block'}),
    html.Div(dcc.Dropdown(
        id='yaxis',
        options=[{'label': 'Price', 'value': 'price'}], #[{'label': i.title(), 'value': i} for i in feature],
        clearable=False,
        searchable=False,
        value='price'),style={'width': '30%', 'display': 'inline-block'}),
    dcc.Graph(id='feature-graphic'),
])
#-----------------------------------------page-4-zipcode-wise--------------------------------------------------------------
layout_page_4 = html.Div([
    html.Table([
        html.Tr([
        html.Th(html.Img(src=app.get_asset_url('infogen.png')), style={'textAlign': 'left'}),
        html.Th(html.H2("House Price Prediction")),
        html.Th(dcc.Link('Home', href='/page-0'),style={'text-align': 'center'}),                                                                                                                           
        html.Th(dcc.Link('Data', href='/page-1'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Distribution', href='/page-2'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Scatter', href='/page-3'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Zipcode', href='/page-4'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Price Predictor', href='/page-5'),style={'text-align': 'right'}),
       ])]),
    # html.H2('Page 4'),
    html.Div([
    # html.Div('Zipcode range: 98001-98119',style={'width': '10%', 'float':'center'}),    
    dcc.Input(id='filt',type='number',min=98001, max=98119,value=98001,style={'width': '10%', 'float':'center'}),
    dcc.Graph(id='line-graph')],style={'textAlign': 'center'})
])
#-----------------------------------------page-5-Price predictor---------------------------------------------------------------
layout_page_5 = html.Div([
    html.Table([
        html.Tr([
        html.Th(html.Img(src=app.get_asset_url('infogen.png')), style={'textAlign': 'left'}),
        html.Th(html.H2("House Price Prediction")),
        html.Th(dcc.Link('Home', href='/page-0'),style={'text-align': 'center'}),                                                                                                                           
        html.Th(dcc.Link('Data', href='/page-1'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Distribution', href='/page-2'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Scatter', href='/page-3'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Zipcode', href='/page-4'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Price Predictor', href='/page-5'),style={'text-align': 'right'}),
       ])]),
    # html.H2('Page 5'),
    html.Table([

        html.Tr([
                html.Th('No. of Bedrooms',style={'text-align': 'center'}),                                                                                                                           
                html.Th('No. of Bathrooms',style={'text-align': 'center'}),
                html.Th('Living Area',style={'text-align': 'center'}),
                html.Th('Total Area',style={'text-align': 'center'}),
                html.Th('No. of Floors',style={'text-align': 'center'}),                                                                                                                           
                html.Th('Waterfront',style={'text-align': 'center'}),
                html.Th('View',style={'text-align': 'center'}),
                html.Th('Condition',style={'text-align': 'center'})
            ]),
        html.Tr([
            html.Th(html.Div(daq.NumericInput( id='my-dropdown',min=1,max=8,value=1,size=120),style={'text-align': 'center'})),
            html.Th(html.Div(dcc.Dropdown(
        id='my-dropdown1',
        options=[
            {'label': '0.5', 'value': 0.5},
            {'label': '0.75', 'value': 0.75},
            {'label': '1.0', 'value': 1.0},
            {'label': '1.25', 'value': 1.25},
            {'label': '1.5', 'value': 1.5},
            {'label': '1.75', 'value': 1.75},
            {'label': '2.0', 'value': 2.0},
            {'label': '2.25', 'value': 2.25},
            {'label': '2.5', 'value': 2.5},
            {'label': '2.75', 'value': 2.75},
            {'label': '3.0', 'value': 3.0},
            {'label': '3.25', 'value': 3.25},
            {'label': '3.5', 'value': 3.5},
            {'label': '3.75', 'value': 3.75},
            {'label': '4.0', 'value': 4.0},
            {'label': '4.25', 'value': 4.25},
            {'label': '4.5', 'value': 4.5},
            {'label': '4.75', 'value': 4.75},
            {'label': '5.0', 'value': 5.0},
            {'label': '5.25', 'value': 5.25},
            {'label': '5.5', 'value': 5.5},
            {'label': '5.75', 'value': 5.75},
            {'label': '6.0', 'value': 6.0},
            {'label': '6.25', 'value': 6.25},
            {'label': '6.5', 'value': 6.5},
            {'label': '6.75', 'value': 6.75},
            {'label': '7.0', 'value': 7.0},
            {'label': '7.25', 'value': 7.25},
            {'label': '7.5', 'value': 7.5},
            {'label': '7.75', 'value': 7.75},
            {'label': '8.0', 'value': 8.0}
        ],value='0.5',
        clearable=False,
        searchable=False,
          # style={"height" : "25%", "width" : "40%"},

          ))),
            html.Th(html.Div(daq.NumericInput( id='sqft_live',min=290, max=13540,value=290,size=120),style={'text-align': 'center'})),
            html.Th(html.Div(daq.NumericInput( id='sqft_lot',min=520, max=1651359,value=290,size=120),style={'text-align': 'center'})),
            html.Th(html.Div(dcc.Dropdown(
        id='floor',
        options=[
            {'label': '1.0', 'value': 1.0},
            {'label': '1.5', 'value': 1.5},
            {'label': '2.0', 'value': 2.0},
            {'label': '2.5', 'value': 2.5},
            {'label': '3.0', 'value': 3.0},
            {'label': '3.5', 'value': 3.5},
        ],value=1.0,#style={"height" : "55%", "width" : "70%"},
        clearable=False,
        searchable=False,

        ))),
                html.Th(html.Div(dcc.Dropdown(
        id='waterfront',
        options=[
            {'label': 'Yes', 'value': 1.0},
            {'label': 'No', 'value': 0.0},
        ],value=0.0,#style={"height" : "25%", "width" : "70%"},
        clearable=False,
        searchable=False,
        ))),
                html.Th(html.Div(daq.NumericInput( id='view',min=0, max=4,value=0,size=120),style={'text-align': 'center'})),
                html.Th(html.Div(daq.NumericInput( id='condition',min=1, max=5,value=1,size=120),style={'text-align': 'center'}))
            ]),
            html.Tr([
                html.Td(id='output-container'),                                                                                                                           
                html.Td(id='output-container1'),
                html.Td(id='output-container2'),
                html.Td(id='output-container3'),
                html.Td(id='output-container4'),
                html.Td(id='output-container5'),
                html.Td(id='output-container6'),
                html.Td(id='output-container7')
            ]),
            html.Tr([
                html.Th('Grade',style={'text-align': 'center'}),                                                                                                                           
                html.Th('Area above ground',style={'text-align': 'center'}),
                html.Th('Basement Area',style={'text-align': 'center'}),
                html.Th('Year of built',style={'text-align': 'center'}),
                html.Th('Year of Renovation',style={'text-align': 'center'}),                                                                                                                           
                html.Th('Zipcode',style={'text-align': 'center'}),
                html.Th('Living Area of nearby property',style={'text-align': 'center'}),
                html.Th('Total Area of nearby property',style={'text-align': 'center'})
            ]),
            html.Tr([
                html.Th(html.Div(daq.NumericInput( id='grade',min=1, max=13,value=1,size=120),style={'text-align': 'center'})),
                html.Th(html.Div(daq.NumericInput( id='sqft_above',min=1190, max=9410,value=1190,size=120),style={'text-align': 'center'})),
                html.Th(html.Div(daq.NumericInput( id='sqft_basement',min=0, max=4820,value=0,size=120),style={'text-align': 'center'})),
                html.Th(html.Div(daq.NumericInput( id='year_built',min=1900, max=2015,value=1900,size=120),style={'text-align': 'center'})),
                html.Th(html.Div(daq.NumericInput( id='year_renew',min=0, max=2015,value=0,size=120),style={'text-align': 'center'})),
                html.Th(html.Div(daq.NumericInput( id='zipcode',min=98001, max=98119,value=98001,size=120),style={'text-align': 'center'})),
                html.Th(html.Div(daq.NumericInput( id='sqft_live_area_nearby',min=399, max=6210,value=399,size=120),style={'text-align': 'center'})),
                html.Th(html.Div(daq.NumericInput( id='sqft_tot_area_nearby',min=651, max=871200,value=651,size=120),style={'text-align': 'center'}))
            ]),
            html.Tr([
                html.Td(id='output-container8'),
                html.Td(id='output-container9'),
                html.Td(id='output-container10'),
                html.Td(id='output-container11'),
                html.Td(id='output-container12'),
                html.Td(id='output-container13'),
                html.Td(id='output-container14'),
                html.Td(id='output-container15')
            ]),
            html.Tr([
                html.Th(html.Button(id='submit', n_clicks=0, children='Estimate',style={'color': 'white','background-color': '#1fbfcf'}),style={'text-align': 'center'}),
                html.Th('Estimated Cost: ',style={'text-align': 'center','color': '#1fbfcf','fontSize': 20}),                                                                                                                           
                html.Th(id='output-container16',style={'text-align': 'left','color': 'red','fontSize': 20}),
                html.Th('Accuracy of Model: ',style={'text-align': 'center'}),
                html.Th(str(round(model.score(X_test, Y_test)*100,2))+'%',style={'text-align': 'left','color': 'green','fontSize': 20}),
            ]),
    ]),
    html.Div(),
    html.Br(),
    dcc.Graph(
            id='line',
             figure={
            'data': [
                go.Scatter(
                    x = dx.index,
                    y = dx.Actual,
#                    z = random_z,
                    mode = 'lines+markers',
                    name = 'Actual'
                ),
                go.Scatter(
                    x = dx.index,
                    y = dx.Predicted,
#                    z = random_z,
                    mode = 'lines',
                    name = 'Predicted')
            ],
            'layout': go.Layout(
                title = 'Line chart showing Actual Vs Predicted Price',
                yaxis = {'title': 'Price (USD)'},
                xaxis = {'title': 'Houses'},
                # 'height': 1000,
                # 'margin': {'l': 10, 'b': 20, 't': 0, 'r': 0}
                # hovermode='closest',
                width=1580,
            height=700,
            margin=go.layout.Margin(
                l=50,
                r=10,
                b=100,
                t=100,
                pad=4
                )
            )
        }
    )
])
#---------------------------------------------Page-6-Result---------------------------------------------------
layout_page_6 = html.Div([
    html.Table([
        html.Tr([
        html.Th(html.Img(src=app.get_asset_url('infogen.png')), style={'textAlign': 'left'}),
        html.Th(html.H2("House Price Prediction")),
        html.Th(dcc.Link('Home', href='/page-0'),style={'text-align': 'center'}),                                                                                                                           
        html.Th(dcc.Link('Data', href='/page-1'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Distribution', href='/page-2'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Scatter', href='/page-3'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Zipcode', href='/page-4'),style={'text-align': 'center'}),
        html.Th(dcc.Link('Price Predictor', href='/page-5'),style={'text-align': 'right'}),
       ])]),
    html.Div([
    dash_table.DataTable(
        id='datatable-filtering-fex',
        columns=[
            {"name": i, "id": i, "deletable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        # filter_action="native",
    ),
    html.Div(id='datatable-filter-container')
    ]),
    
    ])

def serve_layout():
    if flask.has_request_context():
        return url_bar_and_content_div
    return html.Div([
        url_bar_and_content_div,
        layout_index,
        other_layout,
        layout_page_2,
        layout_page_3,
        layout_page_4,
        layout_page_5,
        layout_page_6
    ])

app.layout = serve_layout


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df = pd.read_csv(file)
            df = df.drop(['id','date'],axis =1)
            Y = df['price']
            del df['price']
            X = df.values
            X_train, X_test , Y_train , Y_test =train_test_split(X, Y , test_size = 0.10,random_state =2)
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
            model.fit(X_train, Y_train)
            # return model

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.P("Inset X axis data"),
        dcc.Dropdown(id='xaxis-data',
                     options=[{'label':x, 'value':x} for x in df.columns]),
        html.P("Inset Y axis data"),
        dcc.Dropdown(id='yaxis-data',
                     options=[{'label':x, 'value':x} for x in df.columns]),
        html.Button(id="submit-button", children="Create Graph"),
        html.Hr(),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=15
        ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])



# Index callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/page-1":
        return other_layout
    if pathname == "/page-2":
        return layout_page_2
    elif pathname == "/page-3":
        return layout_page_3
    elif pathname == "/page-4":
        return layout_page_4
    elif pathname == "/page-5":
        return layout_page_5
    elif pathname == "/page-6":
        return layout_page_6
    else:
        return layout_index





@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children







#------------------------------Data-table---------------------------------------------
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
    Output('table-filtering-be', "data"),
    [Input('table-filtering-be', "filter")])
def update_table(filter):
    filtering_expressions = filter.split(' && ')
    dff = kc
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    return dff.to_dict('records')
#--------------------------------------------dynamic-barplot-----------------------------------------
@app.callback(
    Output('bar-graph', 'figure'),
    [Input('feature', 'value')]) 
def update_graph(code):
    dx = pd.Series.to_frame(kc[code].value_counts())
    dx['count'] = list(dx.index)

    return{'data':[
    go.Bar(
    x=dx['count'], # assign x as the dataframe column 'x'
    y=dx[code],
    # text=x,
    # textposition = 'auto',
    marker=dict(
        cmax=39,
        cmin=0,
        color=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20, 
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,40],
#             colorbar=dict(
#                 title='Colorbar'
#             ),
        colorscale='Earth'),
    )],
    'layout': go.Layout(
        title = 'Count of '+code.capitalize() ,
        yaxis = {'title': 'Count'},
        xaxis=dict(
        title=code.capitalize(),
        # tickmode='linear',
        # ticks='outside',
        # tick0=0,
        # dtick=0.25,
        # ticklen=8,
        # tickwidth=1,
        tickcolor='black'),
        # hovermode='closest',
        autosize=False,
        width=1600,
        height=600,
        margin=go.layout.Margin(
            l=100,
            r=100,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff'
        )}
#------------------------------------scatterplot------------------------------
@app.callback(
    Output('feature-graphic', 'figure'),
    [Input('xaxis', 'value'),
     Input('yaxis', 'value')])
def update_graph(xaxis_name, yaxis_name):
    return {
        'data': [go.Scatter(
            x=kx[xaxis_name],
            y=kx[yaxis_name],
            text=kx['txt'],
            hoverinfo = 'text',
            marker=dict(
            size=8,
            cmax=39,
            cmin=0,
            # color=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            # colorscale='Earth',
            opacity=0.5,
            line={'width': 0.5, 'color': 'blue'}
        ),
            # text=df['name'],
            mode='markers',
        )],
        'layout': go.Layout(
            xaxis={'title': xaxis_name.title()},
            yaxis={'title': yaxis_name.title()},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest',autosize=False,
            width=1500,
            height=600,

        )
    }
#----------------------zipcode-wise-line-plot------------------------------------
@app.callback(
    Output('line-graph', 'figure'),
    [Input('filt', 'value')]) 
def update_graph(codes):
    do=df.ix[df['Zipcode']==codes,1:3]
    return {
            'data': [
                go.Scatter(
                    # x = dx.index,
                    y = do.Actual,
#                    z = random_z,
                    mode = 'lines+markers',
                    name = 'Actual'
                ),
                go.Scatter(
                    # x = dx.index,
                    y = do.Predicted,
#                    z = random_z,
                    mode = 'lines',
                    name = 'Predicted')
            ],
            'layout': go.Layout(
                title = 'Line chart showing Actual Vs Predicted Price | Zipcode: '+str(codes),
                yaxis = {'title': 'Price (USD)'},
                xaxis = {'title': 'Houses'},
                # hovermode='closest',
                autosize=False,
            width=1500,
            height=600,
            margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
            )
        }
#--------------------------------------price-predictor--------------------------
@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_output(value):
    return 'Number of Bedrooms: {}'.format(value)

@app.callback(
    dash.dependencies.Output('output-container1', 'children'),
    [dash.dependencies.Input('my-dropdown1', 'value')])
def update_output(value):
    return 'Number of Bathrooms: {}'.format(value)

@app.callback(
    dash.dependencies.Output('output-container2', 'children'),
    [dash.dependencies.Input('sqft_live', 'value')])
def update_output(value):
    return 'Area of living: {} sqft'.format(value)

@app.callback(
    dash.dependencies.Output('output-container3', 'children'),
    [dash.dependencies.Input('sqft_lot', 'value')])
def update_output(value):
    return 'Total area: {} sqft'.format(value)

@app.callback(
    dash.dependencies.Output('output-container4', 'children'),
    [dash.dependencies.Input('floor', 'value')])
def update_output(value):
    return 'Number of floors: {}'.format(value)

@app.callback(
    dash.dependencies.Output('output-container5', 'children'),
    [dash.dependencies.Input('waterfront', 'value')])
def update_output(value):
    if value==0:
        return 'Waterfront present: No'
    else:
        return 'Waterfront present: Yes'
    # return 'Water front: {}'.format(wf)

@app.callback(
    dash.dependencies.Output('output-container6', 'children'),
    [dash.dependencies.Input('view', 'value')])
def update_output(value):
    return '(An index from 0 to 4 of how good the view of the property was).View of the property: {}'.format(value)

@app.callback(
    dash.dependencies.Output('output-container7', 'children'),
    [dash.dependencies.Input('condition', 'value')])
def update_output(value):
    return '(Condition Rating 1 to 5)Condition of property: {}'.format(value)

@app.callback(
    dash.dependencies.Output('output-container8', 'children'),
    [dash.dependencies.Input('grade', 'value')])
def update_output(value):
    return '(1-3:Low|7:Avg|11-13:High)Grade for quality level of construction and design: {}'.format(value)

@app.callback(
    dash.dependencies.Output('output-container9', 'children'),
    [dash.dependencies.Input('sqft_above', 'value')])
def update_output(value):
    return 'Area of interior housing space above ground level: {} sqft'.format(value)

@app.callback(
    dash.dependencies.Output('output-container10', 'children'),
    [dash.dependencies.Input('sqft_basement', 'value')])
def update_output(value):
    return 'Area of interior housing space below ground level: {} sqft'.format(value)

@app.callback(
    dash.dependencies.Output('output-container11', 'children'),
    [dash.dependencies.Input('year_built', 'value')])
def update_output(value):
    return 'Year the house was built: {}'.format(value)

@app.callback(
    dash.dependencies.Output('output-container12', 'children'),
    [dash.dependencies.Input('year_renew', 'value')])
def update_output(value):
    if value==0:
        return 'Year the house was last renovated: Never'
    else:
        return 'Year the house was last renovated: {}'.format(value)

@app.callback(
    Output(component_id='output-container13', component_property='children'),
    [Input(component_id='zipcode', component_property='value')]
)
def update_output_div(input_value):

    Latitude=nomi.query_postal_code(input_value)[9]
    Longitude=nomi.query_postal_code(input_value)[10]

    update_output_div.Latitude=nomi.query_postal_code(input_value)[9]
    update_output_div.Longitude=nomi.query_postal_code(input_value)[10]
    # othr=nomi.query_postal_code(input_value)
    return 'Zipcode: {} | Latitude: {} | Longitude: {}'.format(input_value,Latitude,Longitude)

@app.callback(
    dash.dependencies.Output('output-container14', 'children'),
    [dash.dependencies.Input('sqft_live_area_nearby', 'value')])
def update_output(value):
    return 'Area of living for the nearest 15 neighbors: {}'.format(value)

@app.callback(
    dash.dependencies.Output('output-container15', 'children'),
    [dash.dependencies.Input('sqft_tot_area_nearby', 'value')])
def update_output(value):
    return 'Total area for the nearest 15 neighbors: {}'.format(value)

@app.callback(
    Output('output-container16', 'children'),
    [Input('submit', 'n_clicks')],
    [State('my-dropdown', 'value'),
     State('my-dropdown1', 'value'),
     State('sqft_live','value'),  
     State('sqft_lot','value'),
     State('floor','value'),
     State('waterfront','value'),
     State('view','value'),
     State('condition','value'),
     State('grade','value'),
     State('sqft_above','value'),
     State('sqft_basement','value'),
     State('year_built','value'),
     State('year_renew','value'),
     State('zipcode','value'),
     State('sqft_live_area_nearby','value'),
     State('sqft_tot_area_nearby','value')])
def pred(n_clicks,No_of_Bedrooms,No_of_Bathrooms,Sqft_live_area,Sqft_tot_area,No_of_floors,
     Water_front,View,Condition,Grade,Sqft_area_above,Sqft_area_basement,Year_built,
     Year_renew,Zipcode,Sqft_live_area_nearby,Sqft_tot_area_nearby):

    ip=[No_of_Bedrooms,No_of_Bathrooms,Sqft_live_area,Sqft_tot_area,No_of_floors,
         Water_front,View,Condition,Grade,Sqft_area_above,Sqft_area_basement,Year_built,
         Year_renew,Zipcode,update_output_div.Latitude,update_output_div.Longitude,Sqft_live_area_nearby,Sqft_tot_area_nearby]
    print(ip)
    
    user_input=np.asarray([No_of_Bedrooms,No_of_Bathrooms,Sqft_live_area,Sqft_tot_area,No_of_floors,
         Water_front,View,Condition,Grade,Sqft_area_above,Sqft_area_basement,Year_built,
         Year_renew,Zipcode,update_output_div.Latitude,update_output_div.Longitude,Sqft_live_area_nearby,Sqft_tot_area_nearby],dtype=np.float64)

    print(user_input)
    testme=user_input.reshape(1, -1)
    # pred_price=model.predict(testme)
    # acc=model.score(X_test, Y_test)*100
    # pred_price=round(model.predict(testme)[0],2)
    print (" ${:,.2f}".format(model.predict(testme)[0]))
    # print(pred_price)
    return " ${:,.2f}".format(model.predict(testme)[0])
    # return ' ${}'.format(pred_price)
#---------------------------------------------Results-page-6-------------------------------------
@app.callback(
    Output('datatable-filter-container', "children"),
    [Input('datatable-filtering-fex', "data")])
def update_graph(rows):
    if rows is None:
        dff = df
    else:
        dff = pd.DataFrame(rows)

    return html.Div()



if __name__ == '__main__':
    app.run_server(debug=False)
