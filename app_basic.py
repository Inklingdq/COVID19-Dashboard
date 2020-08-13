import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
import pandas as pd
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly
import pickle
from datetime import datetime, timedelta
import os
from os.path import join as oj

import nn

## find the closet pkl file to load
def load_data():
    cached_dir = "/home/ubuntu/new_uploader/data"
    for i in range(10):
        d = (datetime.today() - timedelta(days=i)).date()
        cached_fname = oj(cached_dir, f'preds_{d.month}_{d.day}_cached.pkl')
        if os.path.exists(cached_fname):
            with open(cached_fname, 'rb') as f:
                df = pickle.load(f)
            break
    return df
## clean df: fill missing state abbreviation; create align-deaths/cases cols
def clean_data(df):
  def fillstate(df):
    us_state_abbrev = {
            'AL': 'Alabama',
            'AK': 'Alaska',
            'AZ': 'Arizona',
            'AR': 'Arkansas',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'HI': 'Hawaii',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'IA': 'Iowa',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'ME': 'Maine',
            'MD': 'Maryland',
            'MA': 'Massachusetts',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MS': 'Mississippi',
            'MO': 'Missouri',
            'MT': 'Montana',
            'NE': 'Nebraska',
            'NV': 'Nevada',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NY': 'New York',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VT': 'Vermont',
            'VA': 'Virginia',
            'WA': 'Washington',
            'WV': 'West Virginia',
            'WI': 'Wisconsin',
            'WY': 'Wyoming'
          }
    for i in range(df.shape[0]):
        if df.loc[i, "State"] not in us_state_abbrev.values():
            df.loc[i, "State"] = us_state_abbrev[df.loc[i,"StateName"]]
  def align(df, key):
    def find_nonzero(arr):
      i = 0
      for i in range(len(arr)):
        if arr[i]:
          break
      return i - 1
            
    ans = []
    for i in range(len(df)):
      k = find_nonzero(df.loc[i, key])
      if k == 0:
          k = 1
      ans.append(df.loc[i, key][k - 1:])
    df['aligned_' + key] = ans

  fillstate(df)
  df = df.sort_values(by = 'countyFIPS').reset_index()
  df['pos'] = df['CountyName'] + ', ' + df['StateName']
  align(df, 'deaths')
  align(df, 'cases')
  return df
##load and clean data
df = load_data()
df = clean_data(df)

# Initialize the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


def get_options(list_counties):
    dict_list = []
    for i in list_counties:
        dict_list.append({'label': i, 'value': i})

    return dict_list

def get_feats():
    dict_list = []
    feats = {"# ICU Beds": "#ICU_beds", "Recent Daily Deaths": "#Deaths_07-03-2020", "Median Age": "MedianAge2010", "Total Deaths": "tot_deaths"}
    for k, v in feats.items():
        dict_list.append({'label': k, 'value': v})

    return dict_list

app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H1('COVID19 County Nearest Neighbors & Visualization'),
                                 html.P('Visualizing time series of multiple counties.', style={'font-style': 'italic'}),
                                 html.Br(),
                                 html.P('Nearest Neighbors or Manual Selection', style={'text-align': 'center'}),
                                 html.Div(daq.ToggleSwitch(id='my-toggle-switch',
                                                           size=60, color='red',
                                                           value=False)),
                                 html.Br(),
                                 html.Div(id='toggle-switch-output'),
                                 html.Br(),
                                 html.P('Use the toggle button above to switch between dashboards: nearest neighbors or manually choosing counties to visualize.', style={'font-style': 'italic', 'text-align': 'center'}),
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries', config={'displayModeBar': True})
                             ])
                              ],style = {'display': 'inline-block', 'width': '100%','height':'200%'}),
                                
        ]

)


# Callback for timeseries price
@app.callback(Output('timeseries', 'figure'),
              [Input('countyselector', 'value'), Input('nnfeatselector', 'value')])
def update_graph(selected_counties, selected_features):
    print(selected_counties)
    print(selected_features)
    print(isinstance(selected_counties, list))
    cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', 
        '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
        '#bcf60c', '#fabebe', '#008080', '#e6beff', 
        '#9a6324', '#fffac8', '#800000', '#aaffc3', 
        '#808000', '#ffd8b1', '#000075', '#808080', 
        '#ffffff', '#000000']
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Cases", "Deaths", "Cases (Aligned to first case)", "Deaths (Aligned to first death)"))
    keys = ['cases', 'deaths', 'aligned_cases', 'aligned_deaths']
    import sys
    sys.path.append("/home/ubuntu/new_uploader/viz")
    #sys.path.append("/usr/local/google/home/danqingwang/covid19-severity-prediction/viz")
    import viz_map_utils
    date1 = viz_map_utils.date_in_data(df)
#     print(date1)
    date2 = [i for i in range(len(date1))]
    
    # if multiple counties are selected, aka manual select
    if isinstance(selected_counties, list):
        for i in range(4):
            key = keys[i]
            for index, pos in enumerate(selected_counties):
                row = df[df['pos'] == pos]
                if i < 2:
                    dates = date1
                else:
                    dates = date2
                fig.add_trace(go.Scatter(x=dates,
                            y=row[key].to_list()[0],
                            line=dict(color=cols[index]),
                            name = row['CountyName'].values[0] +', ' + row['StateName'].values[0],
                            showlegend = i == 0
                            ),row = i //2 + 1, col = i - (i// 2)*2 + 1) 
    # if a single county is selected, aka nearest neighbors
    elif selected_counties is not None:
        # plot selected county
        for i in range(4):
            key = keys[i]
            row = df[df['pos'] == selected_counties]
            if i < 2:
                dates = date1
            else:
                dates = date2
            fig.add_trace(go.Scatter(x=dates,
                        y=row[key].to_list()[0],
                        line=dict(color=cols[0]),
                        name = row['CountyName'].values[0] +', ' + row['StateName'].values[0],
                        showlegend = i == 0
                        ),row = i //2 + 1, col = i - (i// 2)*2 + 1)
        # if features are selected, run nearest neighbors and plot results    
        if selected_features is not None:
            if len(selected_features)!= 0:
                neighs = nn.model(df, selected_features, selected_counties)
                print(neighs)
                for i in range(4):
                    key = keys[i]
                    for index, pos in enumerate(neighs):
                        row = df[df['pos'] == pos]
                        if i < 2:
                            dates = date1
                        else:
                            dates = date2
                        print(row['CountyName'].values[0] +', ' + row['StateName'].values[0])
                        fig.add_trace(go.Scatter(x=dates,
                                    y=row[key].to_list()[0],
                                    line=dict(color=cols[index+1]),
                                    name = row['CountyName'].values[0] +', ' + row['StateName'].values[0],
                                    showlegend = i == 0
                                    ),row = i //2 + 1, col = i - (i// 2)*2 + 1)
                
    fig.update_layout(height=1000,                         
                      template='plotly_dark',
                      xaxis_title="Time",
                      yaxis_title="Count",
                      yaxis_showgrid=True)

    # edit axis labels
    fig['layout']['xaxis']['title']='Date'
    fig['layout']['xaxis2']['title']='Date'
    fig['layout']['xaxis3']['title']='Days'
    fig['layout']['xaxis4']['title']='Days'
    fig['layout']['yaxis']['title']='Cases'
    fig['layout']['yaxis2']['title']='Deaths'
    fig['layout']['yaxis3']['title']='Cases'
    fig['layout']['yaxis4']['title']='Deaths'                                
    return fig


@app.callback(
    dash.dependencies.Output('toggle-switch-output', 'children'),
    [dash.dependencies.Input('my-toggle-switch', 'value')])
def update_output(value):
    switch_bool = 'The switch is {}. '.format(value)
    if value == True:
        return html.Div(className='div-for-dropdown',
                        children=[
                            html.P('Pick one or more counties from the dropdown below.'),
                            dcc.Dropdown(id='countyselector', options=get_options(df['pos'].unique()),
                                         multi=True, value=[df['pos'][0]],
                                         style={'backgroundColor': '#1E1E1E'},
                                         className='countyselector'
                                        ),
                            dcc.Dropdown(id='nnfeatselector', options=get_feats(),
                                         multi=True,
                                         style={'display': 'none'},
                                         className='nnfeatselector'
                                        )
                                 ],
                        )
    else:
        return html.Div(className='div-for-dropdown',
                        children=[
                            html.P('Pick a county from the dropdown below.'),
                            dcc.Dropdown(id='countyselector', options=get_options(df['pos'].unique()),
                                         multi=False, value=df['pos'][0],
                                         style={'backgroundColor': '#1E1E1E'},
                                         className='countyselector'
                                        ),
                            html.Br(),
                            html.P('Pick one or more features from the dropdown below.'),
                            dcc.Dropdown(id='nnfeatselector', options=get_feats(),
                                         multi=True,
                                         style={'backgroundColor': '#1E1E1E'},
                                         className='nnfeatselector'
                                        ),
                                 ],
                        ) 
    

if __name__ == '__main__':
    app.run_server(debug=True, host = '0.0.0.0')
