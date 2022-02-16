# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:39:59 2020

@author: KANE
"""
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table 
import plotly.graph_objs as go
import datetime as dt
import plotly.express as px
#import pandas_datareader.data as web
import pandas as pd


import base64
import datetime
import io


iTemperature = "Rejection Counts"
app= dash.Dash()

#df = pd.read_csv("cat5.csv")
colors = {
    "graphBackground": "#212529",
    "background": "#000000",
    "text": "#ffffff"
}


#df = pd.read_csv("cat5.csv")

app.layout = html.Div([
        
        
          dcc.Tabs([
                
                
#                 dcc.Tab(label='Metal Detector', children=[
                        
#                         html.H1('Mettel Detector'),
                        
                        
                        
                        
                        
                        
        
        
        
#          dcc.Tabs([
                
                 
#                   dcc.Tab(label='Add File', children=[
                          
#                                       dcc.Upload(
#         id='upload-data2',
#         children=html.Div([
#             'Drag and Drop or ',
#             html.A('Select Files')
#         ]),
#         style={
#             'width': '100%',
#             'height': '60px',
#             'lineHeight': '60px',
#             'borderWidth': '1px',
#             'borderStyle': 'dashed',
#             'borderRadius': '5px',
#             'textAlign': 'center',
#             'margin': '10px'
#         },
#         # Allow multiple files to be uploaded
#         multiple=True
#     ),

#  html.Div(id='output-data-upload4'),
                          
                          
#                           ]),
                 
                 
                
#                 dcc.Tab(label='Date Wise', children=[
                        
                        
                        
                        
                        
                        
             
    
    
    
    
    
#      html.Div(children="Graph based on rejection counts ", style={
#         "textAlign": "center",
#         "color": colors["text"]
#     }),

#     html.Div(children="", style={
#         "color": colors["background"]
#     }),

#     dcc.DatePickerRange(
#         id="date-picker-rangeM4",
#         start_date=dt.datetime(2018, 1, 2),
#         end_date=dt.datetime(2019, 9, 23),
#         min_date_allowed=dt.datetime(2018, 1, 2),
#         max_date_allowed=dt.datetime(2019, 9, 23),
#         end_date_placeholder_text="Select a date"
#     ),
    
#     html.Div([
#             html.Button(id='submit-buttonM4',
#                         n_clicks=0,
#                         children='Submit',
#                         style={'fontSize':24,'marginLeft':'30px'}
            
#             )        
            
#     ],style={'display':'inline-block'}),

#     dcc.Graph(id="in-temp-graphM4"),
#     dcc.Graph(id="pie-chart-statusM4"),
    
    
    
                        
                        
                        
                        
                        
                        
                        
#                         ]),



# #tab 2


                
                
#                 dcc.Tab(label='Report/Status Graph', children=[
                        
                        
                        
                        
                        
                        
#     html.Div([
            
#            html.P('Total Hrs machine worked :'),  html.P(id='output-sum'),
#              html.P('Total number of reject count  :'), html.P(id='output-sum2'),
#              html.P('Average Hrs machine worked :'), html.P(id='output-sum3'),
#              html.P('Average number of reject count :'), html.P(id='output-sum4'),
#              html.P('Average Hrs of Safe Zone :'), html.P(id='output-sum5'),
            
#             ], style={'marginBottom': 50, 'marginTop': 25, 'paddingTop':100}),
   
    
   
                       
                        
                        
                        
                        
                        
                        
                        
                        
#                          html.Div(children="Graph based on rejection counts ", style={
#         "textAlign": "center",
#         "color": colors["text"]
#     }),

#     html.Div(children="", style={
#         "color": colors["background"]
#     }),

#     dcc.DatePickerRange(
#         id="date-picker-rangeM",
#         start_date=dt.datetime(2018, 1, 2),
#         end_date=dt.datetime(2019, 9, 23),
#         min_date_allowed=dt.datetime(2018, 1, 2),
#         max_date_allowed=dt.datetime(2019, 9, 23),
#         end_date_placeholder_text="Select a date"
#     ),
    
#     html.Div([
#             html.Button(id='submit-buttonM',
#                         n_clicks=0,
#                         children='Submit',
#                         style={'fontSize':24,'marginLeft':'30px'}
            
#             )        
            
#     ],style={'display':'inline-block'}),

#     dcc.Graph(id="in-temp-graphM"),
    
  
                        
                        
                        
#                         ])
                        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   
    
    
    
    
    
   
    
    
# ])
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                              
#                         ]),
               

                
                
                
                dcc.Tab(label='Multi Sensor Data', children=[
                        
                        html.H1('Data Analysis'),
                        
                         dcc.Tabs([
                
                
                dcc.Tab(label='Add File', children=[
                        
                        
                        
                       
                        
                        
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
    ),html.Div(id='output-data-upload'),
    
                        
                        
                        ]),
                
                
        dcc.Tab(label='Select Multiple Days', children=[
            html.H1('MultiSensor Data Analysis'),
                     
                          
                         dcc.DatePickerSingle(
                                 
                                 id="date-picker-range",
    date=dt.datetime(2018, 1, 2)
   
)  
   ,
   
   
   
    
    
    html.Div([
            html.Button(id='submit-button',
                        n_clicks=0,
                        children='Submit',
                        style={'fontSize':24,'marginLeft':'30px'}
            
            )        
            
    ],style={'display':'inline-block'}),
                          
                          
                dcc.Graph(id='my_graph'),
                           #   figure={'data':[
                            #          {'x':[1,2],'y':[3,1]}
                                        
                            #],'layout':{'title':'Default Title'}}
                
                dcc.Graph(id="pie_judgement"),
        ]),
            
            
        dcc.Tab(label='Single Day Analysis', children=[
             html.H1('MultiSensor Data Analysis'),
                
                          
                          
                         # dcc.DatePickerSingle(
    #date='2017-06-21',
    #display_format='MMM Do, YY'
#)  
 #  ,
   
   
   
    dcc.DatePickerRange(
        id="date-picker-range2",
        start_date=dt.datetime(2018, 1, 2),
        end_date=dt.datetime(2019, 9, 23),
        min_date_allowed=dt.datetime(2018, 1, 2),
        max_date_allowed=dt.datetime(2019, 9, 23),
        end_date_placeholder_text="Select a date"
    ),
    
    html.Div([
            html.Button(id='submit-button2',
                        n_clicks=0,
                        children='Submit',
                        style={'fontSize':24,'marginLeft':'30px'}
            
            )        
            
    ],style={'display':'inline-block'}),
                          
                          
                dcc.Graph(id='my_graph2'),
                           #   figure={'data':[
                            #          {'x':[1,2],'y':[3,1]}
                                        
                            #],'layout':{'title':'Default Title'}}
                
                dcc.Graph(id="pie_judgement2"),
                
                
        ]),
            
            
            
            #tab2 ends
            
            
        dcc.Tab(label='Batch wise analysis', children=[
           
                
                 html.H1('Checkweigher Data Analysis Batch Wise'),
                
                 
                
                
                          
                          
                         dcc.DatePickerSingle(
                                 
                                 id="date-picker-range3",
    date=dt.datetime(2018, 1, 2)
   
)  
   ,dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'Batch 1', 'value': 'Batch 1'},
            {'label': 'batch 2', 'value': 'Batch 2'},
            {'label': 'Batch 3', 'value': 'Batch 3'}
        ],
        value='Batch 1'
    ),
    
                
   
   
   
    
    
    html.Div([
            html.Button(id='submit-button3',
                        n_clicks=0,
                        children='Submit',
                        style={'fontSize':24,'marginLeft':'30px'}
            
            )        
            
    ],style={'display':'inline-block'}),
            
            html.Div(id='dd-output-container'),
                          
                          
                dcc.Graph(id='my_graph3'),
                           #   figure={'data':[
                            #          {'x':[1,2],'y':[3,1]}
                                        
                            #],'layout':{'title':'Default Title'}}
                
                dcc.Graph(id="pie_judgement3"),
                
                
                
                
                
                
                
                
                
                
                
                #end
        ]),
    ]),
                        
                        
                        
                        
                        
                        ])
                ]),
        
        
        
       
        
        
        
         
        
        
        ])
         
         





#tab : upload file


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    global df
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            
            
            

            
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
       

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        }),
        
    
    
    ])
            




@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
















@app.callback(
    Output("my_graph", "figure"),
    [Input('submit-button','n_clicks')],
    [State("date-picker-range", "date"),
        
    
    
    ])


def update_graph(n_clicks,date):
   
    
    date = pd.to_datetime(date)
    
   # date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
    

    df1 = df[df.Date.between(
        dt.datetime.strftime(date, "%Y-%m-%d"),
        dt.datetime.strftime(date, "%Y-%m-%d")
    )]



    
    trace1 = go.Scatter(
        x = df1.Count,
        y = df1.Weight,
        mode = "lines",
        name = iTemperature
    )

    return {
        "data": [trace1],
        "layout": go.Layout(
            title = iTemperature,
            plot_bgcolor = colors["graphBackground"],
            paper_bgcolor = colors["graphBackground"]
        )
    }


@app.callback(Output('pie_judgement','figure'),
             [Input('submit-button','n_clicks')],
    [State("date-picker-range", "date")
        
    ])
def update_pie(n_clicks,date):
    
    date = pd.to_datetime(date)
   # end_date = pd.to_datetime(end_date)

    df2 = df[df.Date.between(
        dt.datetime.strftime(date, "%Y-%m-%d"),
        dt.datetime.strftime(date, "%Y-%m-%d")
    )]

    
    piechart = px.pie(
            
            names=df2.Judgement,
            hole=.2,
            
            )
      
    

    return (piechart)
      

  


#fuction of graph 2
    

@app.callback(
    Output("my_graph2", "figure"),
    [Input('submit-button2','n_clicks')],
    [State("date-picker-range2", "start_date"),
        State("date-picker-range2", "end_date")
    
    
    ])


def update_graph(n_clicks,start_date, end_date):
   
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    

    df1 = df[df.Date.between(
        dt.datetime.strftime(start_date, "%Y-%m-%d"),
        dt.datetime.strftime(end_date, "%Y-%m-%d")
    )]



    
    trace1 = go.Scatter(
        x = df1.Count,
        y = df1.Weight,
        mode = "lines",
        name = iTemperature
    )

    return {
        "data": [trace1],
        "layout": go.Layout(
            title = iTemperature,
            plot_bgcolor = colors["graphBackground"],
            paper_bgcolor = colors["graphBackground"]
        )
    }


@app.callback(
    Output("pie_judgement2", "figure"),
    [Input('submit-button2','n_clicks')],
    [State("date-picker-range2", "start_date"),
        State("date-picker-range2", "end_date")
        
    ])
def update_pie(n_clicks,start_date, end_date):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df2 = df[df.Date.between(
        dt.datetime.strftime(start_date, "%Y-%m-%d"),
        dt.datetime.strftime(end_date, "%Y-%m-%d")
    )]

    piechart = px.pie(
            
            names=df2.Judgement,
            hole=.2,
            
            )
      
    

    return (piechart)







#tab 3 : batch-wise

#drop down
@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)







@app.callback(
    Output("my_graph3", "figure"),
    [Input('submit-button3','n_clicks')],
    [State("date-picker-range3", "date"),
     State("demo-dropdown","value")
        
    
    
    ])


def update_graph3(n_clicks,date,value):
   
    df1=df
    
    date = pd.to_datetime(date)
    
    if value== "Batch 1":
        df['Date'] = pd.to_datetime(df['Date']) 
        mask = (df['Date'] >= date) & (df['Date'] <= date) & (df['Batch']==1)
        df1 = df.loc[mask]
    elif value== "Batch 2":
        df['Date'] = pd.to_datetime(df['Date']) 
        mask = (df['Date'] >= date) & (df['Date'] <= date) & (df['Batch']==2)
        df1 = df.loc[mask]
    elif value== "Batch 3":
        df['Date'] = pd.to_datetime(df['Date']) 
        mask = (df['Date'] >= date) & (df['Date'] <= date) & (df['Batch']==3)
        df1 = df.loc[mask]
    #else:
     #  return 'You have selected "{}"'.format(value)
    
    
    

    
    
    


    
    trace1 = go.Scatter(
        x = df1.Count,
        y = df1.Weight,
        mode = "lines",
        name = iTemperature
    )

    return {
        "data": [trace1],
        "layout": go.Layout(
            title = iTemperature,
            plot_bgcolor = colors["graphBackground"],
            paper_bgcolor = colors["graphBackground"]
        )
    }














@app.callback(Output('pie_judgement3','figure'),
             [Input('submit-button3','n_clicks')],
    [State("date-picker-range3", "date"),State("demo-dropdown","value")
        
    ])
def update_pie3(n_clicks,date,value):
    
    
    
    date = pd.to_datetime(date)
    
    if value== "Batch 1":
        df['Date'] = pd.to_datetime(df['Date']) 
        mask = (df['Date'] >= date) & (df['Date'] <= date) & (df['Batch']==1)
        df2 = df.loc[mask]
    elif value== "Batch 2":
        df['Date'] = pd.to_datetime(df['Date']) 
        mask = (df['Date'] >= date) & (df['Date'] <= date) & (df['Batch']==2)
        df2 = df.loc[mask]
    elif value== "Batch 3":
        df['Date'] = pd.to_datetime(df['Date']) 
        mask = (df['Date'] >= date) & (df['Date'] <= date) & (df['Batch']==3)
        df2 = df.loc[mask]
    #else:
     #  return 'You have selected "{}"'.format(value)
    
    

    
    piechart = px.pie(
            
            names=df2.Judgement,
            hole=.2,
            
            )
      
    

    return (piechart)
    






#Mettel Detector code:
    


def parse_contentsM(contents, filename, date):
    content_type, content_string = contents.split(',')
    global dfmettel
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            dfmettel = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            
            
            

            
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            dfmettel = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
       

        dash_table.DataTable(
            data=dfmettel.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in dfmettel.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        }),
    
        
        
    
        
    
    
        
    
    
       
   
        
    
    
    
    
    
    
    ])
            

    
    
    
    
    
    
    
    
    

@app.callback(Output('output-data-upload4', 'children'),
              [Input('upload-data2', 'contents')],
              [State('upload-data2', 'filename'),
               State('upload-data2', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contentsM(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



@app.callback(
    Output("in-temp-graphM4", "figure"),
    [Input('submit-buttonM4','n_clicks')],
    [State("date-picker-rangeM4", "start_date"),
        State("date-picker-rangeM4", "end_date")
    
    
    ])





def update_graph(n_clicks,start_date, end_date):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    dfmettel1 = dfmettel[dfmettel.Date.between(
        dt.datetime.strftime(start_date, "%Y-%m-%d"),
        dt.datetime.strftime(end_date, "%Y-%m-%d")
    )]

    trace1 = go.Scatter(
        x = dfmettel1.Date,
        y = dfmettel1.Reject_Count,
        mode = "lines",
        name = iTemperature
    )

    return {
        "data": [trace1],
        "layout": go.Layout(
            title = iTemperature,
            plot_bgcolor = colors["graphBackground"],
            paper_bgcolor = colors["graphBackground"]
        )
    }



@app.callback(
    [Output('output-sum','children'),
     Output('output-sum2','children'),
     Output('output-sum3','children'),
     Output('output-sum4','children'),
     Output('output-sum5','children')],
    [Input('submit-buttonM4','n_clicks')],
    [State("date-picker-rangeM4", "start_date"),
        State("date-picker-rangeM4", "end_date")
    
    
    ])

def update_sum(n_clicks,start_date, end_date):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    dfmettel1 = dfmettel[dfmettel.Date.between(
        dt.datetime.strftime(start_date, "%Y-%m-%d"),
        dt.datetime.strftime(end_date, "%Y-%m-%d")
    )]
            
    dfmettel3 = dfmettel1.loc[pd.to_datetime(dfmettel1['Date']).between(pd.Timestamp(start_date),pd.Timestamp(end_date)),['Duration_hrs']].sum()
    dfmettel2 = dfmettel1.loc[pd.to_datetime(dfmettel1['Date']).between(pd.Timestamp(start_date),pd.Timestamp(end_date)),['Reject_Count']].sum()
    dfmettel4= dfmettel1['Duration_hrs'].count()
    
    
    
    dfmettel1['Date'] = pd.to_datetime(dfmettel1['Date']) 
    mask = (dfmettel1['Date'] >= start_date) & (dfmettel1['Date'] <= end_date) & (dfmettel1['Machine_Status']=="SFZ")
    safe = dfmettel1.loc[mask]
    
    safez = safe.loc[pd.to_datetime(safe['Date']).between(pd.Timestamp(start_date),pd.Timestamp(end_date)),['Duration_hrs']].sum()
    safecount= safe['Duration_hrs'].count()
    
    
    
    
    return dfmettel3,dfmettel2,dfmettel3/dfmettel4,dfmettel2/dfmettel4,safez/safecount





@app.callback(
    Output("in-temp-graphM", "figure"),
     
    [Input('submit-buttonM','n_clicks')],
    [State("date-picker-rangeM", "start_date"),
        State("date-picker-rangeM", "end_date")
    
    
    ])





def update_graph(n_clicks,start_date, end_date):

    dfmettel['Date'] = pd.to_datetime(dfmettel['Date']) 
    mask = (dfmettel['Date'] >= start_date) & (dfmettel['Date'] <= end_date) & (dfmettel['Machine_Status']=="SFZ")
    dfmettel1 = dfmettel.loc[mask]
    
    trace1 = go.Scatter(
        x = dfmettel1.Date,
        y = dfmettel1.Reject_Count,
        mode = "lines",
        name = "Safe ZONE"
    )
    
    dfmettel['Date'] = pd.to_datetime(dfmettel['Date']) 
    mask = (dfmettel['Date'] >= start_date) & (dfmettel['Date'] <= end_date) & (dfmettel['Machine_Status']=="ACZ")
    dfmettel2 = dfmettel.loc[mask]
    
    trace2 = go.Scatter(
        x = dfmettel2.Date,
        y = dfmettel2.Reject_Count,
        mode = "lines",
        name = "Accepted ZONE"
    )
    
    dfmettel['Date'] = pd.to_datetime(dfmettel['Date']) 
    mask = (dfmettel['Date'] >= start_date) & (dfmettel['Date'] <= end_date) & (dfmettel['Machine_Status']=="DGZ")
    dfmettel3 = dfmettel.loc[mask]
    
    trace3 = go.Scatter(
        x = dfmettel3.Date,
        y = dfmettel3.Reject_Count,
        mode = "lines",
        name = "Danger ZONE"
    )


    return {
        "data": [trace1,trace2,trace3],
        "layout": go.Layout(
            title = iTemperature,
            plot_bgcolor = colors["graphBackground"],
            paper_bgcolor = colors["graphBackground"]
        )
    }




@app.callback(
    Output("pie-chart-statusM4", "figure"),
    [Input('submit-buttonM4','n_clicks')],
    [State("date-picker-rangeM4", "start_date"),
        State("date-picker-rangeM4", "end_date")
        
    ])

def update_pie(n_clicks,start_date, end_date):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    dfmettel2 = dfmettel[dfmettel.Date.between(
        dt.datetime.strftime(start_date, "%Y-%m-%d"),
        dt.datetime.strftime(end_date, "%Y-%m-%d")
    )]

    piechart = px.pie(
            data_frame=dfmettel2,
            names=dfmettel2.Machine_Status,
            hole=.2,
            
            )
      
    

    return (piechart)
















if __name__ == '__main__':
    app.run_server(debug=True)