################################################### Library/Package Imports ###################################################

### General
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

### Dashboards
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from datetime import date
from dash.dependencies import Input, Output

##################################################### Data Pre-Processing #####################################################

netflix = pd.read_csv('NetflixOriginals.csv',encoding='ISO-8859-1',parse_dates=['Premiere'])
netflix.head()

netflix['Rating Quality'] = netflix['IMDB Score'].map(lambda x: "Poor Rating" if x < 4 
                                                                              else("Mediocre Rating" if x < 6 
                                                                              else ("Above Average Rating" if x < 8 
                                                                              else " Exceptional Rating")))
netflix['Film Length Group'] = netflix['Runtime'].map(lambda x: "Short Film" if x < 60 else ("Medium Length Film" if x < 120 
                                                                                                                    else "Long Film"))
netflix['Release Year'] = netflix['Premiere'].dt.year
netflix['Release Month'] = netflix['Premiere'].dt.month
netflix['Release Day'] = netflix['Premiere'].dt.day
netflix['RMonth Text'] = netflix['Release Month'].map({1:'January',2:'February',3:'March',4:'April',5:'June',6:'July',
                                                       7:'July', 8:'August', 9:'September',10:'October',
                                                       11:'November',12:'December'})

####################################### Creating Static Visualizations for Dashboard ##########################################

### Figure 1
top10_genre = pd.DataFrame(netflix.Genre.value_counts()[:10])
top10_genre.reset_index(inplace=True)

fig_genre = px.treemap(top10_genre, path=['index'], values='Genre')
fig_genre.update_layout(title_text='Top 10 Genres',
                  title_x=0.5, 
                  title_font=dict(size=20))
fig_genre.update_traces(textinfo="label+value+percent parent")

### Figure 2
top10_languages = pd.DataFrame(netflix.Language.value_counts()[:10])
top10_languages.reset_index(inplace=True)

fig_lang = px.treemap(top10_languages, path=['index'], values='Language')
fig_lang.update_layout(title_text='Top 10 Movie Languages',
                  title_x=0.5, 
                  title_font=dict(size=20))
fig_lang.update_traces(textinfo="label+value+percent parent")

### Figure 3
releasefreq = pd.DataFrame(netflix['Release Year'].value_counts())
releasefreq.reset_index(inplace=True)

fig_relfreq = px.treemap(releasefreq, path=['index'], values='Release Year')
fig_relfreq.update_layout(title_text='Netflix Originals released from 2014 - 2021',
                  title_x=0.5, 
                  title_font=dict(size=20))
fig_relfreq.update_traces(textinfo="label+value+percent parent")\

### Figure 4
avgrating = pd.DataFrame(round(netflix.groupby('Release Year')['IMDB Score'].mean(),2))
avgrating.reset_index(inplace=True)
avgruntime = pd.DataFrame(round(netflix.groupby('Release Year')['Runtime'].mean(),2))
avgruntime.reset_index(inplace=True)
rVSrt = avgrating.merge(avgruntime,how='left', on='Release Year')
rVSrt.rename(columns = {'Runtime': 'Average Runtime','IMDB Score':'Average IMDB Score'},inplace = True)

rvsrt_fig = px.bar(rVSrt, x='Release Year', y='Average IMDB Score',
             hover_data=['Average IMDB Score', 'Average Runtime'], color='Average Runtime', height=400)
rvsrt_fig.update_layout(title_text='Average IMDB Rating of Netflix Originals from 2014 - 2021',
                  title_x=0.5, 
                  title_font=dict(size=20),
                  plot_bgcolor = 'white',
                  xaxis_title = '')
rvsrt_fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))

### Figure 5
fig_arVsrt = px.scatter(netflix, 
                        x="Premiere", 
                        y="IMDB Score", 
                        color="Film Length Group",
                        hover_data=['Title',"Genre",'IMDB Score','Runtime'])
fig_arVsrt.update_traces(marker_size=6)
fig_arVsrt.update_layout(
    title="IMDB Score for All Netflix Originals Released from 2014-2020",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='rgb(102, 102, 102)',
        tickfont_color='rgb(102, 102, 102)',
        showticklabels=True,
        ticks='outside', # ticks pointing outside
        tickcolor='rgb(102, 102, 102)',
    ),
    margin=dict(l=140, r=60, b=70, t=100),

    legend=dict(
        font_size=10,
        yanchor='top',
        xanchor='right',
        bgcolor = 'white', 
        bordercolor  = 'white',
        borderwidth = 0
    ),
    width=1300,
    height=800,
    paper_bgcolor='white',
    plot_bgcolor='white',
    hovermode='closest',
)

fig_arVsrt.update_traces(marker_size=10)

### Figure 6
x = netflix.loc[:,'Release Month']
y = netflix.loc[:,'Release Year']
z = netflix.loc[:,'Title']

df = pd.crosstab(index=y,columns=x,values=z,aggfunc='count')
df.fillna(0,inplace=True)

fig_hm = go.Figure()
fig_hm.add_trace(go.Heatmap(x=df.columns,
                         y=df.index,
                         z=df.values,
                         colorscale='rdylgn',
                         hovertemplate = 'Movies Released:%{z}',
                         name=''))
fig_hm.update_layout( title = {
                        'text': "Number of Originals Released per Month",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    title_font=dict(size=20),
                    xaxis_title = 'Month',
                    yaxis_title ='Year',
                    xaxis_type = 'category',
                    yaxis_type = 'category',
                    xaxis = dict(tickvals = list(range(1,13,1)), ticktext = ['Jan','Feb','Mar','Apr','May',
                                                                        'Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
                    height = 600, 
                    plot_bgcolor='rgba(0,0,0,0)')
fig_hm.show()




####################################################### Dash App Creation #######################################################

table = netflix[['Title', 'Genre', 'Premiere', 'Runtime', 'IMDB Score', 'Language',
       'Rating Quality', 'Film Length Group', 'Release Year', 'RMonth Text']]
table.rename(columns = {'RMonth Text':'Release Month'}, inplace = True)
table['id'] = table['Title']
table.set_index('id', inplace=True, drop=False)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = html.Div([
    html.H1('Netflix Originals Performance Analysis Dashboard', 
            style={'color': 'maroon', 'fontSize': '40px','textAlign':'center'}),
    html.P('''
            This Dashboard provides a summary of the public's reception (IMDB ratings) to all the Netflix originals
            released between 2014 and May 31st 2021. The main metrics that are considered in the analysis are the IMDB Score
            and Runtime of the films. There are other qualitative features included in the dataset that can be viewed on the
            filterable Data Table in tab 2.

            Special thanks to Luis from Kaggle (view profile here: https://www.kaggle.com/luiscorter) for scraping the dataset.
            To actually view the dataset on Kaggle please follow the link presented below: 
            ''',
        style={'textAlign':'center'}),
    html.Ul([
        html.Li(['Link to dataset here: ',
                html.A('Netflix Originals Dataset on Kaggle',
                href='https://www.kaggle.com/luiscorter/netflix-original-films-imdb-scores')]),
            ]),


    dbc.Tabs([
        # Tab 1 Creation
        dbc.Tab([
            html.H2('Netflix Originals: Summary Tab',
                     style={'color': 'maroon','fontSize': '20px','textAlign':'center'}),
            
            html.P('''
            This tab presents a summary of characteristics and performance of all netflix original movies released
            from 2014 - 2021. 
            '''),
            html.Div([
                    dcc.Graph(figure = rvsrt_fig)
                    ],style={'padding': '0px 20px 20px 20px'}),
            html.Div([
                    dcc.Graph(figure = fig_relfreq)
                    ],style={ 'width': '30%','display': 'inline-block', 'padding': '0px 20px 20px 20px'}), #, 'padding': '0px 30px 30px 30px'
            html.Div([
                    dcc.Graph(figure = fig_genre)
                    ],style={'width': '30%', 'display': 'inline-block', 'padding': '0px 20px 20px 20px'}),
            html.Div([
                    dcc.Graph(figure = fig_lang)
                    ],style={ 'width': '30%','display': 'inline-block', 'padding': '0px 20px 20px 20px'}),
            html.Div([
                    dcc.Graph(figure = fig_hm)
                    ],style={ 'padding': '0px 20px 20px 20px'}),


            ], label ='Performance Summary tab'),
        
        # Tab 2 Creation
        dbc.Tab([
            html.H1('Filterable Table of Netflix Originals',
                    style={'color': 'maroon','fontSize': '20px','textAlign':'center'}),

            html.P('''
            This table presented below contains all the data for each of the variables in the the Dataset. Feel free to play around and
            search for your favourite Netflix originals to see what their IMDB Rating is as well as other cool qualitative facts about
            the film.
            '''),

            dash_table.DataTable(
                                id='datatable-row-ids',
                                columns=[
                                        {'name': i, 'id': i, 'deletable': True} for i in table.columns
                                        # omit the id column
                                        if i != 'id'
                                ],
                                data=table.to_dict('records'),
                                editable=True,
                                filter_action="native",
                                sort_action="native",
                                sort_mode='multi',
                                row_selectable='multi',
                                row_deletable=True,
                                selected_rows=[],
                                page_action='native',
                                page_current= 0,
                                page_size= 20,
                        ),
                        html.Div(id='datatable-row-ids-container')

            ], label ='Table of All Released Netflix Originals'),
        
        # Tab 3 Creation
        dbc.Tab([
            html.Div(
                    [dcc.Slider( id='year-slider',
                        min=netflix['Release Year'].min(),
                        max=netflix['Release Year'].max(),
                        value=netflix['Release Year'].min(),
                        marks={str(year): str(year) for year in netflix['Release Year'].unique()},
                        step=None)            
                ], style={'width': '100%', 'display': 'inline-block', 'padding': '0px 20px 20px 20px'}),
            
            html.Div(
                    [ dcc.Graph(id='tm-rating_qual'),

                ],style={'width': '30%','display': 'inline-block', 'padding': '0px 20px 20px 20px'}),

            
            html.Div(
                    [ dcc.Graph(id='tm-genre'),

                ],style={'width': '30%','display': 'inline-block', 'padding': '0px 20px 20px 20px'}),

            html.Div(
                    [ dcc.Graph(id='tm-length'),

                ],style={'width': '30%','display': 'inline-block', 'padding': '0px 20px 20px 20px'}),             
            
            
            html.Div(
                    [ dcc.Graph(id='imdb summary'),

                ],style={'padding': '0px 20px 20px 20px'}),

            html.Div(
                    [ dcc.Graph(id='movie-scatterplot'),

                ],style={'display': 'inline-block', 'padding': '0px 20px 20px 20px'}),

                        

        ],label ='Year over Year Performance'), 

        # Tab 4 Creation - Future Project Development
        # dbc.Tab([], label ='Dynamic k-Means Clustering')

    ]),
])

### Call Back Functions
@app.callback(
        Output('tm-genre', 'figure'),
        Input('year-slider', 'value'))

def update_tmgenre(year):
    plotdata = netflix.loc[netflix['Release Year']== year]
    df = pd.DataFrame(plotdata.Genre.value_counts()[:10])
    df.reset_index(inplace=True)
    fig = px.treemap(df, path=['index'], values='Genre')
    fig.update_layout(title_text='Top 10 Genres of Films released in {}'.format(year),
                    title_x=0.5, 
                    title_font=dict(size=20))
    fig.update_traces(textinfo="label + value+percent parent")
    return fig

@app.callback(
        Output('tm-length', 'figure'),
        Input('year-slider', 'value'))

def update_tmgenre(year):
    plotdata = netflix.loc[netflix['Release Year']== year]
    df = pd.DataFrame(plotdata['Film Length Group'].value_counts())
    df.reset_index(inplace=True)
    fig = px.treemap(df, path=['index'], values='Film Length Group')
    fig.update_layout(title_text='Breakdown of Length of Films released in {}'.format(year),
                    title_x=0.5, 
                    title_font=dict(size=20))
    fig.update_traces(textinfo="label + value+percent parent")
    return fig


@app.callback(
        Output('tm-rating_qual', 'figure'),
        Input('year-slider', 'value'))

def update_tmgenre(year):
    plotdata = netflix.loc[netflix['Release Year']== year]
    df = pd.DataFrame(plotdata['Rating Quality'].value_counts())
    df.reset_index(inplace=True)
    fig = px.treemap(df, path=['index'], values='Rating Quality')
    fig.update_layout(title_text='Breakdown of Rating Quality of Films released in {}'.format(year),
                    title_x=0.5, 
                    title_font=dict(size=20))
    fig.update_traces(textinfo="label + value+percent parent")
    return fig



@app.callback(
        Output('imdb summary', 'figure'),
        Input('year-slider', 'value'))
def update_figure(year):
        
        plotdata = netflix.loc[netflix['Release Year']== year].sort_values(by='IMDB Score', ascending = False)
        data = plotdata[0:20]
        data.reset_index(inplace=True)
        
        fig = px.bar(data[0:10],x='IMDB Score', y='Title', orientation = 'h')


        fig = px.bar(data[0:10], x='IMDB Score', y='Title', hover_data=['IMDB Score', 'Film Length Group'],
                    color='Film Length Group')
        fig.update_layout(title_text='Top 10 Netflix Originals Released in {} w/ Highest IMDB Scores'.format(year),
                        title_x=0.5, 
                        title_font=dict(size=20),
                        plot_bgcolor = 'white',
                        xaxis_title = 'IMDB Score',
                        yaxis_title = '')
        return fig

@app.callback(
        Output('movie-scatterplot', 'figure'),
        Input('year-slider', 'value'))
def update_scatter(year):
    plotdata = netflix.loc[netflix['Release Year']== year]
    fig = px.scatter(plotdata, 
                        x="Premiere", 
                        y="IMDB Score", 
                        color="Film Length Group",
                        hover_data=['Title',"Genre",'IMDB Score','Runtime'])
    fig.update_traces(marker_size=6)
    fig.update_layout(
        title = {
                    'text': "IMDB Score for All Netflix Originals Released in {}".format(year),
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                    },
        title_font=dict(size=20),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgb(102, 102, 102)',
            tickfont_color='rgb(102, 102, 102)',
            showticklabels=True,
            ticks='outside',
            tickcolor='rgb(102, 102, 102)',
        ),
        margin=dict(l=140, r=60, b=70, t=100), 
        
        legend=dict(
            font_size=10,
            yanchor='top',
            xanchor='right',
            bgcolor = 'white', 
            bordercolor  = 'white',
            borderwidth = 0
        ),
        width=1800,
        height= 650,
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
    )
    fig.update_traces(marker_size=10)
    return fig


@app.callback(
    Output('datatable-row-ids-container', 'children'),
    Input('datatable-row-ids', 'derived_virtual_row_ids'),
    Input('datatable-row-ids', 'selected_row_ids'),
    Input('datatable-row-ids', 'active_cell'))
def update_graphs(row_ids, selected_row_ids, active_cell):
    selected_id_set = set(selected_row_ids or [])

    if row_ids is None:
        dff = table
        # pandas Series works enough like a list for this to be OK
        row_ids = table['id']
    else:
        dff = table.loc[row_ids]

    active_row_id = active_cell['row_id'] if active_cell else None

    colors = ['#FF69B4' if id == active_row_id
              else '#7FDBFF' if id in selected_id_set
              else '#0074D9'
              for id in row_ids]
    return [
        dcc.Graph(
            id=column + '--row-ids',
            figure={
                'data': [
                    {
                        'x': table['Title'],
                        'y': table[column],
                        'type': 'bar',
                        'marker': {'color': colors},
                    }
                ],
                'layout': {
                    'xaxis': {'automargin': True},
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': column}
                    },
                    'height': 250,
                    'margin': {'t': 10, 'l': 10, 'r': 10},
                },
            },
        )
        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        for column in ['Title', 'Genre', 'Premiere', 'Runtime', 'IMDB Score', 'Language', 'Rating Quality', 'Film Length Group', 'Release Year', 'Release Month'] if column in table
    ]
if __name__ == '__main__':
    app.run_server(debug=True)
