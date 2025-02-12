import pandas as pd
import plotly.express as px
from plotly.offline import plot
import re

def compute_head_cont(files, ref, saveindf = False, waterlevel = False):
    """
    waterlevel: bool, optional
        If False, the column "WaterLevel" in df will be interpreted as the water column above the diver.
        If True, it will be interpreted as the water depth
    """
    if saveindf:
        bigdf = pd.DataFrame()
    for row in ref.iterrows():
        if waterlevel:
            for file in files:
                if row[1].id_punto in file:
                    df = pd.read_excel(file)
                    df['prof_diver'] = row[1].prof_diver
                    df['quota_ref'] = row[1].quota_ref
                    df['soggiacenza'] = df.WaterLevel
                    df['quota_falda'] = df.quota_ref - df.soggiacenza
        else:
            for file in files:
                if row[1].id_punto in file:
                    df = pd.read_excel(file)
                    df['prof_diver'] = row[1].prof_diver
                    df['quota_ref'] = row[1].quota_ref
                    df['soggiacenza'] = df.prof_diver - df.WaterLevel
                    df['quota_falda'] = df.quota_ref - df.soggiacenza
        if saveindf:
            bigdf = pd.concat([bigdf, df.loc[:, ['MonitoringPoint', 'TimeStamp', 'Temperature', 'quota_falda']]])
    if saveindf:
        bigdf.columns = ['MonitoringPoint', 'TimeStamp', 'Temperature', 'quota_falda']
        bigdf.reset_index(inplace=True, drop=True)
        return bigdf
    
def load_keller_csv(file):
    df = pd.read_csv(file, skiprows=8).iloc[:,2:]

    df.Time = [re.sub(' PM', '', (re.sub(' AM', '', t))) for t in df.Time]

    df['datetime'] = df.Date + ' ' + df.Time
    df.Date = pd.to_datetime(df.Date, format = '%d/%m/%Y')
    df.datetime = pd.to_datetime(df.datetime, format = '%d/%m/%Y %H:%M:%S')

    for col in df.columns[2:7]:
        if df[col].dtype != 'float':
            df[col] = [re.sub(',', '.', v) for v in df[col]]
            df[col] = pd.to_numeric(df[col])

    return df

def interactive_TS_visualization(df, xlab = 'X', ylab = 'Y', file = 'temp.html',
                                 plottype = 'line', legend_title = "Variables",
                                 markers = False, ret = False, **kwargs):
    """
    Function to interactively plot a pandas.DataFrame
    Needs plotly and plotly.express

    df: pandas.DataFrame
        The DataFrame to plot
    xlab, ylab: string, optional
        The x and y axes labels. Default is X, Y
    file: string, optional
        The name of the output .html file containing the plot.
        Default is temp.html
    plottype: string, optional
        If == line, plots a line plot, otherwise a scatterplot.
        Default is line
    markers: bool, optional
        If True, plots markers in the line plot. Default is False
    ret: bool, optional
        If True, returns a plotly.express.line/scatter and doesn't save a .html file.
        Default is False
    kwargs: dictionary
        Additional parameters to be passed to plotly.express.line/scatter.update_layout

    Returns
    none or plotly.express.line/scatter
    """
    if plottype == 'line':
        figure = px.line(df, markers = markers)
    else:
        figure = px.scatter(df)
    figure.update_layout(
        xaxis_title = xlab,
        yaxis_title = ylab,
        legend_title = legend_title,
        **kwargs
        )
    if ret:
        return figure
    else:
        plot(figure, filename = file)