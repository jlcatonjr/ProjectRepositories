import pandas as pd
import matplotlib.pyplot as plt
stack_dfs = pd.read_csv("StateGovFinances2009to2021.csv", index_col = ["State", "Year", "Format"]).sort_index()
panel_dfs_dict = {k:stack_dfs[stack_dfs.index.get_level_values(2)==k].reset_index().set_index(["State","Year"]).sort_index() for k in stack_dfs.index.get_level_values(2).unique()}

plot_df = panel_dfs_dict['State & local government amount'][["1GENERAL REVENUE","1EXPENDITURE", '1INDIVIDUAL INCOME', '1CORPORATE INCOME', "1PROPERTY", "1SPECIAL ASSESSMENTS"]]
plot_df.rename(columns = {k:k.replace("1", "").title() for k in plot_df.keys()}, inplace = True)
plot_df["Total Income"] = plot_df["Individual Income"].add(plot_df["Corporate Income"])
pct_df = plot_df.apply(lambda x: x.div(plot_df["General Revenue"]))
states = pct_df.index.get_level_values("State").unique()
color_map = {value: idx for idx, value in enumerate(states)}
pct_df['state'] = pct_df.index.get_level_values('State').map(color_map)
ax = pct_df.plot.scatter(x = "Property", y = "Individual Income", c = "state", colorbar = False, cmap = "viridis", figsize = (20,10))
# # Create a colorbar with the correct labels
cbar = plt.colorbar(ax.collections[0], ticks=range(len(states)))
cbar.ax.set_yticklabels(states)
plt.close()


from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import threading
import time
import asyncio
from pyppeteer import launch
import nest_asyncio

# Apply the nest_asyncio patch
nest_asyncio.apply()

# Function to run the Dash app
def run_dash_app():
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Interactive Scatter Plot"),
        html.Label("X Axis:"),
        dcc.Dropdown(
            id='x-axis',
            options=[{'label': col, 'value': col} for col in pct_df.columns],
            value=pct_df.columns[0]
        ),
        html.Label("Y Axis:"),
        dcc.Dropdown(
            id='y-axis',
            options=[{'label': col, 'value': col} for col in pct_df.columns],
            value=pct_df.columns[1]
        ),
        html.Label("Color:"),
        dcc.Dropdown(
            id='color',
            options=[{'label': col, 'value': col} for col in pct_df.columns],
            value=pct_df.columns[2]
        ),
        dcc.Graph(id='scatter-plot')
    ])

    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('x-axis', 'value'),
        Input('y-axis', 'value'),
        Input('color', 'value')
    )
    def update_scatter_plot(x_col, y_col, color_col):
        fig = px.scatter(pct_df, x=x_col, y=y_col, color=color_col)
        return fig

    app.run_server(debug=False, use_reloader=False)

# Function to save the HTML using pyppeteer
async def save_html():
    browser = await launch()
    page = await browser.newPage()
    await page.goto('http://127.0.0.1:8050')
    await asyncio.sleep(5)  # Wait for the page to load

    content = await page.content()
    with open("interactive_scatter_plot.html", "w", encoding="utf-8") as file:
        file.write(content)

    await browser.close()

if __name__ == '__main__':
    # Run the Dash app in a separate thread
    app_thread = threading.Thread(target=run_dash_app)
    app_thread.daemon = True
    app_thread.start()

    # Wait for the Dash app to start
    time.sleep(5)

    # Save the HTML
    asyncio.get_event_loop().run_until_complete(save_html())