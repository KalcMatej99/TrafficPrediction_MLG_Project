import argparse
import os.path

import plotly.express as px
import sys

import pandas as pd
import numpy as np


def plot_graph(draw: bool):
    # Obtain data from csv
    counters = pd.read_csv("../data/counters_non_temporal_aggregated_data.csv")

    # This constructs a dataframe from which we can draw connections between counters
    edge_longitude = []
    edge_latitude = []
    edge_type = []
    for counter_direction in counters['id']:
        counter_predecessors = str(counters[counters['id']==counter_direction]['predecessors'].iloc[0])
        if counter_predecessors == 'nan':
            continue
        else:
            counter_predecessors = counter_predecessors.split(',')

        for predecessor_id in counter_predecessors:
            if predecessor_id in counters['id'].unique():
                edge_longitude.extend([
                    counters[counters['id']==counter_direction]['longitude'].values[0], 
                    counters[counters['id']==predecessor_id]['longitude'].values[0], 
                    None
                ])

                edge_latitude.extend([
                    counters[counters['id']==counter_direction]['latitude'].values[0], 
                    counters[counters['id']==predecessor_id]['latitude'].values[0], 
                    None
                ])
                edge_type.extend(["Connection" for _ in range(3)])
    edges_df = pd.DataFrame()
    edges_df["longitude"] = edge_longitude
    edges_df["latitude"] = edge_latitude
    edges_df["type"] = edge_type

    # Making sure cluster value is viewed as a category and not a numerical value
    #counters_df["cluster"] = ["Cluster " + str(value) for value in counters_df[cluster_type].values]
    
    # Apply crea access token for mapbox
    px.set_mapbox_access_token(
        "pk.eyJ1IjoiY3JlYWFuZHJheiIsImEiOiJjbDN2aWQ2dHExczZ1M2JsdHltdnhoYXAxIn0.z1N9hTHvqi6F5kIrLtjUBg")

    #def to_string(rgb: Tuple[int]) -> str:
    #    rgb_int_list = [int(x * 256) for x in rgb]  # To integers
    #    rgb_int_list.pop()  # Remove alpha value
    #    return "rgb(%d,%d,%d)" % tuple(rgb_int_list)  # To hex string

    # Draw scatter of counters
    #number_of_clusters = np.max(counters_df[cluster_type])
    #colors = plt.cm.get_cmap("hsv", number_of_clusters + 2)
    #colors = [to_string(colors(i)) for i in range(number_of_clusters + 2)]
    fig2 = px.scatter_mapbox(counters,
                                lat='latitude',
                                lon='longitude',
                                hover_name="opis",
                                text="id",
                                size=np.ones((len(counters))) * 10,
                                #color="cluster",
                                size_max=17,
                                #title=cluster_type,
                                #color_discrete_sequence=colors
                                )

    # Draw the connections
    fig = px.line_mapbox(edges_df,
                            lat="latitude",
                            lon="longitude",
                            color="type",
                            #title=cluster_type,
                            color_discrete_sequence=["#000000"]
                            )
    for data in fig2.data:
        fig.add_trace(data)  # Adds the scatter trace to the first figure

    # Draw cities
    '''cities = pd.DataFrame(CITIES.values())
    cities["type"] = "City"
    fig3 = px.scatter_mapbox(cities,
                                lat='latitude',
                                lon='longitude',
                                hover_name="city",
                                size=np.ones((len(cities))) * 10,
                                size_max=12,
                                color="type",
                                title="30 biggest Slovenian cities",
                                color_discrete_sequence=["#000000"]
                                )
    for data in fig3.data:
        fig.add_trace(data)  # Adds the scatter trace to the first figure'''

    # Update layout to streets style and show
    fig.update_layout(mapbox_style="streets")

    # Save plot to file
    plot_path = os.path.join("./data/", "map_view.html")
    html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    file = open(plot_path, "w")
    file.write(html)
    file.close()

    # Show plot in browser
    if draw:
        fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise graph of counters on a map.')
    #parser.add_argument('experiment_name', type=str, help='Name of the experiment.')
    #args = parser.parse_args()

    sys.exit(plot_graph(True))
