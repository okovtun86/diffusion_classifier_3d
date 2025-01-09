# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:29:20 2025

@author: okovt
"""

#%%
import pickle
import xml.etree.ElementTree as ET

with open('X_3D_v2.pkl', 'rb') as f:
    simulated_data = pickle.load(f)  

curr_tracks = simulated_data['Anomalous']

#%%

root = ET.Element('TrackMate')
model = ET.SubElement(root, 'Model')
all_spots = ET.SubElement(model, 'AllSpots')

spot_id = 0
track_id = 0
track_data = []

for trajectory in curr_tracks:
    frame_number = 0
    track = []  

    for point in trajectory:
        x, y, z = point  
        
        spots_in_frame = all_spots.find(f"SpotsInFrame[@frame='{frame_number}']")
        if spots_in_frame is None:
            spots_in_frame = ET.SubElement(all_spots, 'SpotsInFrame', {'frame': str(frame_number)})
        
        spot_attrs = {
            'ID': str(spot_id),
            'POSITION_X': str(x),
            'POSITION_Y': str(y),
            'POSITION_Z': str(z),
            'POSITION_T': str(frame_number),
            'QUALITY': '1.0',  
        }
        ET.SubElement(spots_in_frame, 'Spot', spot_attrs)
        track.append(spot_id)  
        spot_id += 1
        frame_number += 1

    track_data.append((track_id, track))  
    track_id += 1

all_tracks = ET.SubElement(model, 'AllTracks')

for track_id, track in track_data:
    track_element = ET.SubElement(all_tracks, 'Track', {'TRACK_ID': str(track_id), 'name': f'Track_{track_id}'})
    for i in range(len(track) - 1):
        edge_attrs = {
            'SPOT_SOURCE_ID': str(track[i]),
            'SPOT_TARGET_ID': str(track[i + 1]),
            'EDGE_TIME': str(i), 
        }
        ET.SubElement(track_element, 'Edge', edge_attrs)

tree = ET.ElementTree(root)
tree.write('anomalous_tracks_for_imaris.xml', encoding='UTF-8', xml_declaration=True)

print("TrackMate-compatible XML file saved as 'simulated_trajectories.xml'.")

