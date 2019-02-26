import pandas as pd



def get_dataframe_from_state(state):
    num_kilobots = state['kilobots'].shape[0]
    num_objects = state['objects'].shape[0]

    kilobot_index, objects_index, light_index = get_multiindices(num_kilobots, num_objects)

    pass