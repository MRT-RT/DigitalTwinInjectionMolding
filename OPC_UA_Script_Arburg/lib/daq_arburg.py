# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:20:05 2021

@author: alexs
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unicodedata
import re
import logging

from opcua import Client

# %% config logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
console_log = logging.StreamHandler()
console_log.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
console_log.setFormatter(formatter)
# add handle to logger
logger.addHandler(console_log)

# %% Class definitions

class signal_struct:
    def __init__(self, node_id, num_signals=1):
        self.node_id = node_id
        self.num_signals = num_signals
        

class NewCycleWatcher:
    """ A simple class, set to watch its variable. """
    def __init__(self, init_value, new_cycle_value, old_cycle_value):
        self.value = init_value
        self.new_cycle_value = new_cycle_value
        self.old_cycle_value = old_cycle_value

    def set_value(self, new_value):
        if self.value != new_value:
            last_value = self.value
            self.value = new_value
            if last_value == self.old_cycle_value and new_value == self.new_cycle_value:
                return True
        return False
        

# %% function definitions

def time_to_name(timestamp=None, pre=None, ext=None, smallest="s") -> str:
    """
    Creates a filename from the current time and a given extension ext.
    A prefix string can be given with pre.
    The formatting of the timestamp uses the smallest unit according to the smallest
    argument.
    ========================
    smallest:
        "s": seconds
        "m": minutes
        "d": days
        "mon": month
        "y": year
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()

    if ext is None:
        ext = ""

    if pre is None:
        pre = ""
    else:
        pre = str(pre)

    if smallest == "s":
        name_str = "{}messung-{:02d}{:02d}{:02d}{:02d}{:02d}{:02d}{}".format(
            pre,
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
            ext,
            )
    elif smallest == "m":
        name_str = "{}messung-{:02d}{:02d}{:02d}{:02d}{:02d}{}".format(
            pre,
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            ext,
            )
    elif smallest == "d":
        name_str = "{}messung-{:02d}{:02d}{:02d}{}".format(
            pre,
            timestamp.year,
            timestamp.month,
            timestamp.day,
            ext,
            )
    elif smallest == "mon":
        name_str = "{}messung-{:02d}{:02d}{}".format(
            pre,
            timestamp.year,
            timestamp.month,
            ext,
            )
    elif smallest == "y":
        name_str = "{}messung-{:02d}{}".format(
            pre,
            timestamp.year,
            ext,
            )

    return name_str


def get_node(client, node_str):
    return client.get_node(node_str)


def combine_single_signal_times(data, pairing) -> np.array:
    combined_times = np.array([])
    for key in pairing:
        try:
            combined_times = np.append(combined_times, data[key].time)
        except KeyError as e:
            logger.exception(f"Key {e} not found. Empty time list will be returned")

    return combined_times


def combine_single_signal_values(data, pairing) -> np.array:
    for idx, key in enumerate(pairing):
        try:
            if idx == 0:
                combined_data = data[key].values
            else:
                combined_data = np.append(
                    combined_data,
                    data[key].values,
                    axis=0,
                    )
        except KeyError as e:
            logger.exception(f"Key {e} not found. Empty dataset will be returned")

    return combined_data


def combine_signals(data, combination_list, del_combined=True):

    combined_data = dict()
    for key, d in data.items():
        combined_data[key] = d.create_DataFrame()

    all_paired_keys = []
    for key, pairing in combination_list.items():
        all_paired_keys.extend(pairing)

        combined_values = combine_single_signal_values(data, pairing)
        combined_times = combine_single_signal_times(data, pairing)
        time_and_data = np.append(combined_times.reshape(-1, 1), combined_values, axis=1)

        col_names = ["time"]
        col_names.extend([key])
        col_names.extend(["state"])

        combined_data[key] = pd.DataFrame(
            data=time_and_data,
            columns=col_names,
            )

    if del_combined:
        for key in list(set(all_paired_keys)):
            # don't del new keys that are named like old ones
            if key not in list(combination_list.keys()):
                try:
                    del combined_data[key]
                except KeyError:
                    pass

    return combined_data


class NodeData():
    """
    Class that can hold all relevant data read from one node at a certain time. Also
    a saving option is included to store the Data in a hdf file.
    """
    def __init__(self, is_array_node):
        self.is_array_node = is_array_node
        self.name = None

    def save_to_hdf(self, filename=None, root_key=""):
        if filename is None:
            filename = time_to_name(ext=".h5")
        df = self.create_DataFrame()
        key = root_key + "/" + self.display_name.replace("-", "_")
        df.to_hdf(
            filename,
            key=key,
            )

    def create_DataFrame(self):
        if self.is_array_node:
            values = np.append(self.time.reshape(-1, 1), self.values, axis=1)
            value_headers = \
                [f"{self.name}_{i}" for i
                 in np.arange(np.shape(self.values)[1] - 1)]
            columns = ["time"]
            columns.extend(value_headers)
            columns.extend(["states"])
        else:
            values = [self.values]
            columns = [self.name]

        df = pd.DataFrame(
            data=values,
            columns=columns,
            )
        return df


class NodeAccess():
    """
    Class that can hold one node and provides several accessing options for this node.
    """
    def __init__(self, node, num_signals=None):
        if num_signals is None:
            num_signals = 1

        self.num_signals = num_signals
        self.node = node
        self.is_array_node = self.is_node_array_valued()
        self.is_scalar_value = not self.is_array_node
        self.data = None

    def is_node_array_valued(self) -> bool:
        r = self.node.get_value_rank().value

        if r == -1:
            is_array_node = False
        elif r == 2:
            is_array_node = True

        return is_array_node

    def get_server_timestamp(self):
        return self.node.get_data_value().ServerTimestamp

    def update_stored_data(self):
        self.data = self.get_all_data()
        self.name = self.data.display_name
        self.node_id = self.data.node_id

    def get_all_data(self) -> dict:
        """
        This method gets the current data of this node. Automatically creates
        a correct time array if the node holds an array. This time array will be
        calculated from the offset time and sampling rate.
        """
        current_data = NodeData(self.is_array_node)
        current_data.display_name = self.get_node_name()
        current_data.node_id = self.get_node_id()
        current_data.server_timestamp = self.get_server_timestamp()

        if self.is_array_node:
            values = self.get_values_from_array_node()
            states = self.get_states_from_array_node()

            sample_time = self.get_node_sample_time()
            time_max = self.get_node_max_time()
            time_offset = self.get_node_time_offset()

            current_data.dt = sample_time
            current_data.num_samples = self.get_node_num_samples()
            current_data.time_offset = time_offset
            current_data.time = np.arange(
                time_offset,
                time_offset + time_max + sample_time,
                sample_time
                )

            current_data.values = np.append(values, states, axis=0).T

        else:
            current_data.values = self.get_values_from_scalar_node()

        return current_data

    def get_node_max_time(self):
        num_samples = self.get_node_num_samples()
        dt = self.get_node_sample_time()
        return dt * (num_samples - 1)

    def get_values_from_scalar_node(self):
        return self.node.get_data_value().Value.Value

    def get_values_from_array_node(self):
        node_values = self.node.get_data_value().Value.Value
        array_values = []
        index_signal_1 = 10
        for loop_counter in range(self.num_signals):
            ndx = index_signal_1 + loop_counter * 3
            array_values.append(node_values[ndx].Value)

        return np.asarray(array_values)

    def get_states_from_array_node(self):
        node_states = [self.node.get_data_value().Value.Value[7].Value]
        return np.asarray(node_states)

    def get_node_name(self):
        return self.node.get_display_name().Text

    def get_node_id(self):
        return self.node.nodeid.Identifier

    def get_node_sample_time(self):
        return self.node.get_data_value().Value.Value[3].Value

    def get_node_num_samples(self):
        return self.node.get_data_value().Value.Value[5].Value

    def get_node_time_offset(self):
        return self.node.get_data_value().Value.Value[2].Value

    def save_to_hdf(self, filename=None):
        self.data.save_to_hdf(filename=filename)


def create_nodes(signals_dict: dict, client: Client) -> dict:
    """
    Create a dict with all nodes defined in signals_dict.
    
    The keys in this dict are the keys of signals_dict, the values are
    NodeAccess elements which can be used to access the data stream of this
    node.
    
    Parameters
    ----------
    signals_dict : dict
        Dictionary holding several pairs of key: signal_struct.
    client : Client
        OPC UA Client to connect to.

    Returns
    -------
    dict

    """
    
    nodes = dict()
    for signal_name, id_str in signals_dict.items():
        nodes[signal_name] = NodeAccess(
            get_node(client, id_str.node_id),
            num_signals=id_str.num_signals,
            )
    return nodes


def plot_df_over_time(df, ax=None):
    """
    helper function for quick plotting of data over time
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # plotting data assuming the first column holds the time and the others hold data
    ax.plot(df.iloc[:, 0], df.iloc[:, 1:])
    return ax



def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')