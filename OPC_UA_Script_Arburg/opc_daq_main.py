# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:14:32 2021

@author: Alexander Schrodt (as@ing-schrodt.de)
"""

from opcua import Client
from time import sleep
import logging

from lib.daq_arburg import create_nodes
from lib.daq_arburg import combine_signals
from lib.daq_arburg import time_to_name
from lib.daq_arburg import slugify
from lib.daq_arburg import NewCycleWatcher

import Arburg_320_config as cfg
#import Arburg_470_config as cfg
#import Arburg_520_config as cfg
# %% config logging
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

# create console handler and set level to debug
console_log = logging.StreamHandler()
console_log.setLevel(logging_level)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
console_log.setFormatter(formatter)
# add handle to logger
logger.addHandler(console_log)


# %% main function


if __name__ == '__main__':
    
    # Sicherstellen schöner keys als Signalnamen (keine Leerzeichen, etc.)
    signals = {slugify(key): val for key, val in cfg.SIGNALS.items()}
    
    # Allrounder 520 E
    client = Client(cfg.CLIENT_ADDRESS)
    sleep_time = cfg.SLEEP_TIME
    
    new_cycle = False

    cycle_state = NewCycleWatcher(
            init_value=-1,
            new_cycle_value=cfg.NEW_CYCLE_VALUE,
            old_cycle_value=cfg.OLD_CYCLE_VALUE,
            )

    try:
        client.connect()
        
        # Create the node to access the new cycle signal
        new_cycle_node = create_nodes(cfg.NEW_CYCLE_SIGNAL, client)
        
        # create all nodes
        nodes = create_nodes(signals, client)

        while True:
            sleep(sleep_time)
            logger.debug('Waiting for new cycle.')

            # THE FOLLOWING 2 LINES ARE ONLY FOR SINGLE CYCLE USE
            # Remove for productive operation
            # if new_cycle == False:
            #     break

            new_cycle_signal_value = new_cycle_node['new_cycle_signal'].get_values_from_scalar_node()

            new_cycle = cycle_state.set_value(new_cycle_signal_value)

            if new_cycle:
                logger.info('New cycle found. Reading data from Machine now.')

                new_cycle = False  # wird nicht wirklich gebraucht, nur zur Sicherheit

                # get data from all nodes
                raw_data = dict()
                for name, node in nodes.items():
                    logger.debug(f'Getting value of node: {name}: {node}')
                    node.update_stored_data()

                    raw_data[name] = node.data
                    raw_data[name].name = name



                # use the first timestamp that can be found in raw_data as overall timestamp
                for d in raw_data.values():
                    timestamp = d.server_timestamp
                    break

                cycle_counter = raw_data['cycle_counter'].values

                # combine the raw data according to combine_signal_list
                #combined_data = combine_signals(raw_data, cfg.COMBINE_SIGNALS)
                
                if cfg.USE_FILENAME_CYCLE_PREFIX:
                    file_prefix = f"{cfg.NAME_OF_MEASUREMENT}-cycle-{cycle_counter:07d}-"
                else:
                    file_prefix = cfg.NAME_OF_MEASUREMENT

                filename = time_to_name(
                    timestamp=timestamp,
                    pre=file_prefix,
                    smallest=cfg.NEW_FILE_TIMER,
                    ext=".h5",
                    )

                """
                combined data and raw_data will both be stored in the same hdf
                file with different keys
                """
                root_key = f"cycle_{cycle_counter}"

                for data in raw_data.values():
                    data.save_to_hdf(filename, root_key=root_key)


    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        raw_data = dict()
        logger.exception(message)

    finally:
        client.disconnect()
