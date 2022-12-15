# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:14:32 2021

@author: Alexander Schrodt (as@ing-schrodt.de)
"""

from lib.daq_arburg import signal_struct

# %%

CLIENT_ADDRESS = 'opc.tcp://host_computer:1@141.51.63.37:4880/Arburg/'
NAME_OF_MEASUREMENT = 'name_der_messung'  # Name der Messreihe, wird als Namenspräfix für die Messdatei verwendet
SLEEP_TIME = 0.25  # Zeitintervall in Sekunden, in dem nach einem neuen Zyklus gesucht werden soll


"""
Bei Messgrafiken muss die Anzahl der Signale explizit angegeben werden (num_signals=x).
Standardmäßig wird der Wert auf 1 gesetzt.
"""
SIGNALS = {
    # node ID for cycle_counter seems to vary on different machines!
    'cycle_counter': signal_struct('ns=2;i=238982'),  # this key must be included with the key name 'cycle_counter'!
    
    # all monitoring chart node IDs seem to vary on different machines
    
     #Monitoring Charts (signals must be configured on the machine)
    'monitoring_chart_1': signal_struct('ns=2;i=178052'),
    'monitoring_chart_2': signal_struct('ns=2;i=179612'),
    'monitoring_chart_3': signal_struct('ns=2;i=181172'),
    'monitoring_chart_4': signal_struct('ns=2;i=182732'),
    'monitoring_chart_5': signal_struct('ns=2;i=184292'),
    'monitoring_chart_6': signal_struct('ns=2;i=185852'),
    'monitoring_chart_7': signal_struct('ns=2;i=187412'),
    'monitoring_chart_8': signal_struct('ns=2;i=188972'),
    
    # Extended Monitoring Charts
    # not found in the node ID dump file, maybe has to be set visible by Arburg
    # Maybe they must be enabled in the machine to be visible to the outside
    # 'extended_monitoring_chart_1': signal_struct('ns=2;i=658822'),  #FIXME: not found in dump
    # 'extended_monitoring_chart_2': signal_struct('ns=2;i=658862'),  #FIXME: not found in dump
    # 'extended_monitoring_chart_3': signal_struct('ns=2;i=658902'),  #FIXME: not found in dump
    # 'extended_monitoring_chart_4': signal_struct('ns=2;i=658942'),  #FIXME: not found in dump
    
    # Measurement Charts
    'measurement_chart_1': signal_struct('ns=2;i=142912', num_signals=4),
    'measurement_chart_2': signal_struct('ns=2;i=144482', num_signals=4),
    'measurement_chart_3': signal_struct('ns=2;i=573862', num_signals=4),
    'measurement_chart_4': signal_struct('ns=2;i=574062', num_signals=4),

    # the scalar values seem to have the same node IDs on different machines
    'Zykluszeit': signal_struct('ns=2;i=140842'),
    'Zylinderheizzone 1_ist': signal_struct('ns=2;i=207272'),
    'Zylinderheizzone 2_ist': signal_struct('ns=2;i=207422'),
    'Zylinderheizzone 3_ist': signal_struct('ns=2;i=207572'),
    'Zylinderheizzone 4_ist': signal_struct('ns=2;i=207722'),
    'Zylinderheizzone 5_ist': signal_struct('ns=2;i=207872'),
    'Zylinderheizzone 1_soll': signal_struct('ns=2;i=207262'),
    'Zylinderheizzone 2_soll': signal_struct('ns=2;i=207412'),
    'Zylinderheizzone 3_soll': signal_struct('ns=2;i=207562'),
    'Zylinderheizzone 4_soll': signal_struct('ns=2;i=207712'),
    'Zylinderheizzone 5_soll': signal_struct('ns=2;i=207862'),
    'Heisskanal_soll': signal_struct('ns=2;i=148132'),
    'Heisskanal_ist': signal_struct('ns=2;i=148192'),
    'Einspritzstrom_soll': signal_struct('ns=2;i=201092'),
    'Nachdruck_Volumenstrom_1': signal_struct('ns=2;i=201172'),
    'Nachdruckhöhe_1_soll': signal_struct('ns=2;i=201292'),
    'Nachdruckzeit_1_soll': signal_struct('ns=2;i=201282'),
    'Nachdruck_Volumenstrom_2': signal_struct('ns=2;i=416782'),
    'Nachdruckhöhe_2_soll': signal_struct('ns=2;i=201332'),
    'Nachdruckzeit_2_soll': signal_struct('ns=2;i=201322'),
    'Nachdruck_Volumenstrom_3': signal_struct('ns=2;i=416792'),
    'Nachdruckhöhe_3_soll': signal_struct('ns=2;i=201372'),
    'Nachdruckzeit_3_soll': signal_struct('ns=2;i=201362'),
    'Dosiervolumen_soll': signal_struct('ns=2;i=201972'),
    'Dosiervolumen_ist': signal_struct('ns=2;i=201732'),
    'Umschaltvolumen_soll': signal_struct('ns=2;i=201112'),
    'Umschaltvolumen_ist': signal_struct('ns=2;i=202422'),
    'Massepolster': signal_struct('ns=2;i=202672'),
    #'Staudruck_ist': signal_struct('ns=2;i=202242'),
    'Staudruck_soll': signal_struct('ns=2;i=201962'),
    'maximaler Spritzdruck': signal_struct('ns=2;i=202472'),
    'Umschaltspritzdruck': signal_struct('ns=2;i=202522'),
    'Dosierzeit': signal_struct('ns=2;i=202732'),
    'Einspritzzeit': signal_struct('ns=2;i=202582'),
    'Einspritzgeschwindigkeit_soll': signal_struct('ns=2;i=201092'),
    }

# Dieses dict enthält das Signal/die Node, welche zur Erkennung eines neuen Zyklusses verwendet wird
NEW_CYCLE_SIGNAL = {
    'new_cycle_signal': signal_struct('ns=2;i=56842'),  # Node-Id ist für 'Werkzeug öffnen'
    }

#COMBINE_SIGNALS = {
    # 'Extended_Monitoring_Charts': [
    #     'extended_monitoring_chart_1',
    #     'extended_monitoring_chart_2',
    #     'extended_monitoring_chart_3',
    #     'extended_monitoring_chart_4',
    #     ],
    #'Monitoring_Charts_1': [
     #   'monitoring_chart_1',
      #  'monitoring_chart_2',
       # 'monitoring_chart_3',
    #    'monitoring_chart_4',
       # ],
    #'Monitoring_Charts_2': [
     #   'monitoring_chart_4',
      #  'monitoring_chart_5',
       # 'monitoring_chart_6',
        #],
    #'Irgendein_Name' : [
    #    'monitoring_chart_7',
    #    'monitoring_chart_8',
    #    ],
    #'Measurement_Charts': [
     #   'measurement_chart_1',
      #  'measurement_chart_2',
       # 'measurement_chart_3',
        #'measurement_chart_4',
        #],
 #   }

OLD_CYCLE_VALUE = 0  # Wert, den die new cycle node annimmt, BEVOR neuer Zyklus erkannt werden soll
NEW_CYCLE_VALUE = 1  # Wert, den die new cycle node annimmt, wenn ein neuer Zyklus erkannt werden soll
USE_FILENAME_CYCLE_PREFIX = False  # Auf True setzen, um eine neue Datei für jeden Zyklus zu erzeugen
NEW_FILE_TIMER = 'd'  # character representing the timestep at which a new file is created ('d' = 1 per day, 'h' = 1 per hour, etc.)
