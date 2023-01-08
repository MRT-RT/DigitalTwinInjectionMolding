DataManager.py
Falls noch keine h5-Datei existiert, in der die Daten in dem die für die Modellbildung 
erforderlichen Format vorliegen, dieses Skript zuerst aufrufen. Es liest alle Daten aus
der h5-Datei in die opc_daq_main.py schreibt aus, konvertiert sie in ein entsprechendes 
Format und speichert sie in einer anderen vom Nutzer spezifizierten h5-Datei (target_hdf5)

Live_Model_Reestimation.py
Nimmt die für die Modellbildung vorgesehenen Daten aus der target_hdf5 und bildet ein statisches
Modell. In diesem Fall ein MLP mit einer Schicht und 6 Neuronen. Das Skript erzeugt in einem spezi-
fizierten Pfad 3 Dateien:
	-overview: 		Ein DataFrame der den Wert der Kostenfunktion auf Trainings-, und Validierungsdaten 
				sowie die damit assoziierten Parameter für jeden Optimierungsdurchlauf. Außerdem 
				Best Fit Rate auf den Validierungsdaten.
	-models:		Ein Dictionary welches die Modellobjekte selbst mit den bereits zugewiesenen opti-
				mierten Modellparametern enthält. So kann ein optimiertes Modell direkt geladen 
				werden, ohne dass man das Modellobjekt erst generieren muss und ihm dann die Parameter
				aus overview zuweist.
	-live_models:	Enthält die 10 besten Modelle aus models. Auf diese Datei greift DigitalTwin.py 
				zu.
Existiert noch keine live_models.pkl oder soll die vorhandene Datei überschrieben werden, muss dieses 
Skript als nächstes gestartet werden.
In Zeile 119 ist go = False gesetzt, sodass die while-Schleife nur einmal ausgeführt wird. Sollen ständig neue
Modelle basierend auf den Daten in target_hdf5 geschätzt werden, muss go = True gesetzt werden.


DigitalTwin.py
Startet das Programm zur Prädiktion der Qualitätsgrößen und Berechnung der optimalen Prozessparameter.
Drei Pfade müssen vom Nutzer angegeben werden:

	- source_live_h5: Die h5-Datei, in die opc_daq_main.py die Live-Daten schreibt. 
	- dm_path: Der Pfad zu der Datei, in der sich der Data Manager befindet
	- model_path: Der Pfad zu der live_models.pkl, in welcher sich die zu verwendenden Modelle befinden

Das Programm legt zwei Dictionaries an, die in jedem Zyklus um einen Eintrag erweitert werden: mb_save und opt_save.
In mb_save wird die von mir so bezeichnete Modellbank gespeichert. Diese umfasst die Modelle sowie die von den Modellen
gemachten Prädiktionen und Informationen welche Güte die Modelle haben (BFR)
opt_save beinhaltet die berechneten optimalen Prozessparameter als DataFrame. 
 





