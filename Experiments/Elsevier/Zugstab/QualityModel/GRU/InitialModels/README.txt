Ergebnisse sind identisch mit /Experiments/GRU_CCTA, wurden lediglich konvertiert:

GRU_c1_3sub_all.pkl ist nun über loss_val iterierbar, das war vorher nicht möglich, da loss_val ein Casadi-DM war
GRU_c1_3sub_all.dict ist ein aus dem DataFrame GRU_c1_3sub_all.pkl erzeugtes Dictionary, dass zusätzlich zu den Modellparametern auch die kompletten Modelle enthält. So können diese direkt geladen und ausgewertet werden, ohne das Modell vorher noch mit der entsprechenden Struktur zu erzeugen und dann erst die Parameter zuzuweisen.


