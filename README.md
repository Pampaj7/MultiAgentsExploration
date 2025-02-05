# Multi-Agent Exploration

## Descrizione
Questo progetto implementa una simulazione di esplorazione basata su **sistemi multi-agente**, utilizzando:
- **Celle di Voronoi** per suddividere l'area esplorabile.
- **Entropia** per guidare l'esplorazione verso aree meno conosciute.
- **Aggiornamento probabilistico della mappa** con una soglia per definire le celle libere o occupate.
- **Algoritmi di percorso minimo** (*A*, BFS) per la navigazione efficiente.
- **Visualizzazione in tempo reale** con Matplotlib.

~~~

## 📂 Struttura del Progetto
/multiagent-exploration

│── main.py # Punto di ingresso del programma.  
│── config.py # Configurazioni generali.  
│── environment.py # Classe per la gestione della mappa  
│── agent.py # Classe per la gestione degli agenti  
│── exploration.py # Algoritmi di esplorazione  
│── pathfinding.py # Algoritmi di percorso minimo  
│── visualization.py # Visualizzazione con Matplotlib  
│── utils.py # Funzioni di utilità  
│── requirements.txt # Dipendenze  
└── README.md # Documentazione  

~~~

## 🛠️ Dipendenze

Per eseguire questo progetto, assicurati di avere Python 3.x installato.

### Installazione delle Dipendenze

Per installare tutte le dipendenze necessarie, esegui il seguente comando nel terminale:

~~~bash
pip install -r requirements.txt
~~~

### Dipendenze del Progetto

Le principali librerie e pacchetti necessari per il corretto funzionamento del progetto sono:

- `numpy`: Per operazioni numeriche avanzate.
- `matplotlib`: Per la visualizzazione della simulazione in tempo reale.
- `scipy`: Per calcoli scientifici e ottimizzazioni.
- `networkx`: Per gestire grafi e algoritmi di percorso.
- `pandas`: Per la gestione dei dati tabulari (se necessario).
- `scikit-learn`: Per algoritmi di machine learning (se necessari).

Per ulteriori dettagli, consultare il file `requirements.txt` incluso nel progetto.

~~~
