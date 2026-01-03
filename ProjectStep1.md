# Projektbeskrivning

ValueMeridian: Maskininlärningsbaserad värdering av småhus

## Problembeskrivning

Syftet med projektet är att undersöka om maskininlärning kan användas för att förutsäga slutpris på bostäder (småhus/villor) baserat på historiska försäljningsdata.

Problemet jag vill lösa är:

Givet ett antal egenskaper hos ett hus (t.ex. boyta, tomtstorlek, byggår och geografiskt läge), kan en modell uppskatta ett rimligt marknadsvärde baserat på tidigare försäljningar i området?

Projektet fokuserar på Partille kommun, som är intressant eftersom den ligger nära Göteborg och har tydliga geografiska skillnader i prisbild trots relativt liten yta.

Målet är inte att skapa en kommersiell värderingstjänst, utan att:

- förstå hur datakvalitet, feature engineering och modellval påverkar resultatet
- bygga och utvärdera en regressionsmodell i ett realistiskt sammanhang

## Datakällor

Den huvudsakliga datakällan är Boolis GraphQL-API, där historiska försäljningar kan hämtas.

Datat består av:

- sålda villor, radhus och parhus
- inom ett definierat geografiskt område (Partille, areaId 268)
-	från och med år 2010

### För varje försäljning kan följande typer av information hämtas:

- slutpris
- utgångspris
- pris per kvadratmeter
- boyta
- tomtstorlek
- antal rum
- byggår
- driftkostnad
- geografiska koordinater (latitud/longitud)
- mäklar- och byråinformation

Datat hämtas automatiskt via ett eget Python-skript och lagras lokalt i CSV-format för vidare analys i Pandas.

### Databeskrivning och kvalitet

Datamängden består av flera tusen rader (beroende på filtrering och tidsintervall), där varje rad motsvarar en såld bostad.

Vid en första genomgång av datat observeras följande:

#### Null-värden förekommer, exempelvis: 

- saknad tomtstorlek
- saknad driftkostnad
- ibland saknade koordinater

#### Datatyper:

- numeriska: priser, ytor, rum, byggår
- kategoriska: hustyp, mäklarbyrå
- geografiska: latitud/longitud

#### Extrema värden förekommer:

- mycket stora tomter
- mycket höga eller låga kvadratmeterpriser

Dessa aspekter behöver hanteras genom:
- filtrering eller imputering av null-värden
- normalisering eller robusta modeller
- eventuell borttagning av uppenbara outliers


### Val av fält (features)

Följande fält bedöms som särskilt relevanta för värdering:

- boyta (living area)
- tomtstorlek
- antal rum
- byggår
- driftkostnad
- geografiskt läge (latitud/longitud)
- härledda geografiska mått, t.ex.:
- avstånd till Göteborg
- avstånd till Partille centrum
- avstånd till större trafikleder eller arbetsplatser

Vissa attribut som visas på Boolis webbplats (t.ex. biyta, eldstad, uteplats) finns inte strukturerat i API:t. I detta projekt kommer dessa därför antingen:

- att utelämnas, eller
- approximeras via proxy-features (t.ex. byggår, tomtstorlek, textanalys om tillgängligt)

Alla valda fält kommer att konverteras till numeriskt format genom:

- skalning/normalisering
- encoding av kategoriska variabler (t.ex. one-hot eller target encoding)


### Typ av problem

Detta är ett regressionsproblem, eftersom målet är att förutsäga ett kontinuerligt värde (slutpris).

Datat är labeled, eftersom det finns historiska försäljningspriser som kan användas som facit vid träning.

Projektet kommer därför att använda supervised learning.

Reinforcement learning och deep neural networks bedöms inte som lämpliga i detta skede, eftersom:

- problemet är statiskt (inte sekventiellt)
- mängden data är relativt begränsad
- tolkbarhet är viktigare än extrem modellkomplexitet

### Planerad modellering (översikt)

I senare steg planeras följande:

- baseline-modell (t.ex. linjär regression)
- träd-baserade modeller (Random Forest, Gradient Boosting)
- utvärdering med MAE (Mean Absolute Error) och RMSE (Root Mean Squared Error)
- analys av feature importance

### Dokumentation och vidare arbete

I detta steg implementeras ingen färdig modell, utan fokus ligger på:

- problemformulering
- datainsamling
- förståelse för datakvalitet och begränsningar

I nästa steg kan:

- Jupyter Notebooks användas för explorativ dataanalys
- modeller tränas och utvärderas
- resultat analyseras och diskuteras
