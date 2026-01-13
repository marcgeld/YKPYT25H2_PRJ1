# Kort rapport – ValueMeridian

ValueMeridian: Maskininlärningsbaserad värdering av småhus

## Problembeskrivning

Syftet med projektet är att undersöka om det går att förutsäga slutpris för småhus (villkor: villa, radhus, kedjehus) i Partille kommun baserat på historiska försäljningsdata.
Målet är att bygga en regressionsmodell som, givet egenskaper såsom boyta, tomtstorlek, byggår och geografiskt läge, kan ge en rimlig uppskattning av bostadens marknadsvärde.
Problemet är formulerat som ett supervised regression problem, där slutpris (sold_price_raw) används som målvariabel.

## Datainsamling och förberedelse

Data har hämtats via Boolis GraphQL-API och omfattar cirka 6 500 historiska försäljningar från år 2010 och framåt. 
För varje objekt har bland annat följande information samlats in:

- Slutpris och pris per kvadratmeter
- Boyta, tomtstorlek och antal rum
- Byggår
- Geografiska koordinater (latitud/longitud)
- Områdestillhörighet
- Mäklare och mäklarfirma

Datat har validerats vid inläsning, bland annat genom att säkerställa att alla ytmått anges i kvadratmeter (m²). 
Strukturen har plattats till ett tabellformat anpassat för Pandas och maskininlärning.

## Exploratory Data Analysis (utforskande dataanalys),

EDA användes för att förstå datats struktur, kvalitet och relationer mellan variabler.
Med hjälp av Jupyter Notebook:en [value_meridian_eda.ipynb](value_meridian_eda.ipynb) genomfördes statistiska analyser 
och visualiseringar för att identifiera mönster, avvikelser och potentiella prediktorer för slutpris.

Analysen visade bland annat att:

- Slutpriser är högerskev-fördelade[^1], med ett fåtal dyra objekt (outliers).
- Boyta och geografiskt läge är starkt korrelerade med pris.
- Tomtstorlek har betydelse, men i mindre grad än boyta.
- Vissa fält, såsom listpris, saknas för en del objekt men bedöms inte vara kritiska.
- Mäklaregenskaper uppvisar viss korrelation med pris, men sannolikt via selektion snarare än kausalitet.

För att stärka modellen skapades även härledda features (feature engineering), exempelvis:

- Husets ålder (baserat på byggår)
- Förhållandet mellan tomtstorlek och boyta
- Avstånd till kategoriserade viktiga platser (t.ex. centrum, kollektivtrafik)

## Modellering och optimering

En första baseline-modell tränades med linjär regression för att etablera en referensnivå.
Därefter testades mer flexibla modeller, såsom träd-baserade metoder (t.ex. Random Forest), 
vilka bättre kunde hantera icke-linjära samband och interaktioner mellan features.

Modellen förbättrades genom:

- Feature engineering (t.ex. avståndsfeatures och kvotvariabler)
- Justering av modellparametrar (hyperparametertuning[^2])
- Jämförelse mellan olika uppsättningar av indatafeatures

Dessa steg ledde till tydligt förbättrad prediktiv prestanda jämfört med baseline-modellen.

## Utvärdering och jämförelser

Modellernas effektivitet jämfördes genom att använda samma tränings- och testuppdelning och 
utvärdera med standardmått för regression, såsom medelfel och residualanalys. 
Alternativa modeller och feature-uppsättningar testades för att bedöma robusthet och generaliserbarhet.

Analysen visade att:

- Geografiska features ger ett betydande bidrag till modellens prestanda.
- Mer komplexa modeller presterar bättre än enkla linjära modeller, men med ökad risk för överanpassning.
- Det finns utrymme för förbättring, exempelvis genom mer detaljerad geografisk representation eller kompletterande datakällor.

En viktig begränsning i datamaterialet är att modellen endast använder boyta (BOA) och saknar information om biyta (BIA), såsom källare eller souterrängplan.
För många småhus, särskilt 50–70-talshus i Partille, utgör biytan en betydande del av den upplevda och faktiska användbara ytan.
Avsaknaden av denna information leder till att modellen systematiskt undervärderar hus med stor biyta, då dessa behandlas som betydligt mindre bostäder än vad marknaden i praktiken gör.

## Slutsats och vidare arbete

Projektet visar att det är fullt möjligt att bygga en fungerande modell för bostadsvärdering baserat på öppet tillgänglig försäljningsdata.
Resultaten är rimliga och följer förväntade mönster på bostadsmarknaden.

## Vidare förbättringar kan inkludera:

- Mer avancerad hantering av geografisk information
- Ytterligare feature engineering
- Test av andra modelltyper eller ensemble-metoder

## Teknisk implementation och kodstruktur

Koden är strukturerad i tydliga moduler

 - datainsamling: [fetch_data_to_local_cache.py](src/value_meridian/fetch_data_to_local_cache.py)
 - analys införande och nya features (Jupyter Notebook) [value_meridian_eda.ipynb](value_meridian_eda.ipynb)
 - modelträning / utvärdering [train.py](src/value_meridian/train.py)
 - inferens [inference.py](src/value_meridian/inference.py)

Utöver dessa huvudmoduler finns funktioner som är delade mellan modulerna:

 - Kod för hantering av features, [features.py](src/value_meridian/features.py).
 - Ett sammanhållande schema för POI information, [schema.py](src/value_meridian/schema.py).

Samt en projekt fil [pyproject.toml](pyproject.toml) som hanterar beroenden och paketering och möjliggör körning 
via `uv run <module>`. exempelvis:

```bash
vm-fetch --area-id 268 --output data/partille_raw.csv
vm-train --csv data/partille_sold_eda.csv --target sold_price_raw --model-out data/partille_model.joblib --tune
vm-infer --model data/partille_model.joblib --input data/example_infer.yaml
```

[^1]: En högerskev fördelning (eller positivt skev fördelning) innebär att de flesta värden ligger mot den nedre, vänstra delen av grafen, medan en liten grupp extremt höga värden drar ut en "svans" åt höger, vilket gör att medelvärdet blir högre än medianen (Medelvärde > Median > Typvärde). Det är vanligt vid exempelvis inkomster, där få tjänar mycket men majoriteten tjänar mer måttligt, skapande den långa svansen.

[^2]: Hyperparameter-tuning (eller optimering) är processen att hitta de bästa inställningarna (hyperparametrarna) för en maskininlärningsmodell, såsom inlärningshastighet eller antal lager, som styr hur modellen lär sig, för att maximera dess prestanda och noggrannhet på ny data. Detta görs genom att testa olika kombinationer av hyperparametrar, ofta med tekniker som grid search, random search och Bayesian optimization, för att minska modellens förlustfunktion och förbättra dess förmåga att generalisera. 
