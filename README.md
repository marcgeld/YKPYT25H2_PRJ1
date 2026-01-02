# ValueMeridian

Local property valuation toolkit — fetch data, train models, run inference.  
Built for the **YKPYT25H2** course exercises.

## Prerequisites

- Python **3.14+** installed locally
  (The project specifies `>=3.14`, but works likley with older versions of Python)
- `uv` or `pip` for environment and dependency management
- macOS / Linux / WSL recommended (Windows works, but commands may vary)

## Set up the virtual environment (using [uv](https://docs.astral.sh/uv/getting-started/installation/))

```bash
uv venv --python 3.14
source .venv/bin/activate
```

```bash
python3 -m venv .venv
source .venv/bin/activate
```

```bash
uv pip install --upgrade pip
```

```bash
pip install --upgrade pip
```

## Install

Install the project (editable mode)

```bash
pip install -e .
```

## Run

### vm-fetch – fetch data from GraphQL endpoint (Booli)

```bash
vm-fetch --area-id 268 --output data/partille_raw.csv
```

Or with [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv run vm-fetch --area-id 268 --output data/partille_raw.csv
```

### vm-train – train a baseline model

```bash
vm-train --data data/partille_raw.csv --model-out data/partille_model.joblib
```

Or with [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv run vm-train --data data/partille_raw.csv --model-out data/partille_model.joblib
```

### vm-infer – run inference given model & features

```bash
vm-infer --model data/partille_model.joblib --living-area 160 --rooms 5 --plot-area 500
```

Or with [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv run vm-infer --model data/partille_model.joblib --living-area 160 --rooms 5 --plot-area 500
```
## Feature egineering and model training notebook

| Place                         | Note                                                                                                                  |lat/lon|
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------|-------|
| Göteborg Centralstation       | Represents access to the main job market, commuting options and city amenities.                                       |57.70887, 11.97456|
| Partille Centrum              | Captures local services, shopping and the local town centre effect.                                                   |57.7393, 12.1064|
| Mölndal Centrum               | Alternative employment hub in the region.                                                                             |57.6559, 12.0137|
| Landvetter Flygplats          | Relevant for frequent travellers and some professional profiles.                                                      |57.6660, 12.2942|
| Skatås / Delsjön              | Captures proximity to nature and recreational areas.                                                                  |57.6900, 12.0270|
| Östra Sjukhuset (Sahlgrenska) | Represents both a major employer and proximity to healthcare, which can affect perceived security and attractiveness. |57.7274, 12.0446|

Calculate haversine distance to key locations
```python
df["dist_gbg_km"]      = haversine(df.lat, df.lon, 57.70887, 11.97456)
df["dist_partille_km"] = haversine(df.lat, df.lon, 57.7393, 12.1064)
df["dist_molndal_km"]  = haversine(df.lat, df.lon, 57.6559, 12.0137)
df["dist_airport_km"]  = haversine(df.lat, df.lon, 57.6660, 12.2942)
df["dist_skatas_km"]   = haversine(df.lat, df.lon, 57.6900, 12.0270)
df["dist_ostra_sjukhuset_km"]   = haversine(df.lat, df.lon, 57.7274, 12.0446)
```

## References

Lookup area id for `partille` with [cURL](https://curl.se)

```bash
curl "https://www.booli.se/graphql" \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "User-Agent: value-meridian-cli/0.1" \
  -H "x-apollo-operation-name: areaSuggestionSearch" \
  -H "apollo-require-preflight: true" \
  -H "Origin: https://www.booli.se" \
  -H "Referer: https://www.booli.se/" \
  --data-raw '{
    "operationName": "areaSuggestionSearch",
    "query": "query areaSuggestionSearch { areaSuggestionSearch(search: \"partille\") { suggestions { id displayName parent parentType parentDisplayName parentTypeDisplayName parentId } } }"
  }'
  ```

  Expected response:

  ```json
  {"data":{"areaSuggestionSearch":{"suggestions":[{"id":"268","displayName":"Partille kommun","parent":"Västra Götalands län","parentType":"Län","parentDisplayName":"Västra Götalands län","parentTypeDisplayName":"Län","parentId":"23"},{"id":"1007045","displayName":"Sävedalen","parent":"Partille","parentType":"Kommun","parentDisplayName":"Partille kommun","parentTypeDisplayName":"Kommun","parentId":"268"},{"id":"898568","displayName":"Ugglum","parent":"Partille","parentType":"Kommun","parentDisplayName":"Partille kommun","parentTypeDisplayName":"Kommun","parentId":"268"},{"id":"927795","displayName":"Öjersjö","parent":"Partille","parentType":"Kommun","parentDisplayName":"Partille kommun","parentTypeDisplayName":"Kommun","parentId":"268"},{"id":"2364","displayName":"Furulund","parent":"Partille","parentType":"Kommun","parentDisplayName":"Partille kommun","parentTypeDisplayName":"Kommun","parentId":"268"},{"id":"4428","displayName":"Mellby","parent":"Partille","parentType":"Kommun","parentDisplayName":"Partille kommun","parentTypeDisplayName":"Kommun","parentId":"268"},{"id":"139939","displayName":"Partillevägen","parent":"Härryda","parentType":"Kommun","parentDisplayName":"Härryda kommun","parentTypeDisplayName":"Kommun","parentId":"201"},{"id":"924842","displayName":"Gamla Partillevägen","parent":"Härryda","parentType":"Kommun","parentDisplayName":"Härryda kommun","parentTypeDisplayName":"Kommun","parentId":"201"},{"id":"97804","displayName":"Björndammsterrassen","parent":"Partille","parentType":"Kommun","parentDisplayName":"Partille kommun","parentTypeDisplayName":"Kommun","parentId":"268"}]}}}
  ```
