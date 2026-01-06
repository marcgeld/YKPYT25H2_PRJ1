"""
ValueMeridian - Fetch sold property data from Booli GraphQL and store it as a local CSV cache.

- Uses gql with RequestsHTTPTransport.
- Paginates through all pages.
- Flattens nested GraphQL fields to a single row per sold property.
- Outputs a Pandas-friendly CSV (stable column order).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport


# Default endpoint (you can override with BOOLI_GRAPHQL_URL env var)
DEFAULT_BOOLI_GRAPHQL_URL = "https://www.booli.se/graphql"


# This query matches your new working query structure.
# Note: Booli expects areaId as ID!, not String!.
SEARCH_SOLD_QUERY = gql(
    """
    query searchCount($areaId: ID!, $pageSize: Int!, $page: Int!) {
      searchSold(
        input: {
          filters: [
            { key: "objectType", value: "Villa,Kedjehus-Parhus-Radhus" }
            { key: "tenureForm", value: "Äganderätt" }
            { key: "minSoldDate", value: "2010-01-01" }
            { key: "extendAreas", value: "1" }
          ]
          areaId: $areaId
          pageSize: $pageSize
          page: $page
        }
      ) {
        totalCount
        pages
        result {
          __typename
          ... on Property {
            id
            constructionYear
            operatingCost { raw unit }
            livingArea { raw unit }
            plotArea { raw unit }
            areas { type name }
            rooms { raw unit }
            location {
              address {
                streetAddress
                postCode
              }
              namedAreas
              region {
                municipalityName
                countyName
              }
            }
            objectType
            latitude
            longitude
            url
          }
          ... on SoldProperty {
            soldDate
            listPrice { raw unit }
            firstPrice { raw unit }
            soldPrice { raw unit }
            soldSqmPrice { raw unit }
            agent {
              id
              name
              recommendations
              reviewCount
              overallRating
              premium
            }
            agency {
              id
              name
            }
          }
        }
      }
    }
    """
)


def build_transport(url: str) -> RequestsHTTPTransport:
    """
    Create a Requests transport with headers that work against Booli's Apollo GraphQL endpoint.

    These headers help avoid Apollo CSRF protections and mimic a normal browser origin.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "value-meridian/0.1",
        "x-apollo-operation-name": "searchCount",
        "apollo-require-preflight": "true",
        "Origin": "https://www.booli.se",
        "Referer": "https://www.booli.se/",
    }
    return RequestsHTTPTransport(url=url, headers=headers, verify=True, retries=3)


def fetch_sold_page(client: Client, area_id: str, page: int, page_size: int) -> Dict[str, Any]:
    """
    Fetch one page of sold results from Booli GraphQL.
    """
    variables = {"areaId": area_id, "page": page, "pageSize": page_size}
    result = client.execute(SEARCH_SOLD_QUERY, variable_values=variables)
    return result["searchSold"]


def areas_to_maps(areas: Optional[List[Dict[str, Any]]]) -> Tuple[Dict[str, str], List[str]]:
    """
    Convert the 'areas' list into:
    - a dict with one value per type (first seen wins)
    - a list of "type:name" tokens (de-duplicated, stable) for optional debugging/feature use

    IMPORTANT:
    The 'areas' list in your sample is NOT about floor areas (BOA/BIA), it's geographic/admin metadata.
    """
    by_type: Dict[str, str] = {}
    tokens: List[str] = []
    seen = set()

    for a in areas or []:
        t = a.get("type")
        n = a.get("name")
        if not t or not n:
            continue

        # Token list (dedup)
        token = f"{t}:{n}"
        if token not in seen:
            tokens.append(token)
            seen.add(token)

        # First occurrence per type wins (to keep stable columns)
        if t not in by_type:
            by_type[t] = n

    return by_type, tokens


def join_list(value: Any, sep: str = "|") -> str:
    """
    Join a list into a single string column; return empty string for missing.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        # Deduplicate while preserving order
        out: List[str] = []
        seen = set()
        for v in value:
            s = str(v)
            if s not in seen:
                out.append(s)
                seen.add(s)
        return sep.join(out)
    return str(value)


def require_unit(
    unit: Optional[str],
    expected_unit: str,
    field_name: str,
    object_id: Optional[str] = None,
) -> None:
    """
    Ensure that a field uses the expected unit.

    - If unit is None, the check is skipped (missing data is allowed).
    - If unit is present and does not match expected_unit, raise ValueError.
    """

    if expected_unit is None:
        raise RuntimeError(
            f"Invalid configuration: expected_unit is None for field '{field_name}'"
        )

    # Missing unit is allowed (null-safe)
    if unit is None:
        return

    if unit != expected_unit:
        obj = f" (object id={object_id})" if object_id else ""
        raise ValueError(
            f"Unexpected unit for '{field_name}': '{unit}', "
            f"expected '{expected_unit}'{obj}"
        )

def flatten_result(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a single searchSold 'result' element into a CSV-friendly row.

    Booli appears to return a merged object where sold fields (SoldProperty) are present
    alongside property fields (Property). We therefore safely read both sets of fields
    from the same node.
    """
    object_id = node.get("id")
    operating_cost = node.get("operatingCost") or {}
    living_area = node.get("livingArea") or {}
    plot_area = node.get("plotArea") or {}
    rooms = node.get("rooms") or {}

    # Enforce area units
    require_unit(
        operating_cost.get("unit"),
        expected_unit="kr/mån",
        field_name="operating_cost",
        object_id=object_id,
    )

    require_unit(
        living_area.get("unit"),
        expected_unit="m²",
        field_name="livingArea",
        object_id=object_id,
    )

    require_unit(
        living_area.get("unit"),
        expected_unit="m²",
        field_name="plot_area",
        object_id=object_id,
    )

    require_unit(
        living_area.get("rooms"),
        expected_unit="rum",
        field_name="rooms",
        object_id=object_id,
    )
    list_price = node.get("listPrice") or {}
    first_price = node.get("firstPrice") or {}
    sold_price = node.get("soldPrice") or {}
    sold_sqm_price = node.get("soldSqmPrice") or {}

    require_unit(
        list_price.get("unit"),
        expected_unit="kr",
        field_name="list_price",
        object_id=object_id,
    )

    require_unit(
        list_price.get("unit"),
        expected_unit="kr",
        field_name="list_price",
        object_id=object_id,
    )

    require_unit(
        sold_price.get("unit"),
        expected_unit="kr",
        field_name="sold_price",
        object_id=object_id,
    )

    require_unit(
        sold_sqm_price.get("unit"),
        expected_unit="kr/m²",
        field_name="sold_sqm_price",
        object_id=object_id,
    )

    agent = node.get("agent") or {}
    agency = node.get("agency") or {}

    location = node.get("location") or {}
    address = location.get("address") or {}
    region = location.get("region") or {}

    areas_by_type, areas_tokens = areas_to_maps(node.get("areas"))

    # Post code: sometimes in address.postCode, sometimes only in areas[type=postcode]
    post_code = address.get("postCode") or areas_by_type.get("postcode") or ""

    # Named areas: list like ["Furuskog"]
    named_areas = join_list(location.get("namedAreas"))

    # userDefined can appear multiple times; keep a stable joined list of all userDefined names
    user_defined_names = []
    for token in areas_tokens:
        if token.startswith("userDefined:"):
            user_defined_names.append(token.split(":", 1)[1])
    user_defined_joined = join_list(user_defined_names)

    return {
        # Identity & transaction
        "id": node.get("id"),
        "sold_date": node.get("soldDate"),

        # Prices
        "sold_price_raw": sold_price.get("raw"),
        "sold_price_unit": sold_price.get("unit"),
        "sold_sqm_price_raw": sold_sqm_price.get("raw"),
        "sold_sqm_price_unit": sold_sqm_price.get("unit"),
        "list_price_raw": list_price.get("raw"),
        "list_price_unit": list_price.get("unit"),
        "first_price_raw": first_price.get("raw"),
        "first_price_unit": first_price.get("unit"),

        # Property basics
        "object_type": node.get("objectType"),
        "construction_year": node.get("constructionYear"),
        "living_area_raw": living_area.get("raw"),
        #"living_area_unit": living_area.get("unit"),
        "plot_area_raw": plot_area.get("raw"),
        #"plot_area_unit": plot_area.get("unit"),
        "rooms_raw": rooms.get("raw"),
        #"rooms_unit": rooms.get("unit"),
        "operating_cost_raw": operating_cost.get("raw"),
        #"operating_cost_unit": operating_cost.get("unit"),

        # Location / address
        "street_address": address.get("streetAddress"),
        "post_code": post_code,
        "named_areas": named_areas,
        "municipality_name": region.get("municipalityName") or areas_by_type.get("municipality"),
        "county_name": region.get("countyName") or areas_by_type.get("county"),
        "latitude": node.get("latitude"),
        "longitude": node.get("longitude"),

        # Useful area tags (admin/geo)
        "area_country": areas_by_type.get("country", ""),
        "area_populated_area": areas_by_type.get("populatedArea", ""),
        "area_locality": areas_by_type.get("locality", ""),
        "area_suburb": areas_by_type.get("suburb", ""),
        "area_user_defined": user_defined_joined,
        "area_index_area": areas_by_type.get("indexArea", ""),
        "electricity_bidding_zone": areas_by_type.get("electricityBiddingZone", ""),

        # Agent/agency
        "agent_id": agent.get("id"),
        "agent_name": agent.get("name"),
        "agent_recommendations": agent.get("recommendations"),
        "agent_review_count": agent.get("reviewCount"),
        "agent_overall_rating": agent.get("overallRating"),
        "agent_premium": agent.get("premium"),
        "agency_id": agency.get("id"),
        "agency_name": agency.get("name"),

        # Booli relative URL
        "url": node.get("url"),

        # Debug (optional): keep type to understand unions
        "__typename": node.get("__typename"),
    }


def fetch_all_sold(area_id: str, page_size: int, graphql_url: str) -> List[Dict[str, Any]]:
    """
    Fetch all sold results for an areaId across all pages.
    """
    transport = build_transport(graphql_url)

    # fetch_schema_from_transport=False avoids any schema introspection attempts (some servers block it).
    client = Client(transport=transport, fetch_schema_from_transport=False)

    first = fetch_sold_page(client, area_id=area_id, page=1, page_size=page_size)
    total_pages = int(first.get("pages") or 0)
    total_count = int(first.get("totalCount") or 0)

    print(f"Total count: {total_count}, pages: {total_pages}, page_size: {page_size}")

    rows: List[Dict[str, Any]] = []
    for node in first.get("result", []) or []:
        rows.append(flatten_result(node))

    for p in range(2, total_pages + 1):
        page_data = fetch_sold_page(client, area_id=area_id, page=p, page_size=page_size)
        for node in page_data.get("result", []) or []:
            rows.append(flatten_result(node))

        # Light progress indicator
        if p % 25 == 0 or p == total_pages:
            print(f"Fetched page {p}/{total_pages} (rows so far: {len(rows)})")

    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write rows to CSV with a stable, human-friendly column order.
    """
    if not rows:
        print("No rows to write. CSV not created.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_order = [
        # Identity & transaction
        "id",
        "sold_date",

        # Prices
        "sold_price_raw",
        #"sold_price_unit",
        "sold_sqm_price_raw",
        #"sold_sqm_price_unit",
        "list_price_raw",
        #"list_price_unit",
        "first_price_raw",
        #"first_price_unit",

        # Property
        "object_type",
        "construction_year",
        "living_area_raw",
        #"living_area_unit",
        "plot_area_raw",
        #"plot_area_unit",
        "rooms_raw",
        #"rooms_unit",
        "operating_cost_raw",
        #"operating_cost_unit",

        # Location
        "street_address",
        "post_code",
        "named_areas",
        "municipality_name",
        "county_name",
        "latitude",
        "longitude",

        # Area tags
        "area_country",
        "area_populated_area",
        "area_locality",
        "area_suburb",
        "area_user_defined",
        "area_index_area",
        "electricity_bidding_zone",

        # Agent/agency
        "agent_id",
        "agent_name",
        "agent_recommendations",
        "agent_review_count",
        "agent_overall_rating",
        "agent_premium",
        "agency_id",
        "agency_name",

        # URL + debug
        #"url",
        #"__typename",
    ]

    all_keys = {k for r in rows for k in r.keys()}
    fieldnames = [k for k in preferred_order if k in all_keys]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print("Columns:", ", ".join(fieldnames))


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fetch Booli sold data via GraphQL and write to a local CSV cache."
    )
    parser.add_argument("--area-id", required=True, help="Booli areaId (e.g. 268 for Partille)")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--page-size", type=int, default=100, help="Page size for GraphQL pagination")
    parser.add_argument(
        "--graphql-url",
        default=os.environ.get("BOOLI_GRAPHQL_URL", DEFAULT_BOOLI_GRAPHQL_URL),
        help="Booli GraphQL URL (default: https://www.booli.se/graphql)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)

    rows = fetch_all_sold(
        area_id=str(args.area_id),
        page_size=int(args.page_size),
        graphql_url=str(args.graphql_url),
    )
    write_csv(rows, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())