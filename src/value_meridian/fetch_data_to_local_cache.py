import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

BOOLI_GRAPHQL_URL = "https://www.booli.se/graphql"


def build_transport() -> RequestsHTTPTransport:
    """
    Build the GraphQL HTTP transport for Booli's Apollo endpoint.
    """
    return RequestsHTTPTransport(
        url=BOOLI_GRAPHQL_URL,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": "https://www.booli.se",
            "Referer": "https://www.booli.se/",
            "User-Agent": "value-meridian-cli/0.0.1",
            # Apollo-specific headers to satisfy CSRF / preflight requirements
            "x-apollo-operation-name": "searchCount",
            "apollo-require-preflight": "true",
        },
        verify=True,
        retries=3,
    )


# Filters are hard-coded in the query body as requested.
SEARCH_SOLD_QUERY = gql(
    """
    query searchCount(
      $areaId: ID!,
      $page: Int!,
      $pageSize: Int!
    ) {
      searchSold(
        input: {
          areaId: $areaId
          pageSize: $pageSize
          page: $page
          filters: [
            { key: "objectType", value: "Villa,Kedjehus-Parhus-Radhus" }
            { key: "tenureForm", value: "Äganderätt" }
            { key: "minSoldDate", value: "2010-01-01" }
            { key: "extendAreas", value: "1" }
          ]
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
            livingArea     { raw unit }
            plotArea       { raw unit }
            rooms          { raw unit }
            primaryArea  { type name }
            secondaryArea{ type name id }
            objectType
            streetAddress
            latitude
            longitude
            url
          }
        }
      }
    }
    """
)


def fetch_sold_page(
    client: Client,
    *,
    area_id: str,
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    """
    Fetch a single page of sold properties using the GraphQL client.
    """
    variables = {
        "areaId": area_id,
        "page": page,
        "pageSize": page_size,
    }

    result = client.execute(SEARCH_SOLD_QUERY, variable_values=variables)
    # result is the root of the query, i.e. {"searchSold": {...}}
    return result["searchSold"]


def flatten_property(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a single node from GraphQL into a simple dict for CSV.

    This assumes the node has the fields of a Property as returned
    by the searchSold query.
    """
    living = node.get("livingArea") or {}
    plot = node.get("plotArea") or {}
    rooms = node.get("rooms") or {}
    op_cost = node.get("operatingCost") or {}
    primary = node.get("primaryArea") or {}
    secondary = node.get("secondaryArea") or {}

    return {
        "id": node.get("id"),
        "construction_year": node.get("constructionYear"),
        "operating_cost_raw": op_cost.get("raw"),
        "operating_cost_unit": op_cost.get("unit"),
        "living_area_raw": living.get("raw"),
        "living_area_unit": living.get("unit"),
        "plot_area_raw": plot.get("raw"),
        "plot_area_unit": plot.get("unit"),
        "rooms_raw": rooms.get("raw"),
        "rooms_unit": rooms.get("unit"),
        "primary_area_type": primary.get("type"),
        "primary_area_name": primary.get("name"),
        "secondary_area_type": secondary.get("type"),
        "secondary_area_name": secondary.get("name"),
        "secondary_area_id": secondary.get("id"),
        "object_type": node.get("objectType"),
        "street_address": node.get("streetAddress"),
        "latitude": node.get("latitude"),
        "longitude": node.get("longitude"),
        "url": node.get("url"),
    }

def fetch_all_sold(
    area_id: str,
    *,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch all sold properties for a given area using page/pageSize pagination.
    Returns a list of flattened property dicts ready for CSV/Pandas.
    """
    transport = build_transport()
    client = Client(transport=transport, fetch_schema_from_transport=False)

    all_rows: List[Dict[str, Any]] = []

    # First page: also tells us how many pages there are in total.
    first_page = fetch_sold_page(
        client,
        area_id=area_id,
        page=1,
        page_size=page_size,
    )

    total_count = first_page.get("totalCount")
    total_pages = first_page.get("pages") or 1

    print(f"Total sold items reported by API: {total_count}")
    print(f"Total pages: {total_pages}")

    for node in first_page.get("result", []):
        flat = flatten_property(node)
        all_rows.append(flat)

    # Loop remaining pages until all pages are fetched
    for page in range(2, total_pages + 1):
        page_data = fetch_sold_page(
            client,
            area_id=area_id,
            page=page,
            page_size=page_size,
        )

        for node in page_data.get("result", []):
            flat = flatten_property(node)
            if flat is not None:
                all_rows.append(flat)

        print(f"Fetched page {page}/{total_pages}, rows so far: {len(all_rows)}")

    return all_rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write the flattened rows to a CSV file with a stable, human-friendly column order.
    """
    if not rows:
        print("No rows to write, skipping CSV file.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Preferred logical order of columns
    preferred_order = [
        "id",
        "street_address",
        "latitude",
        "longitude",
        "object_type",
        "construction_year",
        "living_area_raw",
        "living_area_unit",
        "plot_area_raw",
        "plot_area_unit",
        "rooms_raw",
        "rooms_unit",
        "operating_cost_raw",
        "operating_cost_unit",
        "primary_area_type",
        "primary_area_name",
        "secondary_area_type",
        "secondary_area_name",
        "secondary_area_id",
        "url",
    ]

    # Collect all keys that actually exist in the data
    all_keys = {key for row in rows for key in row.keys()}

    # Keep preferred order, but only for keys that really exist
    fieldnames: List[str] = [k for k in preferred_order if k in all_keys]

    # Append any extra keys that are not in the preferred list
    extras = sorted(all_keys - set(fieldnames))
    fieldnames.extend(extras)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print("Columns:", ", ".join(fieldnames))


def main() -> None:
    """
    CLI entrypoint for `vm-fetch`.

    Example:
      vm-fetch --area-id 268 --output data/partille_sold.csv --page-size 100
    """
    parser = argparse.ArgumentParser(
        description="Fetch sold property data from Booli GraphQL into a local CSV cache."
    )
    parser.add_argument(
        "--area-id",
        required=True,
        help="Booli areaId as string, e.g. '268' for Partille.",
    )
    parser.add_argument(
        "--output",
        default="data/sold_properties.csv",
        help="Output CSV file path (default: data/sold_properties.csv).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of items per page to request from the API (default: 100).",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    rows = fetch_all_sold(
        area_id=args.area_id,
        page_size=args.page_size,
    )
    write_csv(rows, output_path)


if __name__ == "__main__":
    main()