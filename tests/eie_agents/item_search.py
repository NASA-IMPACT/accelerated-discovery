import asyncio

from akd.agents.eie_agents.item_search import ItemSearchAgent, ItemSearchInputSchema

agent = ItemSearchAgent()
params = ItemSearchInputSchema(
    bbox=[-125.0011, 24.9493, -66.9326, 49.5904],
    frequency="",
    temporal_extent={"dates": {"start": "2019-01-01T00:00:00Z", "end": "2019-01-20T00:00:00Z"}},
    collections=[
        "https://openveda.cloud/api/stac/collections/nightlights-hd-monthly",
        "https://earth.gov/ghgcenter/api/stac/collections/odiac-ffco2-monthgrid-v2024",
        "https://earth.gov/ghgcenter/api/stac/collections/odiac-ffco2-monthgrid-v2023",
        "https://earth.gov/ghgcenter/api/stac/collections/odiac-ffco2-monthgrid-v2022",
        "https://earth.gov/ghgcenter/api/stac/collections/blueflux-ghgflux-daygrid-v1",
        "https://earth.gov/ghgcenter/api/stac/collections/vulcan-ffco2-yeargrid-v4",
        "https://earth.gov/ghgcenter/api/stac/collections/vulcan-ffco2-elc-res-yeargrid-v4",
        "https://earth.gov/ghgcenter/api/stac/collections/geos-ghg-daygrid-vPRE",
        "https://earth.gov/ghgcenter/api/stac/collections/oco2-mip-co2budget-yeargrid-v1",
        "https://earth.gov/ghgcenter/api/stac/collections/oco2-mip-meanco2budget-yeargrid-v1",
    ],
    # ['https://openveda.cloud/api/stac/collections/nightlights-hd-monthly', ]
    #'https://earth.gov/ghgcenter/api/stac/collections/VIIRS_SNPP_DayNightBand_At_Sensor_Radiance','https://openveda.cloud/api/stac/collections/nightlights-hd-1band','https://openveda.cloud/api/stac/collections/nightlights-derecho', 'https://openveda.cloud/api/stac/collections/black-marble-night-lights-2025-burma-earthquake', 'https://openveda.cloud/api/stac/collections/ercot-houston-nightlights-freeze', 'https://openveda.cloud/api/stac/collections/greenville-nightlights-tornadoes-2024', 'https://openveda.cloud/api/stac/collections/lakeview-nightlights-tornadoes-2024', 'https://openveda.cloud/api/stac/collections/black-marble-night-lights-houston-tx-2021-deep-freeze', 'https://openveda.cloud/api/stac/collections/black-marble-night-lights-houston-tx-2021-deep-freeze']
)


async def main():
    result = await agent.arun(params)
    print("result is:", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
