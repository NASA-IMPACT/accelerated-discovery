"""
Simple imports file for data search component types.

Import all the base types you need for working with the data search components
in notebooks or scripts.
"""

# Base agent config

# Main agent and config

# Base agent schemas
from akd.agents.data_search.components.known_parameters import QueryApproach
from akd.agents.data_search.components.scientific_decomposition import (
    ScientificDecomposition,
)

# Core component types
from akd.agents.data_search.components.topic_splitting import Topic

print("✅ All data search component types imported successfully!")

# Example usage:
if __name__ == "__main__":
    # Create a scientific decomposition
    decomposition = ScientificDecomposition(
        title="Satellite-derived Land Surface Temperature (LST) products",
        scientific_justification="Direct retrievals of land surface temperature are the primary observable needed to compare surface thermal conditions on the two dates. LST products (e.g., MODIS LST, VIIRS LST, Landsat-based LST, reprocessed LST collections) provide gridded estimates of surface skin temperature derived from thermal infrared radiances using radiative transfer and emissivity corrections.",
    )

    print("\n📊 Example ScientificDecomposition:")
    print(f"   Title: {decomposition.title}")
    print(f"   Justification: {decomposition.scientific_justification[:100]}...")

    # Create a topic
    topic = Topic(
        title="Land Surface Temperature Analysis",
        functional_context="Analysis of thermal conditions using satellite data",
    )

    print("\n📌 Example Topic:")
    print(f"   Title: {topic.title}")
    print(f"   Context: {topic.functional_context}")

    # Create a query approach
    query_approach = QueryApproach(
        instrument="MODIS",
        temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
        processing_level="Level 2",
    )

    print("\n🔍 Example QueryApproach:")
    print(f"   Instrument: {query_approach.instrument}")
    print(f"   Temporal: {query_approach.temporal}")
    print(f"   Processing Level: {query_approach.processing_level}")
