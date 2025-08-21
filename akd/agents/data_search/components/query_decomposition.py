"""
Query Decomposition Component for Data Search.

DEPRECATED: This component uses rule-based pattern matching and is being replaced
by LLM-driven scientific angle generation and CMR query generation components.

Transforms natural language queries into structured CMR search parameters.
"""

from typing import List, Optional, Dict, Any
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from loguru import logger

from akd.agents.query import QueryAgent, QueryAgentInputSchema


class DecomposedQuery(BaseModel):
    """Result of query decomposition with structured search parameters."""
    
    # Scientific content
    keywords: List[str] = Field(
        default_factory=list,
        description="Scientific keywords extracted from query"
    )
    data_type_indicators: List[str] = Field(
        default_factory=list,
        description="Indicators of data type (e.g., 'temperature', 'vegetation')"
    )
    
    # Platform and instrument
    platforms: List[str] = Field(
        default_factory=list,
        description="Satellite platforms mentioned (e.g., 'Terra', 'Aqua')"
    )
    instruments: List[str] = Field(
        default_factory=list,
        description="Instruments mentioned (e.g., 'MODIS', 'VIIRS')"
    )
    
    # Processing level
    processing_level: Optional[str] = Field(
        None,
        description="Data processing level if mentioned (L0-L4)"
    )
    
    # Temporal constraints
    temporal_start: Optional[str] = Field(
        None,
        description="Start time in ISO format"
    )
    temporal_end: Optional[str] = Field(
        None,
        description="End time in ISO format"
    )
    temporal_description: Optional[str] = Field(
        None,
        description="Original temporal description from query"
    )
    
    # Spatial constraints
    spatial_bounds: Optional[Dict[str, float]] = Field(
        None,
        description="Spatial bounding box if identified"
    )
    spatial_description: Optional[str] = Field(
        None,
        description="Original spatial description from query"
    )
    
    # Data access preferences
    online_only: bool = Field(
        True,
        description="Prefer online accessible data"
    )
    
    # Search variations
    search_variations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Multiple parameter combinations for comprehensive search"
    )


class QueryDecompositionComponent:
    """
    Component for decomposing natural language data queries into CMR search parameters.
    
    DEPRECATED: This component is deprecated in favor of LLM-driven components.
    Use ScientificAnglesComponent and CMRQueryGenerationComponent instead.
    
    Uses pattern matching, keyword extraction, and the existing QueryAgent to transform
    user queries like "Find MODIS sea surface temperature data from 2023 over the Pacific"
    into structured CMR API parameters.
    """

    def __init__(
        self,
        query_agent: Optional[QueryAgent] = None,
        debug: bool = False
    ):
        """
        Initialize the query decomposition component.
        
        Args:
            query_agent: Optional QueryAgent instance for generating search queries
            debug: Enable debug logging
        """
        self.query_agent = query_agent or QueryAgent()
        self.debug = debug
        
        # Known platform/instrument mappings
        self.platform_patterns = {
            r'\b(terra|terra satellite)\b': 'Terra',
            r'\b(aqua|aqua satellite)\b': 'Aqua', 
            r'\b(sentinel[- ]?[12][ab]?)\b': lambda m: m.group(1).upper(),
            r'\b(landsat[- ]?[0-9]+)\b': lambda m: m.group(1).title(),
            r'\b(suomi[- ]?npp|snpp)\b': 'Suomi-NPP',
            r'\b(noaa[- ]?[0-9]+)\b': lambda m: m.group(1).upper(),
        }
        
        self.instrument_patterns = {
            r'\bmodis\b': 'MODIS',
            r'\bviirs\b': 'VIIRS',
            r'\b(oli|oli[- ]tirs)\b': 'OLI',
            r'\bmsi\b': 'MSI',
            r'\baster\b': 'ASTER',
            r'\bavhrr\b': 'AVHRR',
        }
        
        # Processing level patterns
        self.processing_level_patterns = {
            r'\b(level[- ]?0|l0|raw)\b': 'L0',
            r'\b(level[- ]?1a|l1a)\b': 'L1A',
            r'\b(level[- ]?1b|l1b)\b': 'L1B', 
            r'\b(level[- ]?2|l2)\b': 'L2',
            r'\b(level[- ]?3|l3|gridded)\b': 'L3',
            r'\b(level[- ]?4|l4|model)\b': 'L4',
        }
        
        # Common data type keywords
        self.data_type_keywords = {
            'temperature': ['temperature', 'thermal', 'sst', 'sea surface temperature', 'land surface temperature'],
            'vegetation': ['vegetation', 'ndvi', 'evi', 'greenness', 'chlorophyll'],
            'ocean': ['ocean', 'sea', 'marine', 'sea surface', 'oceanographic'],
            'atmosphere': ['atmosphere', 'atmospheric', 'aerosol', 'cloud', 'precipitation'],
            'land': ['land', 'terrestrial', 'surface', 'topography', 'elevation'],
            'ice': ['ice', 'snow', 'glacier', 'sea ice', 'snow cover'],
        }
        
        # Spatial region patterns (simplified)
        self.region_patterns = {
            r'\b(global|worldwide|earth)\b': {'west': -180, 'south': -90, 'east': 180, 'north': 90},
            r'\b(pacific|pacific ocean)\b': {'west': -180, 'south': -60, 'east': -70, 'north': 60},
            r'\b(atlantic|atlantic ocean)\b': {'west': -80, 'south': -60, 'east': 20, 'north': 70},
            r'\b(california|ca)\b': {'west': -125, 'south': 32, 'east': -114, 'north': 42},
            r'\b(conus|continental us|united states)\b': {'west': -125, 'south': 25, 'east': -66, 'north': 50},
        }

    async def process(self, query: str) -> DecomposedQuery:
        """
        Process a natural language query into structured search parameters.
        
        DEPRECATED: Use ScientificAnglesComponent and CMRQueryGenerationComponent instead.
        
        Args:
            query: Natural language data discovery query
            
        Returns:
            DecomposedQuery with structured parameters and search variations
        """
        logger.warning("QueryDecompositionComponent.process is deprecated. Use LLM-driven components instead.")
        
        if self.debug:
            logger.debug(f"Decomposing query: '{query}'")
        
        query_lower = query.lower()
        
        # Extract platforms and instruments
        platforms = self._extract_patterns(query_lower, self.platform_patterns)
        instruments = self._extract_patterns(query_lower, self.instrument_patterns)
        
        # Extract processing level
        processing_level = self._extract_processing_level(query_lower)
        
        # Extract temporal constraints
        temporal_start, temporal_end, temporal_desc = self._extract_temporal(query_lower)
        
        # Extract spatial constraints
        spatial_bounds, spatial_desc = self._extract_spatial(query_lower)
        
        # Extract data type keywords
        keywords = self._extract_data_keywords(query_lower)
        data_indicators = self._extract_data_type_indicators(query_lower)
        
        # Generate search variations using QueryAgent
        search_variations = await self._generate_search_variations(
            query, keywords, platforms, instruments, processing_level
        )
        
        result = DecomposedQuery(
            keywords=keywords,
            data_type_indicators=data_indicators,
            platforms=platforms,
            instruments=instruments,
            processing_level=processing_level,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            temporal_description=temporal_desc,
            spatial_bounds=spatial_bounds,
            spatial_description=spatial_desc,
            search_variations=search_variations
        )
        
        if self.debug:
            logger.debug(f"Query decomposition result: {len(search_variations)} variations generated")
            
        return result

    def _extract_patterns(self, text: str, patterns: Dict[str, Any]) -> List[str]:
        """Extract matches from text using pattern dictionary."""
        matches = []
        for pattern, replacement in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if callable(replacement):
                    matches.append(replacement(match))
                else:
                    matches.append(replacement)
        return list(set(matches))  # Remove duplicates

    def _extract_processing_level(self, text: str) -> Optional[str]:
        """Extract processing level from text."""
        for pattern, level in self.processing_level_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return level
        return None

    def _extract_temporal(self, text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract temporal constraints from text."""
        temporal_desc = None
        start_time = None
        end_time = None
        
        # Year patterns
        year_match = re.search(r'\b(20[0-2][0-9])\b', text)
        if year_match:
            year = year_match.group(1)
            start_time = f"{year}-01-01T00:00:00Z"
            end_time = f"{year}-12-31T23:59:59Z"
            temporal_desc = f"year {year}"
        
        # Recent/last patterns
        if re.search(r'\b(recent|latest|current)\b', text):
            # Last 6 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            start_time = start_date.strftime("%Y-%m-%dT00:00:00Z")
            end_time = end_date.strftime("%Y-%m-%dT23:59:59Z")
            temporal_desc = "recent (last 6 months)"
        
        elif re.search(r'\blast (year|12 months)\b', text):
            # Last year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            start_time = start_date.strftime("%Y-%m-%dT00:00:00Z") 
            end_time = end_date.strftime("%Y-%m-%dT23:59:59Z")
            temporal_desc = "last year"
        
        return start_time, end_time, temporal_desc

    def _extract_spatial(self, text: str) -> tuple[Optional[Dict[str, float]], Optional[str]]:
        """Extract spatial constraints from text."""
        for pattern, bounds in self.region_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return bounds, match.group(0)
        return None, None

    def _extract_data_keywords(self, text: str) -> List[str]:
        """Extract scientific keywords from text."""
        keywords = []
        for category, terms in self.data_type_keywords.items():
            for term in terms:
                if term in text:
                    keywords.append(term)
        
        # Add any remaining meaningful words
        words = re.findall(r'\b[a-z]{3,}\b', text)
        scientific_words = [w for w in words if w not in {
            'find', 'get', 'search', 'data', 'from', 'over', 'with', 'and', 'the', 'for'
        }]
        keywords.extend(scientific_words[:5])  # Limit to 5 additional words
        
        return list(set(keywords))

    def _extract_data_type_indicators(self, text: str) -> List[str]:
        """Extract specific data type indicators."""
        indicators = []
        
        # Check each category
        for category, terms in self.data_type_keywords.items():
            if any(term in text for term in terms):
                indicators.append(category)
        
        return indicators

    async def _generate_search_variations(
        self,
        original_query: str,
        keywords: List[str],
        platforms: List[str],
        instruments: List[str],
        processing_level: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate multiple search parameter combinations."""
        variations = []
        
        # Base variation with original query keywords
        if keywords:
            base_variation = {
                "keyword": " ".join(keywords[:3]),  # Limit to top 3 keywords
                "platforms": platforms,
                "instruments": instruments,
                "processing_level": processing_level
            }
            variations.append(base_variation)
        
        # Platform-specific variations
        for platform in platforms[:2]:  # Limit to 2 platforms
            variation = {
                "platform": platform,
                "instruments": instruments,
                "processing_level": processing_level
            }
            if keywords:
                variation["keyword"] = " ".join(keywords[:2])
            variations.append(variation)
        
        # Instrument-specific variations
        for instrument in instruments[:2]:  # Limit to 2 instruments
            variation = {
                "instrument": instrument,
                "processing_level": processing_level
            }
            if keywords:
                variation["keyword"] = " ".join(keywords[:2])
            variations.append(variation)
        
        # Use QueryAgent to generate additional search terms
        try:
            query_input = QueryAgentInputSchema(
                query=original_query,
                num_queries=3
            )
            query_output = await self.query_agent.arun(query_input)
            
            # Convert QueryAgent output to search variations
            for i, generated_query in enumerate(query_output.queries[:2]):  # Limit to 2
                variation = {
                    "keyword": generated_query,
                    "processing_level": processing_level
                }
                if platforms and i < len(platforms):
                    variation["platform"] = platforms[i]
                if instruments and i < len(instruments):
                    variation["instrument"] = instruments[i]
                variations.append(variation)
                
        except Exception as e:
            if self.debug:
                logger.warning(f"QueryAgent generation failed: {e}")
        
        # Remove None values and limit total variations
        cleaned_variations = []
        for var in variations[:5]:  # Limit to 5 variations
            cleaned_var = {k: v for k, v in var.items() if v is not None}
            if cleaned_var:  # Only add non-empty variations
                cleaned_variations.append(cleaned_var)
        
        return cleaned_variations