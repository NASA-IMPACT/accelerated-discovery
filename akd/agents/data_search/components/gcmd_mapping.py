"""
GCMD Keyword Mapping Component.

Maps common instrument and platform names to their official GCMD (Global Change Master Directory)
keyword format for accurate CMR searches.
"""

from typing import Dict, List


class GCMDKeywordMapper:
    """Maps common names to official GCMD keywords for NASA Earth science instruments and platforms."""

    # Official GCMD instrument mappings based on CMR searches and NASA documentation
    INSTRUMENT_MAPPINGS = {
        # Precipitation/Weather instruments
        "DPR": "DPR",  # Dual-frequency Precipitation Radar (GPM)
        "GMI": "GMI",  # GPM Microwave Imager
        "TMI": "TMI",  # TRMM Microwave Imager
        "PR": "PR",  # Precipitation Radar (TRMM)
        "VIRS": "VIRS",  # Visible and Infrared Scanner (TRMM)
        # Water surface/Altimetry instruments
        "KaRIn": "Ka-band Radar Interferometer (KaRIn)",  # SWOT
        "KARIN": "Ka-band Radar Interferometer (KaRIn)",  # Alternative spelling
        "SRAL": "SRAL",  # Synthetic Aperture Radar Altimeter (Sentinel-3)
        "POSEIDON-3B": "POSEIDON-3B",  # Jason-3 altimeter
        "DORIS": "DORIS",  # Doppler Orbitography and Radiopositioning Integrated by Satellite
        "GPSP": "GPSP",  # Global Positioning System Payload
        # Land/Ocean imaging instruments
        "MODIS": "MODIS",  # Moderate Resolution Imaging Spectroradiometer
        "VIIRS": "VIIRS",  # Visible Infrared Imaging Radiometer Suite
        "OLI": "OLI",  # Operational Land Imager (Landsat)
        "TIRS": "TIRS",  # Thermal Infrared Sensor (Landsat)
        "MSI": "MSI",  # MultiSpectral Instrument (Sentinel-2)
        "ETM+": "ETM+",  # Enhanced Thematic Mapper Plus (Landsat-7)
        "TM": "TM",  # Thematic Mapper (Landsat)
        # Atmospheric instruments
        "AIRS": "AIRS",  # Atmospheric Infrared Sounder
        "OMI": "OMI",  # Ozone Monitoring Instrument
        "TROPOMI": "TROPOMI",  # TROPOspheric Monitoring Instrument
        "CrIS": "CrIS",  # Cross-track Infrared Sounder
        "OMPS": "OMPS",  # Ozone Mapping and Profiler Suite
        # Ice/Snow instruments
        "AMSR-E": "AMSR-E",  # Advanced Microwave Scanning Radiometer-EOS
        "AMSR2": "AMSR2",  # Advanced Microwave Scanning Radiometer 2
        "AMSR-2": "AMSR2",  # Alternative spelling
        # Ocean color instruments
        "SeaWiFS": "SeaWiFS",  # Sea-viewing Wide Field-of-view Sensor
        "MERIS": "MERIS",  # Medium Resolution Imaging Spectrometer
    }

    # Platform mappings to official GCMD format
    PLATFORM_MAPPINGS = {
        # NASA missions
        "GPM": "GPM",
        "TRMM": "TRMM",
        "SWOT": "SWOT",
        "Terra": "Terra",
        "Aqua": "Aqua",
        "Suomi-NPP": "Suomi-NPP",
        "NOAA-20": "NOAA-20",
        "NOAA-21": "NOAA-21",
        "Landsat-8": "Landsat-8",
        "Landsat-9": "Landsat-9",
        "Landsat-7": "Landsat-7",
        "Landsat-5": "Landsat-5",
        # ESA missions
        "Sentinel-2A": "Sentinel-2A",
        "Sentinel-2B": "Sentinel-2B",
        "Sentinel-3A": "Sentinel-3A",
        "Sentinel-3B": "Sentinel-3B",
        # Altimetry missions
        "Jason-3": "Jason-3",
        "Jason-2": "Jason-2",
        "TOPEX/Poseidon": "TOPEX/Poseidon",
        # Common alternative names
        "SUOMI-NPP": "Suomi-NPP",
        "SNPP": "Suomi-NPP",
        "S2A": "Sentinel-2A",
        "S2B": "Sentinel-2B",
        "S3A": "Sentinel-3A",
        "S3B": "Sentinel-3B",
    }

    # Mission to instrument mappings for when users specify mission instead of instrument
    MISSION_TO_INSTRUMENTS = {
        "GPM": ["DPR", "GMI"],
        "TRMM": ["TMI", "PR", "VIRS"],
        "SWOT": ["Ka-band Radar Interferometer (KaRIn)", "DORIS", "GPSP"],
        "Terra": ["MODIS", "AIRS"],
        "Aqua": ["MODIS", "AIRS"],
        "Landsat-8": ["OLI", "TIRS"],
        "Landsat-9": ["OLI", "TIRS"],
        "Landsat-7": ["ETM+"],
        "Sentinel-2A": ["MSI"],
        "Sentinel-2B": ["MSI"],
        "Sentinel-3A": ["SRAL"],
        "Sentinel-3B": ["SRAL"],
        "Jason-3": ["POSEIDON-3B"],
    }

    # Measurement type to recommended instruments mapping
    MEASUREMENT_TO_INSTRUMENTS = {
        "precipitation": ["DPR", "GMI", "TMI", "PR"],
        "water surface elevation": [
            "Ka-band Radar Interferometer (KaRIn)",
            "SRAL",
            "POSEIDON-3B",
        ],
        "river stage": ["Ka-band Radar Interferometer (KaRIn)", "SRAL"],
        "river discharge": ["Ka-band Radar Interferometer (KaRIn)"],
        "altimetry": ["Ka-band Radar Interferometer (KaRIn)", "SRAL", "POSEIDON-3B"],
        "land cover": ["MODIS", "OLI", "MSI", "ETM+"],
        "vegetation": ["MODIS", "OLI", "MSI", "VIIRS"],
        "ocean color": ["MODIS", "VIIRS", "SeaWiFS"],
        "sea surface temperature": ["MODIS", "VIIRS"],
        "atmospheric": ["AIRS", "OMI", "TROPOMI", "CrIS"],
        "ice": ["MODIS", "VIIRS", "AMSR-E", "AMSR2"],
        "snow": ["MODIS", "VIIRS", "AMSR-E", "AMSR2"],
    }

    @classmethod
    def normalize_instrument_name(cls, instrument: str) -> str:
        """
        Convert a common instrument name to its official GCMD keyword format.

        Args:
            instrument: Common instrument name (case-insensitive)

        Returns:
            Official GCMD instrument keyword, or original if no mapping found
        """
        if not instrument:
            return instrument

        # Try exact match first (case-insensitive)
        instrument_upper = instrument.upper()
        for key, value in cls.INSTRUMENT_MAPPINGS.items():
            if key.upper() == instrument_upper:
                return value

        # Return original if no mapping found
        return instrument

    @classmethod
    def normalize_platform_name(cls, platform: str) -> str:
        """
        Convert a common platform name to its official GCMD keyword format.

        Args:
            platform: Common platform name (case-insensitive)

        Returns:
            Official GCMD platform keyword, or original if no mapping found
        """
        if not platform:
            return platform

        # Try exact match first (case-insensitive)
        platform_upper = platform.upper()
        for key, value in cls.PLATFORM_MAPPINGS.items():
            if key.upper() == platform_upper:
                return value

        return platform

    @classmethod
    def get_instruments_for_mission(cls, mission: str) -> List[str]:
        """
        Get the list of instruments for a given mission/platform.

        Args:
            mission: Mission or platform name

        Returns:
            List of official GCMD instrument names for the mission
        """
        mission_upper = mission.upper()
        for key, instruments in cls.MISSION_TO_INSTRUMENTS.items():
            if key.upper() == mission_upper:
                return instruments
        return []

    @classmethod
    def get_recommended_instruments(cls, measurement_type: str) -> List[str]:
        """
        Get recommended instruments for a specific type of measurement.

        Args:
            measurement_type: Type of measurement (e.g., "precipitation", "water surface elevation")

        Returns:
            List of recommended GCMD instrument names
        """
        measurement_lower = measurement_type.lower()
        for key, instruments in cls.MEASUREMENT_TO_INSTRUMENTS.items():
            if key in measurement_lower or measurement_lower in key:
                return instruments
        return []

    @classmethod
    def suggest_parameters(cls, user_input: str) -> Dict[str, List[str]]:
        """
        Suggest appropriate instrument and platform parameters based on user input.

        Args:
            user_input: User's description of what they're looking for

        Returns:
            Dictionary with 'instruments' and 'platforms' suggestions
        """
        suggestions = {"instruments": [], "platforms": []}
        user_lower = user_input.lower()

        # Check for measurement types
        for measurement, instruments in cls.MEASUREMENT_TO_INSTRUMENTS.items():
            if measurement in user_lower:
                suggestions["instruments"].extend(instruments)

        # Check for explicit mission/platform mentions
        for mission, instruments in cls.MISSION_TO_INSTRUMENTS.items():
            if mission.lower() in user_lower:
                suggestions["instruments"].extend(instruments)
                suggestions["platforms"].append(cls.normalize_platform_name(mission))

        # Remove duplicates while preserving order
        suggestions["instruments"] = list(dict.fromkeys(suggestions["instruments"]))
        suggestions["platforms"] = list(dict.fromkeys(suggestions["platforms"]))

        return suggestions
